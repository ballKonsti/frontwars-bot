#include "learn/mlp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace fw {

namespace {
constexpr char kMagic[8] = {'F', 'W', 'N', 'N', '0', '0', '0', '1'};
constexpr float kBeta1 = 0.9f, kBeta2 = 0.999f, kEps = 1e-8f;
}  // namespace

Mlp::Mlp(std::vector<int> sizes, uint32_t seed) : sizes_(std::move(sizes)) {
    std::mt19937 rng(seed);
    for (size_t l = 0; l + 1 < sizes_.size(); ++l) {
        Layer L;
        L.in = sizes_[l];
        L.out = sizes_[l + 1];
        // He init for ReLU layers; small output layer so an untrained net scores ~0.
        float scale = l + 2 == sizes_.size() ? 0.01f : std::sqrt(2.0f / L.in);
        std::normal_distribution<float> dist(0.0f, scale);
        L.w.resize(size_t(L.in) * L.out);
        for (auto& w : L.w) w = dist(rng);
        L.b.assign(L.out, 0.0f);
        L.mw.assign(L.w.size(), 0.0f);
        L.vw.assign(L.w.size(), 0.0f);
        L.mb.assign(L.out, 0.0f);
        L.vb.assign(L.out, 0.0f);
        layers_.push_back(std::move(L));
    }
}

void Mlp::forward(const std::vector<float>& x, std::vector<std::vector<float>>& acts) const {
    acts.resize(layers_.size() + 1);
    acts[0] = x;
    for (size_t l = 0; l < layers_.size(); ++l) {
        const Layer& L = layers_[l];
        auto& out = acts[l + 1];
        out.assign(L.out, 0.0f);
        const auto& in = acts[l];
        for (int o = 0; o < L.out; ++o) {
            float sum = L.b[o];
            const float* row = &L.w[size_t(o) * L.in];
            for (int i = 0; i < L.in; ++i) sum += row[i] * in[i];
            out[o] = (l + 1 < layers_.size()) ? std::max(0.0f, sum) : sum;
        }
    }
}

float Mlp::predict(const std::vector<float>& x) const {
    if (layers_.empty() || int(x.size()) != inputs()) return 0.0f;
    std::vector<std::vector<float>> acts;
    forward(x, acts);
    return acts.back()[0];
}

float Mlp::train(const std::vector<const std::vector<float>*>& xs, const std::vector<float>& ys, float lr,
                 float huber) {
    if (layers_.empty() || xs.empty()) return 0.0f;
    // Scratch buffers are kept between calls: allocation would otherwise dominate small batches.
    static thread_local std::vector<std::vector<float>> gw, gb, acts;
    static thread_local std::vector<float> delta, prev;
    gw.resize(layers_.size());
    gb.resize(layers_.size());
    for (size_t l = 0; l < layers_.size(); ++l) {
        gw[l].assign(layers_[l].w.size(), 0.0f);
        gb[l].assign(layers_[l].out, 0.0f);
    }

    float loss = 0.0f;
    for (size_t n = 0; n < xs.size(); ++n) {
        forward(*xs[n], acts);
        float err = acts.back()[0] - ys[n];
        loss += err * err;
        delta.assign(1, 2.0f * (huber > 0 ? std::clamp(err, -huber, huber) : err));
        for (size_t l = layers_.size(); l-- > 0;) {
            const Layer& L = layers_[l];
            const auto& in = acts[l];
            for (int o = 0; o < L.out; ++o) {
                const float d = delta[o];
                if (d == 0.0f) continue;  // dead ReLU units: nothing to add
                gb[l][o] += d;
                float* g = &gw[l][size_t(o) * L.in];
                const float* x = in.data();
                for (int i = 0; i < L.in; ++i) g[i] += d * x[i];
            }
            if (l == 0) break;
            prev.assign(L.in, 0.0f);
            for (int o = 0; o < L.out; ++o) {
                const float d = delta[o];
                if (d == 0.0f) continue;
                const float* row = &L.w[size_t(o) * L.in];
                float* pv = prev.data();
                for (int i = 0; i < L.in; ++i) pv[i] += row[i] * d;
            }
            for (int i = 0; i < L.in; ++i)
                if (in[i] <= 0.0f) prev[i] = 0.0f;  // ReLU gradient
            delta.swap(prev);
        }
    }

    ++step_;
    const float inv = 1.0f / float(xs.size());
    const float c1 = 1.0f - std::pow(kBeta1, float(step_));
    const float c2 = 1.0f - std::pow(kBeta2, float(step_));
    auto adam = [&](float& p, float g, float& m, float& v) {
        g *= inv;
        m = kBeta1 * m + (1 - kBeta1) * g;
        v = kBeta2 * v + (1 - kBeta2) * g * g;
        p -= lr * (m / c1) / (std::sqrt(v / c2) + kEps);
    };
    for (size_t l = 0; l < layers_.size(); ++l) {
        Layer& L = layers_[l];
        for (size_t k = 0; k < L.w.size(); ++k) adam(L.w[k], gw[l][k], L.mw[k], L.vw[k]);
        for (int o = 0; o < L.out; ++o) adam(L.b[o], gb[l][o], L.mb[o], L.vb[o]);
    }
    return loss * inv;
}

bool Mlp::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(kMagic, sizeof kMagic);
    int32_t n = int32_t(sizes_.size());
    out.write(reinterpret_cast<const char*>(&n), sizeof n);
    out.write(reinterpret_cast<const char*>(sizes_.data()), sizeof(int) * sizes_.size());
    out.write(reinterpret_cast<const char*>(&step_), sizeof step_);
    for (const auto& L : layers_)
        for (const auto* v : {&L.w, &L.b, &L.mw, &L.vw, &L.mb, &L.vb})
            out.write(reinterpret_cast<const char*>(v->data()), sizeof(float) * v->size());
    return bool(out);
}

bool Mlp::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[sizeof kMagic];
    in.read(magic, sizeof magic);
    if (!in || std::memcmp(magic, kMagic, sizeof kMagic) != 0) return false;
    int32_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof n);
    if (!in || n != int32_t(sizes_.size())) return false;
    std::vector<int> sizes(n);
    in.read(reinterpret_cast<char*>(sizes.data()), sizeof(int) * n);
    if (!in || sizes != sizes_) return false;

    Mlp copy = *this;
    in.read(reinterpret_cast<char*>(&copy.step_), sizeof copy.step_);
    for (auto& L : copy.layers_)
        for (auto* v : {&L.w, &L.b, &L.mw, &L.vw, &L.mb, &L.vb})
            in.read(reinterpret_cast<char*>(v->data()), sizeof(float) * v->size());
    if (!in) return false;
    *this = std::move(copy);
    return true;
}

}  // namespace fw
