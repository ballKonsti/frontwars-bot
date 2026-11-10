#include "learn/model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace fw::nn {

namespace {

constexpr char kMagic[8] = {'F', 'W', 'M', 'D', '0', '0', '0', '4'};
constexpr float kBeta1 = 0.9f, kBeta2 = 0.999f, kEps = 1e-8f, kNormEps = 1e-5f;
constexpr int kAdv1 = 128, kAdv2 = 64;

inline void relu(float* x, int n) {
    for (int i = 0; i < n; ++i) x[i] = std::max(0.0f, x[i]);
}
// dy *= (y > 0)
inline void reluBack(const float* y, float* dy, int n) {
    for (int i = 0; i < n; ++i)
        if (y[i] <= 0.0f) dy[i] = 0.0f;
}
inline float huberLoss(float e, float d) { return std::fabs(e) <= d ? 0.5f * e * e : d * (std::fabs(e) - 0.5f * d); }

}  // namespace

void Linear::init(int in_, int out_, std::mt19937& rng, float gain) {
    in = in_;
    out = out_;
    W.resize(size_t(in) * out, true);
    b.resize(out, false);
    std::normal_distribution<float> dist(0.0f, gain * std::sqrt(2.0f / float(in)));  // He init
    for (auto& w : W.w) w = dist(rng);
}

void Linear::forward(const float* x, float* y) const {
    for (int o = 0; o < out; ++o) {
        const float* row = &W.w[size_t(o) * in];
        float sum = b.w[o];
        for (int i = 0; i < in; ++i) sum += row[i] * x[i];
        y[o] = sum;
    }
}

void Linear::backward(const float* x, const float* dy, float* dx) {
    for (int o = 0; o < out; ++o) {
        const float d = dy[o];
        if (d == 0.0f) continue;
        b.g[o] += d;
        float* g = &W.g[size_t(o) * in];
        for (int i = 0; i < in; ++i) g[i] += d * x[i];
        if (dx) {
            const float* row = &W.w[size_t(o) * in];
            for (int i = 0; i < in; ++i) dx[i] += row[i] * d;
        }
    }
}

void LayerNorm::init(int n_) {
    n = n_;
    gamma.resize(n, false);
    beta.resize(n, false);
    std::fill(gamma.w.begin(), gamma.w.end(), 1.0f);
}

void LayerNorm::forward(const float* x, float* y, float* xhat, float& inv) const {
    float mean = 0, var = 0;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= float(n);
    for (int i = 0; i < n; ++i) var += (x[i] - mean) * (x[i] - mean);
    inv = 1.0f / std::sqrt(var / float(n) + kNormEps);
    for (int i = 0; i < n; ++i) {
        xhat[i] = (x[i] - mean) * inv;
        y[i] = gamma.w[i] * xhat[i] + beta.w[i];
    }
}

void LayerNorm::backward(const float* xhat, float inv, const float* dy, float* dx) {
    float sum = 0, dot = 0;
    for (int i = 0; i < n; ++i) {
        gamma.g[i] += dy[i] * xhat[i];
        beta.g[i] += dy[i];
        const float d = dy[i] * gamma.w[i];
        sum += d;
        dot += d * xhat[i];
    }
    sum /= float(n);
    dot /= float(n);
    for (int i = 0; i < n; ++i) dx[i] += (dy[i] * gamma.w[i] - sum - xhat[i] * dot) * inv;
}

Model::Model(const Shape& shape, uint32_t seed) : shape_(shape) {
    std::mt19937 rng(seed);
    const int D = shape.D, E = shape.E, H = shape.heads;
    selfEnc_.init(shape.self, D, rng);
    ent1_.init(shape.ent, E, rng);
    ent2_.init(E, E, rng);
    query_.init(D, H * E, rng, 0.5f);
    mix_.init(D + H * E, D, rng);
    blocks_.resize(shape.blocks);
    for (auto& b : blocks_) {
        b.norm.init(D);
        b.a.init(D, D, rng);
        b.b.init(D, D, rng, 0.1f);  // blocks start close to the identity
    }
    valueHead_.init(D, 1, rng, 0.05f);  // untrained heads answer ~0
    auxHead_.init(D, shape.aux, rng, 0.05f);
    for (int net = 0; net < 2; ++net) {
        adv_[net].l1.init(D + shape.opt[net] + E, kAdv1, rng);
        adv_[net].l2.init(kAdv1, kAdv2, rng);
        adv_[net].l3.init(kAdv2, 1, rng, 0.05f);
    }
    selfIn_.resize(shape.self);
    entIn_.resize(size_t(kMaxEnts) * shape.ent);
    s0_.resize(D);
    h1_.resize(size_t(kMaxEnts) * E);
    e_.resize(size_t(kMaxEnts) * E);
    q_.resize(size_t(H) * E);
    att_.resize(size_t(H) * kMaxEnts);
    cat_.resize(D + size_t(H) * E);
    t_.resize(size_t(shape.blocks + 1) * D);
    xhat_.resize(size_t(shape.blocks) * D);
    u_.resize(size_t(shape.blocks) * D);
    inv_.resize(shape.blocks);
    advIn_.resize(D + std::max(shape.opt[0], shape.opt[1]) + E);
    a1_.resize(kAdv1);
    a2_.resize(kAdv2);
    auxOut_.resize(shape.aux);
}

std::vector<Param*> Model::params() {
    std::vector<Param*> p;
    auto lin = [&](Linear& l) {
        p.push_back(&l.W);
        p.push_back(&l.b);
    };
    lin(selfEnc_);
    lin(ent1_);
    lin(ent2_);
    lin(query_);
    lin(mix_);
    for (auto& b : blocks_) {
        p.push_back(&b.norm.gamma);
        p.push_back(&b.norm.beta);
        lin(b.a);
        lin(b.b);
    }
    lin(valueHead_);
    lin(auxHead_);
    for (auto& h : adv_) {
        lin(h.l1);
        lin(h.l2);
        lin(h.l3);
    }
    return p;
}

size_t Model::parameters() const {
    size_t n = 0;
    for (Param* p : const_cast<Model*>(this)->params()) n += p->w.size();
    return n;
}

void Model::encode(const Input& in) {
    const int D = shape_.D, E = shape_.E, H = shape_.heads, F = shape_.ent;
    nEnts_ = std::clamp(in.nEnts, 0, kMaxEnts);
    std::copy(in.self, in.self + shape_.self, selfIn_.begin());
    if (nEnts_) std::copy(in.ents, in.ents + size_t(nEnts_) * F, entIn_.begin());

    selfEnc_.forward(selfIn_.data(), s0_.data());
    relu(s0_.data(), D);
    for (int i = 0; i < nEnts_; ++i) {
        ent1_.forward(&entIn_[size_t(i) * F], &h1_[size_t(i) * E]);
        relu(&h1_[size_t(i) * E], E);
        ent2_.forward(&h1_[size_t(i) * E], &e_[size_t(i) * E]);
        relu(&e_[size_t(i) * E], E);
    }

    // Attention pooling: each head asks its own question of the players, given who we are.
    query_.forward(s0_.data(), q_.data());
    std::copy(s0_.begin(), s0_.end(), cat_.begin());
    float* pooled = &cat_[D];
    std::fill(pooled, pooled + size_t(H) * E, 0.0f);
    const float scale = 1.0f / std::sqrt(float(E));
    for (int h = 0; h < H && nEnts_ > 0; ++h) {
        const float* q = &q_[size_t(h) * E];
        float* a = &att_[size_t(h) * kMaxEnts];
        float top = -1e30f;
        for (int i = 0; i < nEnts_; ++i) {
            const float* e = &e_[size_t(i) * E];
            float dot = 0;
            for (int k = 0; k < E; ++k) dot += q[k] * e[k];
            a[i] = dot * scale;
            top = std::max(top, a[i]);
        }
        float sum = 0;
        for (int i = 0; i < nEnts_; ++i) sum += (a[i] = std::exp(a[i] - top));
        for (int i = 0; i < nEnts_; ++i) {
            a[i] /= sum;
            const float* e = &e_[size_t(i) * E];
            float* p = &pooled[size_t(h) * E];
            for (int k = 0; k < E; ++k) p[k] += a[i] * e[k];
        }
    }

    float* t = t_.data();
    mix_.forward(cat_.data(), t);
    relu(t, D);
    for (int b = 0; b < shape_.blocks; ++b) {
        Block& blk = blocks_[b];
        float* next = t + D;
        float* u = &u_[size_t(b) * D];
        // next is used as scratch for the normalised input before it becomes the block's output.
        blk.norm.forward(t, next, &xhat_[size_t(b) * D], inv_[b]);
        std::vector<float>& normed = advIn_;  // reuse a buffer that is at least D wide
        std::copy(next, next + D, normed.begin());
        blk.a.forward(normed.data(), u);
        relu(u, D);
        blk.b.forward(u, next);
        for (int i = 0; i < D; ++i) next[i] += t[i];
        t = next;
    }
    valueHead_.forward(t, &value_);
    auxHead_.forward(t, auxOut_.data());
}

void Model::advForward(int net, const float* opt, int target) {
    const int D = shape_.D, E = shape_.E, F = shape_.opt[net];
    const float* trunk = &t_[size_t(shape_.blocks) * D];
    float* in = advIn_.data();
    std::copy(trunk, trunk + D, in);
    std::copy(opt, opt + F, in + D);
    if (target >= 0 && target < nEnts_)
        std::copy(&e_[size_t(target) * E], &e_[size_t(target) * E] + E, in + D + F);
    else
        std::fill(in + D + F, in + D + F + E, 0.0f);
    Head& h = adv_[net];
    h.l1.forward(in, a1_.data());
    relu(a1_.data(), kAdv1);
    h.l2.forward(a1_.data(), a2_.data());
    relu(a2_.data(), kAdv2);
    h.l3.forward(a2_.data(), &adv_out_);
}

float Model::advantage(int net, const float* opt, int target) {
    advForward(net, opt, target);
    return adv_out_;
}

Losses Model::accumulate(const Input& in, const Target& tg) {
    const int D = shape_.D, E = shape_.E, H = shape_.heads, F = shape_.ent, FO = shape_.opt[tg.net];
    encode(in);
    advForward(tg.net, tg.opt, tg.target);

    Losses loss;
    // Heads: V learns the return, the advantage head what is left of it after V.
    const float ev = value_ - tg.G;
    const float ea = adv_out_ - (tg.G - (tg.hasBaseline ? tg.baseline : value_));
    loss.value = huberLoss(ev, huber);
    loss.advantage = huberLoss(ea, huber);
    const float dA = std::clamp(ea, -huber, huber);
    float dV = std::clamp(ev, -huber, huber) + (detachValue || tg.hasBaseline ? 0.0f : dA);

    static thread_local std::vector<float> dT, dTn, dU, dCat, dS0, dE, dQ, dA1, dA2, dIn, dH1, dAux;
    dT.assign(D, 0.0f);
    dE.assign(size_t(kMaxEnts) * E, 0.0f);

    // Advantage head.
    Head& h = adv_[tg.net];
    dA2.assign(kAdv2, 0.0f);
    h.l3.backward(a2_.data(), &dA, dA2.data());
    reluBack(a2_.data(), dA2.data(), kAdv2);
    dA1.assign(kAdv1, 0.0f);
    h.l2.backward(a1_.data(), dA2.data(), dA1.data());
    reluBack(a1_.data(), dA1.data(), kAdv1);
    dIn.assign(D + FO + E, 0.0f);
    h.l1.backward(advIn_.data(), dA1.data(), dIn.data());
    for (int i = 0; i < D; ++i) dT[i] += dIn[i];
    if (tg.target >= 0 && tg.target < nEnts_)
        for (int k = 0; k < E; ++k) dE[size_t(tg.target) * E + k] += dIn[D + FO + k];

    // Value and auxiliary heads.
    const float* trunk = &t_[size_t(shape_.blocks) * D];
    valueHead_.backward(trunk, &dV, dT.data());
    dAux.assign(shape_.aux, 0.0f);
    for (int k = 0; k < shape_.aux; ++k) {
        if (!tg.aux || !(tg.auxMask >> k & 1)) continue;
        if (kAuxLogit >> k & 1) {
            const float p = 1.0f / (1.0f + std::exp(-auxOut_[k]));
            const float y = tg.aux[k];
            loss.aux += -auxWeight * (y * std::log(p + 1e-7f) + (1 - y) * std::log(1 - p + 1e-7f));
            dAux[k] = auxWeight * (p - y);
        } else {
            const float e = auxOut_[k] - tg.aux[k];
            loss.aux += auxWeight * 0.5f * e * e;
            dAux[k] = auxWeight * e;
        }
    }
    auxHead_.backward(trunk, dAux.data(), dT.data());

    // Trunk, last block first. The block computed next = t + B(relu(A(norm(t)))).
    static thread_local std::vector<float> normed;
    normed.resize(D);
    for (int b = shape_.blocks; b-- > 0;) {
        Block& blk = blocks_[b];
        const float* u = &u_[size_t(b) * D];
        const float* xhat = &xhat_[size_t(b) * D];
        dU.assign(D, 0.0f);
        blk.b.backward(u, dT.data(), dU.data());
        reluBack(u, dU.data(), D);
        for (int i = 0; i < D; ++i) normed[i] = blk.norm.gamma.w[i] * xhat[i] + blk.norm.beta.w[i];
        dTn.assign(D, 0.0f);
        blk.a.backward(normed.data(), dU.data(), dTn.data());
        blk.norm.backward(xhat, inv_[b], dTn.data(), dT.data());  // adds to the skip path's gradient
    }
    reluBack(t_.data(), dT.data(), D);
    dCat.assign(D + size_t(H) * E, 0.0f);
    mix_.backward(cat_.data(), dT.data(), dCat.data());

    // Attention pooling.
    dS0.assign(dCat.begin(), dCat.begin() + D);
    dQ.assign(size_t(H) * E, 0.0f);
    const float scale = 1.0f / std::sqrt(float(E));
    for (int hd = 0; hd < H && nEnts_ > 0; ++hd) {
        const float* dP = &dCat[D + size_t(hd) * E];
        const float* q = &q_[size_t(hd) * E];
        const float* a = &att_[size_t(hd) * kMaxEnts];
        float da[kMaxEnts], mean = 0;
        for (int i = 0; i < nEnts_; ++i) {
            const float* e = &e_[size_t(i) * E];
            float dot = 0;
            for (int k = 0; k < E; ++k) dot += dP[k] * e[k];
            da[i] = dot;
            mean += a[i] * dot;
        }
        for (int i = 0; i < nEnts_; ++i) {
            const float dl = a[i] * (da[i] - mean) * scale;  // through the softmax and the 1/sqrt(E)
            const float* e = &e_[size_t(i) * E];
            float* de = &dE[size_t(i) * E];
            float* dq = &dQ[size_t(hd) * E];
            for (int k = 0; k < E; ++k) {
                de[k] += a[i] * dP[k] + dl * q[k];
                dq[k] += dl * e[k];
            }
        }
    }
    query_.backward(s0_.data(), dQ.data(), dS0.data());

    // Encoders.
    for (int i = 0; i < nEnts_; ++i) {
        float* de = &dE[size_t(i) * E];
        reluBack(&e_[size_t(i) * E], de, E);
        dH1.assign(E, 0.0f);
        ent2_.backward(&h1_[size_t(i) * E], de, dH1.data());
        reluBack(&h1_[size_t(i) * E], dH1.data(), E);
        ent1_.backward(&entIn_[size_t(i) * F], dH1.data(), nullptr);
    }
    reluBack(s0_.data(), dS0.data(), D);
    selfEnc_.backward(selfIn_.data(), dS0.data(), nullptr);
    return loss;
}

void Model::zeroGrad() {
    for (Param* p : params()) std::fill(p->g.begin(), p->g.end(), 0.0f);
}

void Model::step(float lr, float weightDecay, int batch) {
    ++step_;
    const float inv = 1.0f / float(std::max(1, batch));
    const float c1 = 1.0f - std::pow(kBeta1, float(step_));
    const float c2 = 1.0f - std::pow(kBeta2, float(step_));
    for (Param* p : params()) {
        const float wd = p->decay ? lr * weightDecay : 0.0f;
        for (size_t i = 0; i < p->w.size(); ++i) {
            const float g = p->g[i] * inv;
            p->m[i] = kBeta1 * p->m[i] + (1 - kBeta1) * g;
            p->v[i] = kBeta2 * p->v[i] + (1 - kBeta2) * g * g;
            p->w[i] -= lr * (p->m[i] / c1) / (std::sqrt(p->v[i] / c2) + kEps) + wd * p->w[i];
            p->g[i] = 0.0f;
        }
    }
}

bool Model::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(kMagic, sizeof kMagic);
    out.write(reinterpret_cast<const char*>(&shape_), sizeof shape_);
    out.write(reinterpret_cast<const char*>(&step_), sizeof step_);
    for (Param* p : const_cast<Model*>(this)->params())
        for (const auto* v : {&p->w, &p->m, &p->v})
            out.write(reinterpret_cast<const char*>(v->data()), std::streamsize(sizeof(float) * v->size()));
    return bool(out);
}

bool Model::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[sizeof kMagic];
    Shape shape;
    in.read(magic, sizeof magic);
    in.read(reinterpret_cast<char*>(&shape), sizeof shape);
    if (!in || std::memcmp(magic, kMagic, sizeof kMagic) != 0 || !(shape == shape_)) return false;
    Model copy = *this;
    in.read(reinterpret_cast<char*>(&copy.step_), sizeof copy.step_);
    for (Param* p : copy.params())
        for (auto* v : {&p->w, &p->m, &p->v})
            in.read(reinterpret_cast<char*>(v->data()), std::streamsize(sizeof(float) * v->size()));
    if (!in) return false;
    *this = std::move(copy);
    return true;
}

}  // namespace fw::nn
