#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace fw::nn {

// A trainable tensor with its gradient and Adam moments.
struct Param {
    std::vector<float> w, g, m, v;
    bool decay = false;  // weight decay applies to matrices, not to biases or norms
    void resize(size_t n, bool decayed) {
        w.assign(n, 0.0f);
        g.assign(n, 0.0f);
        m.assign(n, 0.0f);
        v.assign(n, 0.0f);
        decay = decayed;
    }
};

// y = W x + b for one sample. backward() adds to the parameter gradients and (if given) to dx.
struct Linear {
    int in = 0, out = 0;
    Param W, b;
    void init(int in, int out, std::mt19937& rng, float gain = 1.0f);
    void forward(const float* x, float* y) const;
    void backward(const float* x, const float* dy, float* dx);
};

// y = gamma * (x - mean) / std + beta over one vector.
struct LayerNorm {
    int n = 0;
    Param gamma, beta;
    void init(int n);
    // xhat (n floats) and inv (1/std) are the cache backward() needs.
    void forward(const float* x, float* y, float* xhat, float& inv) const;
    void backward(const float* xhat, float inv, const float* dy, float* dx);
};

struct Shape {
    int self = 0;       // features describing us
    int ent = 0;        // features per visible player
    int opt[2] = {0, 0};  // option features: troop moves, gold spending
    int D = 192;        // trunk width
    int E = 64;         // player embedding width
    int heads = 4;      // attention heads over the players
    int blocks = 3;     // residual blocks in the trunk
    int aux = 5;        // auxiliary predictions
    bool operator==(const Shape&) const = default;
};

constexpr int kMaxEnts = 16;
constexpr int kAux = 5;  // land share in 30 s / 2 min, survive 2 min, attacked within 30 s, gold in 1 min
constexpr unsigned kAuxLogit = 0b01100;  // which of those are probabilities (trained with cross-entropy)

struct Input {
    const float* self = nullptr;
    const float* ents = nullptr;  // nEnts * shape.ent
    int nEnts = 0;
};

struct Target {
    int net = 0;               // 0 troop, 1 gold: which advantage head
    const float* opt = nullptr;
    int target = -1;           // index into ents of the player the option is aimed at, or -1
    float G = 0;               // standardised return
    // What the position alone promised, estimated by a model that never trained on this game. The
    // advantage head learns G - baseline. Without one, this model's own V is used, which only works
    // while V cannot memorise its training positions.
    bool hasBaseline = false;
    float baseline = 0;
    const float* aux = nullptr;
    unsigned auxMask = 0;      // which aux targets are known
};

struct Losses {
    float value = 0, advantage = 0, aux = 0;
};

// The v4 network: our state and every visible player are encoded separately, the players are pooled
// by attention keyed on our state, a residual trunk mixes them, and heads read the result:
// V(s), auxiliary forecasts, and one advantage head per decision type scoring a single option.
// Not thread-safe: one instance per thread.
class Model {
public:
    Model() = default;
    Model(const Shape& shape, uint32_t seed);

    const Shape& shape() const { return shape_; }
    size_t parameters() const;

    // Inference: encode a state once, then read the heads / score any number of options.
    void encode(const Input& in);
    float value() const { return value_; }
    const float* aux() const { return auxOut_.data(); }
    float advantage(int net, const float* opt, int target);

    // Training: forward + backward for one sample, gradients accumulate until step().
    Losses accumulate(const Input& in, const Target& t);
    void step(float lr, float weightDecay, int batch);
    void zeroGrad();

    bool save(const std::string& path) const;
    bool load(const std::string& path);  // false (and unchanged) if missing or a different shape

    // For gradient-check tests.
    std::vector<Param*> params();
    bool detachValue = true;  // the advantage target uses V(s) as a constant
    float huber = 2.0f;
    float auxWeight = 0.25f;

private:
    struct Block {
        LayerNorm norm;
        Linear a, b;
    };
    struct Head {
        Linear l1, l2, l3;
    };
    void advForward(int net, const float* opt, int target);

    Shape shape_;
    Linear selfEnc_, ent1_, ent2_, query_, mix_, valueHead_, auxHead_;
    std::vector<Block> blocks_;
    Head adv_[2];
    int64_t step_ = 0;

    // Forward caches (also what backward reads).
    int nEnts_ = 0;
    std::vector<float> selfIn_, entIn_, s0_, h1_, e_, q_, att_, cat_, t_, xhat_, u_, advIn_, a1_, a2_, auxOut_;
    std::vector<float> inv_;
    float value_ = 0, adv_out_ = 0;
};

}  // namespace fw::nn
