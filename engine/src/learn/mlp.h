#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace fw {

// Small fully connected net: ReLU hidden layers, linear scalar output, Adam on MSE.
// Sized for a few thousand weights, so plain loops are fast enough.
class Mlp {
public:
    Mlp() = default;
    Mlp(std::vector<int> sizes, uint32_t seed = 1);

    float predict(const std::vector<float>& x) const;
    // One Adam step on a minibatch. Returns the mean squared error before the step.
    // huber > 0 clips each sample's error gradient at that size, so a few wild returns can't wreck the net.
    float train(const std::vector<const std::vector<float>*>& xs, const std::vector<float>& ys, float lr,
                float huber = 0.0f);

    int inputs() const { return sizes_.empty() ? 0 : sizes_.front(); }
    bool empty() const { return sizes_.empty(); }

    bool save(const std::string& path) const;
    bool load(const std::string& path);  // false (and unchanged) if missing or a different shape

private:
    struct Layer {
        int in = 0, out = 0;
        std::vector<float> w, b;   // w is out x in, row-major
        std::vector<float> mw, vw, mb, vb;
    };

    void forward(const std::vector<float>& x, std::vector<std::vector<float>>& acts) const;

    std::vector<int> sizes_;
    std::vector<Layer> layers_;
    int64_t step_ = 0;
};

}  // namespace fw
