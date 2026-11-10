#pragma once

#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "config.h"
#include "learn/episodes.h"
#include "learn/learner.h"

namespace fw {

struct TrainStatus {
    int episodes = 0, running = 0;      // episode files used; how many of them are unfinished games
    size_t samples[3] = {0, 0, 0};      // usable troop / gold / spawn decisions
    size_t heldOut = 0;
    // Measured on held-out games (negative = no held-out games yet). Errors are relative: 1.0 = knows
    // nothing, 0 = perfect. valueError: how well V predicts the return. moveError: how well the
    // advantage heads predict what is left after V. movePairs: of two held-out decisions, how often the
    // one scored higher really did better than its position promised (50% = chance).
    double valueError = -1, moveError = -1, movePairs = -1;
    long steps = 0;
    double stepsPerSecond = 0;
};

// The learning side: reads every episode file, turns decisions into training targets, and trains the
// ensemble. Runs inside fw_trainer for as long as games are being played.
class Trainer {
public:
    Trainer(const Config& cfg, std::string modelDir, uint32_t seed = 11);

    Brain& brain() { return brain_; }
    // Re-reads episode files that changed and rebuilds the training set. Returns usable decisions.
    size_t refresh();
    // `steps` minibatch steps for every ensemble member, members in parallel.
    void trainSteps(int steps);
    TrainStatus evaluate();
    void save();  // weights + status.json (atomic)
    const TrainStatus& status() const { return status_; }

private:
    struct Cached {
        std::filesystem::file_time_type time;
        std::uintmax_t size = 0;
        bool ok = false;
        Episode ep;
    };
    void trainMember(int member, int steps, uint32_t seed, const std::vector<nn::Model>* frozen);
    void trainSpawn(int steps);
    float normG(const Decision& d) const { return float((d.G - brain_.mean) / brain_.sd); }

    const Config& cfg_;
    const LearningConfig& lc_;
    std::string dir_;
    Brain brain_;
    std::map<std::string, Cached> files_;
    std::vector<const Decision*> train_, val_, spawn_;
    std::vector<uint8_t> fold_;  // per train_ entry: the ensemble member that never sees its game
    TrainStatus status_;
    std::mt19937 rng_;
};

}  // namespace fw
