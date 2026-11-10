#pragma once

#include <atomic>
#include <cstdio>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.h"
#include "learn/mlp.h"
#include "state.h"
#include "strategy/planner.h"

namespace fw {

// One candidate decision: its network input and the action it turns into.
struct Option {
    std::string label;
    std::vector<float> x;
    nlohmann::json action;  // null for wait / save
    double ratio = 0;       // fraction of our troops (troop options)
    double troops = 0;      // troops it commits
    double gold = 0;        // gold it spends
    bool rulePick = false;
    bool sane = false;      // troop options: a move that is fine to force when troops are full
};

struct TrainReport {
    int episodes = 0;
    size_t samples[3] = {0, 0, 0};
    // Mean squared error on held-out episodes in units of the return's variance:
    // 1.0 = no better than guessing the average, 0 = perfect. Negative = no hold-out set yet.
    double valBefore[3] = {-1, -1, -1};
    double valAfter[3] = {-1, -1, -1};
};

// Value-based decision layer. Three ensembles of networks (troop moves, gold spending, spawn site) score
// every allowed option; the best one is played, with a decaying bonus for what the rules would do, a
// markdown where the ensemble members disagree, and some exploration.
// Decisions go to an episode file; train() turns all episodes into (decision, discounted future gain)
// samples and fits the networks to them. During a game a background thread keeps studying (startStudy).
class Learner {
public:
    Learner(const Config& cfg, std::string modelDir, uint32_t seed = 7);
    ~Learner();

    void load();
    void save() const;

    // Call once per state, before the phases.
    void observe(const State& s);
    // Picks (or late in the spawn phase, reconsiders) the spawn site. False if the state has no candidates.
    bool spawnPhase(Ctx& c);
    void goldPhase(Ctx& c);
    void troopPhase(Ctx& c);
    // End of game: closes the episode with its outcome.
    void gameOver(bool won, bool eliminated);
    // Rebuilds the training set from the episode files and trains all nets.
    TrainReport train(int steps);
    // Background training while the game runs; train() and the destructor stop it.
    void startStudy();
    void stopStudy();

    size_t samples(int net) const { return nets_[net].seen; }
    double priorWeight(int net) const;
    int inputs(int net) const { return nets_[net].members[0].inputs(); }

    // Exposed for tests.
    std::vector<Option> troopOptions(const State& s, const Plan& plan, bool overextended) const;
    std::vector<Option> goldOptions(const State& s, const Plan& plan) const;
    float score(int net, const Option& o) const;  // ensemble mean
    void addSample(int net, std::vector<float> x, float y);
    void trainSteps(int net, int steps);
    // Position value of a snapshot {tiles, landTiles, troops, maxTroops, gold, wealth}.
    double potential(const double snap[6]) const;

    static constexpr int kTroop = 0;
    static constexpr int kGold = 1;
    static constexpr int kSpawn = 2;
    static constexpr int kNets = 3;

private:
    struct Sample {
        std::vector<float> x;
        float y;
    };
    struct Net {
        std::vector<Mlp> members;  // the ensemble being trained
        std::vector<Mlp> act;      // copies the decisions read (guarded by actMu_)
        std::vector<Sample> data;  // training set (filled by train(), the study thread or addSample)
        size_t seen = 0;           // decisions this net has been trained on; drives prior/epsilon decay
        double mean = 0, sd = 1;   // scale of its targets
    };
    struct Trail {  // a decision of this game whose short-term result is still unfolding
        int net, tick;
        double phi;
        std::vector<float> x;
        bool done;
    };
    struct Live {  // ...and once it is known
        int net;
        std::vector<float> x;
        double partial, discount, value;
    };

    void predict(int net, const std::vector<float>& x, double& mean, double& sd) const;
    const Option& choose(int net, std::vector<Option>& opts);
    void record(int net, const Option& pick, const State& s);
    void closeEpisode();
    int loadEpisodes(std::vector<Sample> train[kNets], std::vector<Sample> val[kNets], bool holdOut);
    void trainMember(int net, int member, int steps, float lrStart, float lrEnd, uint32_t seed);
    void publish(int net);
    void studyLive(int net, const std::vector<float>& x, int tick, double phi);
    void studyLoop();

    const Config& cfg_;
    const LearningConfig& lc_;
    std::string dir_;
    Net nets_[kNets];
    State lastState_;
    int lastTick_ = -1;
    double lastBest_[kNets] = {0, 0, 0};  // value of the best option at each net's latest decision
    double lastPick_ = 0;                 // ensemble mean of the option just chosen
    std::FILE* episode_ = nullptr;
    mutable std::mt19937 rng_;

    mutable std::mutex actMu_;
    std::mutex liveMu_;
    std::vector<Live> live_;
    std::deque<Trail> trail_;
    std::thread study_;
    std::atomic<bool> stop_{false};
    bool studying_ = false;
};

}  // namespace fw
