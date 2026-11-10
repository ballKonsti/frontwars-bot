#pragma once

#include <functional>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.h"
#include "learn/episodes.h"
#include "learn/mlp.h"
#include "learn/model.h"
#include "state.h"
#include "strategy/planner.h"

namespace fw {

// One candidate decision: its option features and the action it turns into.
struct Option {
    std::string label;
    std::vector<float> x;   // features of the option itself (the state is encoded separately)
    int target = -1;        // index into Features::ents of the player it is aimed at
    nlohmann::json action;  // null for wait / save
    double ratio = 0;       // fraction of our troops (troop options); rule score (spawn options)
    double troops = 0;      // troops it commits
    double gold = 0;        // gold it spends
    bool rulePick = false;
    bool sane = false;      // troop options: a move that is fine to force when troops are full
    bool reckless = false;  // never picked by exploration
};

// What the networks see of a state: our own features and one row per visible player.
struct Features {
    std::vector<float> self, ents;
    int nEnts = 0;
    std::vector<int> smalls;  // small ID of each row in ents
    int index(int small) const;
};

Dims learnerDims();
nn::Shape modelShape(const LearningConfig& lc);

// The weights: an ensemble of v4 models for troop and gold decisions and a small ensemble for the spawn
// site, stored in <modelDir>/v4-<size>/. The trainer writes them, engines read them.
struct Brain {
    Brain(const LearningConfig& lc, const std::string& modelDir, uint32_t seed);
    bool load();  // false (and unchanged) unless a complete, matching set is on disk
    void save(const nlohmann::json& extra = nullptr) const;  // extra fields go into status.json

    std::string dir;
    std::vector<nn::Model> members;
    std::vector<Mlp> spawn;
    size_t seen = 0, spawnSeen = 0;  // decisions trained on; drives prior and exploration decay
    double mean = 0, sd = 1;         // scale of the returns the models were trained on
    double spawnMean = 0, spawnSd = 1;
};

// The playing side of the neural layer. Each decision tick it scores every allowed option with the
// ensemble and plays the best, with a decaying bonus for what the rules would do, a markdown where the
// members disagree, and some exploration. Decisions are written to episode files; fw_trainer learns
// from them and publishes new weights, which are picked up while the game runs.
class Learner {
public:
    Learner(const Config& cfg, std::string modelDir, uint32_t seed = 7);
    ~Learner();

    void load();

    // Call once per state, before the phases.
    void observe(const State& s);
    // Picks (or late in the spawn phase, reconsiders) the spawn site. False if the state has no candidates.
    bool spawnPhase(Ctx& c);
    void goldPhase(Ctx& c);
    void troopPhase(Ctx& c);
    // End of game: closes the episode with its outcome.
    void gameOver(bool won, bool eliminated);

    double priorWeight(int net) const;
    int reloads() const { return reloads_; }
    Brain& brain() { return brain_; }
    // Extra score per option from outside the nets (research advice). Net: kTroop / kGold / kSpawn.
    void setAdvice(std::function<double(int, const Option&)> fn) { advice_ = std::move(fn); }

    // Exposed for tests.
    Features features(const State& s, bool overextended) const;
    std::vector<Option> troopOptions(const State& s, const Plan& plan, bool overextended) const;
    std::vector<Option> goldOptions(const State& s, const Plan& plan) const;
    float score(int net, const Features& ft, const Option& o);  // ensemble mean

    static constexpr int kTroop = 0;
    static constexpr int kGold = 1;
    static constexpr int kSpawn = 2;

private:
    void scoreAll(int net, const Features& ft, const std::vector<Option>& opts, std::vector<double>& mean,
                  std::vector<double>& sd);
    const Option& choose(int net, const Features& ft, std::vector<Option>& opts);
    void record(int net, const Features& ft, const Option& pick, const State& s);
    void snapshot(const State& s, double out[kSnap]) const;
    void reloadIfNewer();

    const Config& cfg_;
    const LearningConfig& lc_;
    std::string dir_;
    Brain brain_;
    EpisodeWriter writer_;
    std::function<double(int, const Option&)> advice_;
    State lastState_;
    int lastTick_ = -1;
    double lastGold_ = -1, earned_ = 0, income_ = 0;  // gold earned so far, and per tick (smoothed)
    double lastPick_ = 0, lastSpread_ = 0;            // ensemble mean / spread of the option just chosen
    long long statusTime_ = 0;
    int reloads_ = 0;
    std::mt19937 rng_;
};

}  // namespace fw
