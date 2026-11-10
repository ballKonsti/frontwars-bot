#pragma once

#include <algorithm>
#include <deque>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.h"
#include "state.h"

namespace fw {

// Persists across ticks: cooldowns, troop history, and who has nuked us.
struct Memory {
    std::unordered_map<std::string, int> lastTick;
    std::deque<std::pair<int, double>> troopHistory;  // (tick, troops)
    std::set<int> nukedBy;                            // small IDs
    double spawnHeur = 0, spawnScore = 0;             // what the learner thought of the site it spawned on

    bool ready(const std::string& key, int tick, int cooldown) const {
        auto it = lastTick.find(key);
        return it == lastTick.end() || tick - it->second >= cooldown;
    }
    void mark(const std::string& key, int tick) { lastTick[key] = tick; }
};

// One tick's decisions. Modules draw troops and gold from shared budgets so they don't double-spend.
struct Plan {
    std::vector<nlohmann::json> actions;
    std::vector<std::string> log;
    double troopBudget = 0;  // troops above the home reserve
    double goldBudget = 0;
    bool overextended = false;  // losing troops faster than we regenerate: no new offensives

    double commit(double troops) {
        double t = std::min(troops, troopBudget);
        troopBudget -= t;
        return t;
    }
    bool spend(double gold) {
        if (gold > goldBudget) return false;
        goldBudget -= gold;
        return true;
    }
};

struct Ctx {
    const State& s;
    const Config& cfg;
    Memory& mem;
    Plan& plan;
};

// Modules, in the order the planner runs them.
void spawnModule(Ctx& c);
void diplomacyModule(Ctx& c);
void strategicModule(Ctx& c);  // = strategicBuildModule + nukeModule
void strategicBuildModule(Ctx& c);
void nukeModule(Ctx& c);
void economyModule(Ctx& c);
void militaryModule(Ctx& c);
void expandModule(Ctx& c);
void navalModule(Ctx& c);

// Troops that must stay home this tick.
double homeReserve(const State& s, const Config& cfg);

// Records this tick's troops and reports whether offensives are bleeding us dry.
bool overextended(const State& s, const SustainConfig& cfg, Memory& mem);

class Learner;

// Runs all modules for one state. With a learner (and learning.enabled), troop and gold decisions
// are made by its networks; the rule modules then only supply the prior.
Plan decide(const State& s, const Config& cfg, Memory& mem, Learner* learner = nullptr);

}  // namespace fw
