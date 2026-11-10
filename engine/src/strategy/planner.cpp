#include "strategy/planner.h"

#include <algorithm>

#include "learn/learner.h"

namespace fw {

double homeReserve(const State& s, const Config& cfg) {
    const auto& m = cfg.military;
    return std::max(m.reserveFrac * s.me.maxTroops, m.incomingReserve * s.me.incomingTroops());
}

bool overextended(const State& s, const SustainConfig& cfg, Memory& mem) {
    auto& h = mem.troopHistory;
    if (!h.empty() && s.tick < h.back().first) h.clear();  // new game
    h.emplace_back(s.tick, s.me.troops);
    while (h.size() > 1 && s.tick - h.front().first > cfg.window) h.pop_front();
    if (s.me.outgoing.empty() || s.tick - h.front().first < cfg.window / 2) return false;
    double peak = 0;
    for (const auto& [tick, troops] : h) peak = std::max(peak, troops);
    return peak > 0 && (peak - s.me.troops) / peak > cfg.maxDrop;
}

Plan decide(const State& s, const Config& cfg, Memory& mem, Learner* learner) {
    Plan plan;
    Ctx c{s, cfg, mem, plan};
    if (!s.ready) return plan;
    const bool learning = learner && cfg.learning.enabled;
    if (learning) learner->observe(s);
    if (!s.spawned) {
        if (!learning || !learner->spawnPhase(c)) spawnModule(c);
        return plan;
    }
    if (!s.me.alive) return plan;
    if (learning && s.inSpawn) learner->spawnPhase(c);  // one second look before the game begins

    plan.troopBudget = std::max(0.0, s.me.troops - homeReserve(s, cfg));
    plan.goldBudget = std::max(0.0, s.me.gold - cfg.economy.goldReserve);
    plan.overextended = overextended(s, cfg.sustain, mem);
    if (plan.overextended && mem.ready("log:overextended", s.tick, 100)) {
        mem.mark("log:overextended", s.tick);
        plan.log.push_back("troops bleeding faster than regen, holding offensives");
    }
    for (int src : s.nukeSources) mem.nukedBy.insert(src);

    if (learning) {
        diplomacyModule(c);
        learner->goldPhase(c);
        nukeModule(c);
        learner->troopPhase(c);
        return plan;
    }

    // Diplomacy first (cheap, protects us), then spend gold (strategic before economy so
    // savings for a silo aren't eaten by cities), then spend troops.
    diplomacyModule(c);
    strategicModule(c);
    economyModule(c);
    militaryModule(c);
    expandModule(c);
    navalModule(c);
    return plan;
}

void spawnModule(Ctx& c) {
    if (!c.s.inSpawn || !c.mem.ready("spawn", c.s.tick, 20)) return;
    c.mem.mark("spawn", c.s.tick);
    // The page picks the tile: lots of free land around it, preferably near a corner.
    c.plan.actions.push_back({{"type", "spawn"}, {"edgeWeight", c.cfg.spawn.edgeWeight}});
    c.plan.log.push_back("spawning");
}

}  // namespace fw
