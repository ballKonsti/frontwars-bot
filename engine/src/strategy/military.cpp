#include <algorithm>
#include <cmath>

#include "strategy/planner.h"

namespace fw {

namespace {

// Taking a tile costs roughly the defender's troop density, so that's what we price targets by.
double density(const Player& p) { return p.troops / std::max(1, p.tiles); }

double troopsNeeded(const Player& p, const MilitaryConfig& cfg) {
    double adv = p.isBot() ? cfg.botAdvantage : cfg.advantage;
    return adv * density(p) * std::min(p.tiles, cfg.minGainTiles);
}

}  // namespace

void militaryModule(Ctx& c) {
    const auto& s = c.s;
    const auto& cfg = c.cfg.military;
    if (s.inSpawn || c.plan.overextended) return;

    int running = int(std::count_if(s.me.outgoing.begin(), s.me.outgoing.end(),
                                    [](const Attack& a) { return a.target != 0; }));
    if (running >= cfg.maxConcurrent) return;
    if (s.me.troops < cfg.minTroopFrac * s.me.maxTroops) return;

    // While there is free land, only bots are worth the troops.
    bool landLeft = s.neutralBorder > 0;

    const Player* best = nullptr;
    double bestScore = 0;
    for (const auto& p : s.players) {
        if (p.border == 0 || p.friendly || p.allied) continue;
        if (landLeft && !p.isBot()) continue;
        if (s.me.attackOn(p.small)) continue;
        double need = troopsNeeded(p, cfg);
        if (need > cfg.ratio * s.me.troops) continue;
        // Cheap, wide fronts first; players already attacking us get priority, and players busy
        // fighting someone else have their army out of position.
        bool hostile = std::any_of(s.me.incoming.begin(), s.me.incoming.end(),
                                   [&](const Incoming& a) { return a.attacker == p.small; });
        bool distracted = !hostile && p.underAttack > 0;
        bool finishable = cfg.ratio * s.me.troops >= 1.2 * p.troops;  // eliminating a player pays gold
        double score = (1.0 + std::sqrt(double(p.border))) / (1.0 + need) * (hostile ? 2.0 : 1.0) *
                       (distracted ? cfg.distractedBonus : 1.0) * (finishable ? cfg.finishBonus : 1.0);
        if (score > bestScore) {
            bestScore = score;
            best = &p;
        }
    }
    if (!best) return;

    std::string key = "attack:" + best->id;
    if (!c.mem.ready(key, s.tick, cfg.cooldown)) return;
    double troops = c.plan.commit(cfg.ratio * s.me.troops);
    if (troops < troopsNeeded(*best, cfg)) return;

    c.mem.mark(key, s.tick);
    c.plan.actions.push_back({{"type", "attack"}, {"target", best->id}, {"troops", troops}});
    c.plan.log.push_back("attack " + best->name + " (" + best->type + ") with " + std::to_string(int(troops / 10)));
}

}  // namespace fw
