#include <algorithm>

#include "strategy/planner.h"

namespace fw {

// Ship troops to the nearest free or weakly held shore once land expansion slows down.
// Strong players do this early (islands while land targets still exist).
void navalModule(Ctx& c) {
    const auto& s = c.s;
    const auto& cfg = c.cfg.naval;
    if (!cfg.enabled || s.inSpawn || c.plan.overextended || !s.me.coastal || s.boatTargets.empty()) return;
    if (s.neutralBorder > cfg.maxNeutralBorder) return;  // plenty of land next door is still the better deal
    if (s.me.count("Transport") >= cfg.maxBoats) return;
    if (s.me.troops < cfg.minTroopFrac * s.me.maxTroops) return;
    if (!c.mem.ready("boat", s.tick, cfg.cooldown)) return;

    // With no free land and nobody to fight next door, the sea is the only way to keep growing.
    bool island = s.neutralBorder == 0 && std::none_of(s.players.begin(), s.players.end(), [](const Player& p) {
        return p.border > 0 && !p.friendly && !p.allied;
    });
    const double maxTroops = (island ? cfg.islandTargetTroops : cfg.maxTargetTroops) * s.me.troops;

    const BoatTarget* pick = nullptr;
    const Player* owner = nullptr;
    for (const auto& b : s.boatTargets) {  // already sorted nearest first
        if (b.owner == 0) {
            pick = &b;
            owner = nullptr;
            break;
        }
        const Player* p = s.player(b.owner);
        if (!pick && p && !p->friendly && !p->allied && p->troops < maxTroops) {
            pick = &b;
            owner = p;
        }
    }
    if (!pick) return;

    double troops = c.plan.commit(cfg.ratio * s.me.troops);
    if (troops < 1) return;
    c.mem.mark("boat", s.tick);
    nlohmann::json a{{"type", "boat"}, {"dst", pick->tile}, {"troops", troops}};
    a["target"] = owner ? nlohmann::json(owner->id) : nlohmann::json(nullptr);
    c.plan.actions.push_back(std::move(a));
    c.plan.log.push_back(std::string("boat to ") + (owner ? owner->name : "free land"));
}

}  // namespace fw
