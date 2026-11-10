#include "strategy/planner.h"

namespace fw {

// Grab neutral land. Re-sending an attack on neutral land tops up the running one.
void expandModule(Ctx& c) {
    const auto& s = c.s;
    const auto& cfg = c.cfg.expand;
    if (s.inSpawn || s.neutralBorder == 0 || c.plan.overextended) return;
    if (s.me.troops < cfg.minTroopFrac * s.me.maxTroops) return;
    if (!c.mem.ready("expand", s.tick, cfg.cooldown)) return;

    double want = cfg.ratio * s.me.troops;
    if (const Attack* running = s.me.attackOn(0); running && running->troops > 0.5 * want) return;

    double troops = c.plan.commit(want);
    if (troops < 1) return;
    c.mem.mark("expand", s.tick);
    c.plan.actions.push_back({{"type", "attack"}, {"target", nullptr}, {"troops", troops}});
}

}  // namespace fw
