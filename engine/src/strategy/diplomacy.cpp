#include <algorithm>

#include "strategy/planner.h"

namespace fw {

namespace {

bool atWarWith(const State& s, const Player& p) {
    return s.me.attackOn(p.small) || std::any_of(s.me.incoming.begin(), s.me.incoming.end(),
                                                 [&](const Incoming& a) { return a.attacker == p.small; });
}

}  // namespace

// Alliances with strong neighbours keep a front quiet while we eat the weak ones. Allying the #1 player
// mostly helps them win, so we only do it when they're on our border and would crush us.
void diplomacyModule(Ctx& c) {
    const auto& s = c.s;
    const auto& cfg = c.cfg.diplomacy;
    if (!cfg.enabled || s.inSpawn) return;

    const Player* leader = s.leader();
    auto leaderOk = [&](const Player& p) {
        return &p != leader || (p.border > 0 && p.troops >= cfg.leaderAllyFrac * s.me.troops);
    };

    // Allies on every side leave nowhere to grow (we then sit on full troops until someone eats us),
    // so a new alliance must leave free land or at least one other neighbour we may attack.
    auto keepsFrontOpen = [&](const Player& ally) {
        if (s.neutralBorder > 0) return true;
        return std::any_of(s.players.begin(), s.players.end(), [&](const Player& q) {
            return &q != &ally && q.border > 0 && !q.allied && !q.friendly;
        });
    };

    for (const auto& p : s.players) {
        if (p.isBot()) continue;

        if (cfg.embargoEnemies && !p.allied && !p.embargoed && atWarWith(s, p) &&
            c.mem.ready("embargo:" + p.id, s.tick, 300)) {
            c.mem.mark("embargo:" + p.id, s.tick);
            c.plan.actions.push_back({{"type", "embargo"}, {"target", p.id}, {"action", "start"}});
            c.plan.log.push_back("embargo " + p.name);
        }

        if (p.allied) continue;

        if (p.requestingAlliance && c.mem.ready("reply:" + p.id, s.tick, 50)) {
            bool strong = p.troops >= cfg.acceptIfStrongerFrac * s.me.troops;
            bool quietNeighbor = cfg.acceptNeighbors && p.border > 0 && !s.me.attackOn(p.small);
            bool accept = !p.isTraitor && (strong || quietNeighbor) && leaderOk(p) && keepsFrontOpen(p);
            c.mem.mark("reply:" + p.id, s.tick);
            c.plan.actions.push_back({{"type", "alliance_reply"}, {"requestor", p.id}, {"accept", accept}});
            c.plan.log.push_back(std::string(accept ? "accept" : "decline") + " alliance with " + p.name);
            continue;
        }

        if (p.border > 0 && p.troops >= cfg.requestIfStrongerFrac * s.me.troops && !s.me.attackOn(p.small) &&
            leaderOk(p) && keepsFrontOpen(p) && c.mem.ready("ally:" + p.id, s.tick, cfg.requestCooldown)) {
            c.mem.mark("ally:" + p.id, s.tick);
            c.plan.actions.push_back({{"type", "alliance_request"}, {"recipient", p.id}});
            c.plan.log.push_back("request alliance with " + p.name);
        }
    }

    // Opt-in only: when boxed in by allies, turn on the weakest one.
    if (!cfg.allowBetrayal || s.me.isTraitor || s.neutralBorder > 0) return;
    const Player* weakest = nullptr;
    for (const auto& p : s.players) {
        if (p.border == 0) continue;
        if (!p.allied) return;  // there is still someone fair game
        if (!weakest || p.troops < weakest->troops) weakest = &p;
    }
    if (weakest && weakest->troops < 0.3 * s.me.troops && c.mem.ready("betray", s.tick, 600)) {
        c.mem.mark("betray", s.tick);
        c.plan.actions.push_back({{"type", "break_alliance"}, {"recipient", weakest->id}});
        c.plan.log.push_back("break alliance with " + weakest->name);
    }
}

}  // namespace fw
