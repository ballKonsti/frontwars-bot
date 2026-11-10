#include <algorithm>

#include "strategy/planner.h"

namespace fw {

namespace {

// Silos first (they can hit us back), then SAMs guarding them, then cities.
int targetValue(const std::string& type) {
    if (type == "Missile Silo") return 4;
    if (type == "SAM Launcher") return 3;
    if (type == "City") return 2;
    if (type == "Port") return 1;
    return 0;
}

void buildHigh(Ctx& c, const char* unit) {
    c.plan.actions.push_back({{"type", "build"},
                              {"unit", unit},
                              {"site", "interior"},
                              {"mountainWeight", c.cfg.strategic.mountainWeight}});
    c.plan.log.push_back(std::string("build ") + unit);
}

}  // namespace

// Deterrence doctrine: silos mainly as a threat, a second one for second-strike, SAMs once anyone can
// hit us. Nukes go to players who nuked us first, or to a runaway leader, unless firstStrike is on.
void strategicModule(Ctx& c) {
    strategicBuildModule(c);
    nukeModule(c);
}

void strategicBuildModule(Ctx& c) {
    const auto& s = c.s;
    const auto& cfg = c.cfg.strategic;
    if (!cfg.enabled || s.inSpawn) return;

    const int silos = s.me.count("Missile Silo");
    const int sams = s.me.count("SAM Launcher");
    const int cities = s.me.count("City");
    bool enemySilos = std::any_of(s.enemyStructures.begin(), s.enemyStructures.end(),
                                  [](const Structure& u) { return u.type == "Missile Silo"; });

    // The economy opening (factories, first city) comes first.
    if (cities == 0 || !c.mem.ready("strategic", s.tick, cfg.cooldown)) return;
    const double siloCost = s.cost("Missile Silo", cfg.siloCost);
    const double samCost = s.cost("SAM Launcher", cfg.samCost);

    if (silos == 0 && s.me.tiles >= cfg.minTilesForSilo && cities >= cfg.minCitiesForSilo) {
        if (c.plan.spend(siloCost)) {
            c.mem.mark("strategic", s.tick);
            buildHigh(c, "Missile Silo");
        } else {
            c.plan.goldBudget = 0;  // save up instead of buying cities
        }
        return;
    }
    int samCap = 1 + cities / std::max(1, cfg.citiesPerSam);
    if ((s.nukesInbound > 0 || enemySilos) && sams < samCap && c.plan.spend(samCost)) {
        c.mem.mark("strategic", s.tick);
        buildHigh(c, "SAM Launcher");
        return;
    }
    // More silos as we grow; buildHigh spreads them out, which keeps a second-strike option.
    int siloCap = std::clamp(cities / std::max(1, cfg.citiesPerSilo), 1, std::max(1, cfg.maxSilos));
    if (silos > 0 && silos < siloCap && c.plan.goldBudget >= siloCost + s.cost("Atom Bomb", cfg.atomCost) &&
        c.plan.spend(siloCost)) {
        c.mem.mark("strategic", s.tick);
        buildHigh(c, "Missile Silo");
        return;
    }
}

void nukeModule(Ctx& c) {
    const auto& s = c.s;
    const auto& cfg = c.cfg.strategic;
    if (!cfg.enabled || s.inSpawn) return;
    if (s.me.count("Missile Silo") == 0 || !c.mem.ready("nuke", s.tick, cfg.nukeCooldown)) return;

    const Player* leader = s.leader();
    auto fairTarget = [&](const Player& p) {
        if (cfg.firstStrike) return true;
        if (c.mem.nukedBy.count(p.small)) return true;
        return &p == leader && double(p.tiles) / s.landTiles >= cfg.leaderShare;
    };

    // Strike the most valuable structure of the strongest permitted enemy; retaliation comes first.
    const Structure* best = nullptr;
    double bestScore = 0;
    for (const auto& u : s.enemyStructures) {
        const Player* p = s.player(u.owner);
        if (!p || p->allied || p->friendly || p->isBot() || !fairTarget(*p)) continue;
        double score = targetValue(u.type) * (1.0 + u.level) * (p->border > 0 ? 2.0 : 1.0) * (1.0 + p->troops / 1e5) *
                       (c.mem.nukedBy.count(p->small) ? 3.0 : 1.0);
        if (score > bestScore) {
            bestScore = score;
            best = &u;
        }
    }
    if (!best) return;

    std::string bomb;
    if (c.plan.spend(s.cost("Hydrogen Bomb", cfg.hydrogenCost)))
        bomb = "Hydrogen Bomb";
    else if (c.plan.spend(s.cost("Atom Bomb", cfg.atomCost)))
        bomb = "Atom Bomb";
    else
        return;

    c.mem.mark("nuke", s.tick);
    c.plan.actions.push_back({{"type", "nuke"}, {"unit", bomb}, {"tile", best->tile}});
    c.plan.log.push_back(bomb + " -> " + best->type + " of " + s.player(best->owner)->name);
}

}  // namespace fw
