#include <algorithm>
#include <optional>

#include "strategy/planner.h"

namespace fw {

namespace {

double estimatedCost(double base, int owned, const EconomyConfig& cfg) { return base * (1.0 + cfg.costGrowth * owned); }

void build(Ctx& c, const std::string& unit, const char* site, std::optional<int> near = std::nullopt) {
    nlohmann::json a{{"type", "build"}, {"unit", unit}, {"site", site}};
    if (near) a["near"] = *near;
    c.plan.actions.push_back(std::move(a));
    c.plan.log.push_back("build " + unit);
}

}  // namespace

// One structure per cooldown. Opening (from watching a strong player): Factories first, then the first
// City before anything else. After that: defense where we're attacked, Cities (troop cap + regen),
// Factories (one per 2 cities), standing Defense Posts, Ports, Warships, then City upgrades.
void economyModule(Ctx& c) {
    const auto& s = c.s;
    const auto& cfg = c.cfg.economy;
    if (s.inSpawn || !c.mem.ready("economy", s.tick, cfg.cooldown)) return;

    const int cities = s.me.count("City");
    const int factories = s.me.count("Factory");
    const int ports = s.me.count("Port");
    const int posts = s.me.count("Defense Post");
    const int warships = s.me.count("Warship");
    auto cost = [&](const char* unit, double base, int owned) { return s.cost(unit, estimatedCost(base, owned, cfg)); };
    auto tryBuild = [&](const char* unit, double price, const char* site, std::optional<int> near = std::nullopt) {
        if (!c.plan.spend(price)) return false;
        c.mem.mark("economy", s.tick);
        build(c, unit, site, near);
        return true;
    };

    if (cities == 0) {
        bool done = factories < cfg.openingFactories
                        ? tryBuild("Factory", cost("Factory", cfg.factoryCost, factories), "interior")
                        : tryBuild("City", cost("City", cfg.cityCost, cities), "interior");
        if (!done) c.plan.goldBudget = 0;  // save for the opening
        return;
    }

    // Defense post facing the biggest attacker that outguns us locally.
    const Incoming* worst = nullptr;
    for (const auto& a : s.me.incoming)
        if (a.attacker != 0 && (!worst || a.troops > worst->troops)) worst = &a;
    const double postCost = cost("Defense Post", cfg.postCost, posts);
    if (worst && worst->troops > 0.3 * s.me.troops && posts < 1 + s.me.borderLen / 150 &&
        tryBuild("Defense Post", postCost, "border", worst->attacker))
        return;

    int cityCap = std::max(1, s.me.tiles / std::max(1, cfg.tilesPerCity));
    if (cities < cityCap && tryBuild("City", cost("City", cfg.cityCost, cities), "interior")) return;

    int factoryCap = std::max(1, cities / std::max(1, cfg.citiesPerFactory));
    if (factories < factoryCap && tryBuild("Factory", cost("Factory", cfg.factoryCost, factories), "interior")) return;

    // Standing posts on the most contested border, not just when attacked.
    // ...and one more as soon as a nation stronger than us is on the border.
    if (cfg.citiesPerPost > 0) {
        const Player* front = nullptr;
        for (const auto& p : s.players)
            if (p.border > 0 && !p.friendly && (!front || p.troops > front->troops)) front = &p;
        const bool outgunned = front && !front->isBot() && front->troops > s.me.troops;
        if (front && posts < cities / cfg.citiesPerPost + (outgunned ? 1 : 0) &&
            tryBuild("Defense Post", postCost, "border", front->small))
            return;
    }

    // Ports go to the far ends of our coast: longer trade routes pay more.
    int portCap = std::max(1, cities / std::max(1, cfg.portsPerCity));
    if (s.me.coastal && ports < portCap && tryBuild("Port", cost("Port", cfg.portCost, ports), "coast-far")) return;

    if (ports > 0 && warships < std::min(cfg.maxWarships, ports + 1) &&
        tryBuild("Warship", cost("Warship", cfg.warshipCost, warships), "coast"))
        return;

    // At the city cap: upgrade the lowest-level city instead.
    if (cities >= cityCap) {
        const Unit* low = nullptr;
        for (const auto& u : s.me.units)
            if (u.type == "City" && (!low || u.level < low->level)) low = &u;
        if (low && c.plan.spend(cost("City", cfg.cityCost, cities))) {
            c.mem.mark("economy", s.tick);
            c.plan.actions.push_back({{"type", "upgrade"}, {"unit", "City"}, {"unitId", low->id}});
            c.plan.log.push_back("upgrade City to " + std::to_string(low->level + 1));
        }
    }
}

}  // namespace fw
