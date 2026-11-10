#include <gtest/gtest.h>

#include "config.h"
#include "state.h"
#include "strategy/planner.h"

using nlohmann::json;

namespace {

json baseState() {
    return json::parse(R"({
      "ready": true, "spawned": true, "inSpawn": false, "tick": 1000, "mapW": 2000, "mapH": 1000, "landTiles": 100000,
      "me": {"small": 1, "id": "me", "tiles": 500, "troops": 50000, "maxTroops": 100000, "troopRate": 300,
             "gold": 0, "isTraitor": false, "alive": true, "coastal": false, "borderLen": 120,
             "outgoing": [], "incoming": [], "units": []},
      "neutralBorder": 40, "players": [], "enemyStructures": [], "nukesInbound": 0, "boatTargets": []
    })");
}

json player(int small, const std::string& id, const std::string& type, int tiles, double troops, int border) {
    return {{"small", small}, {"id", id}, {"name", id}, {"type", type}, {"tiles", tiles}, {"troops", troops},
            {"gold", 0}, {"isTraitor", false}, {"allied", false}, {"friendly", false},
            {"requestingAlliance", false}, {"border", border}};
}

fw::Plan run(const json& j, fw::Memory& mem) { return fw::decide(fw::parseState(j), fw::Config{}, mem); }

fw::Plan run(const json& j) {
    fw::Memory mem;
    return run(j, mem);
}

int countType(const fw::Plan& p, const std::string& type) {
    int n = 0;
    for (const auto& a : p.actions) n += a["type"] == type;
    return n;
}

}  // namespace

TEST(Planner, SpawnsWhenNotSpawned) {
    json s = {{"ready", true}, {"spawned", false}, {"inSpawn", true}, {"tick", 5}};
    auto plan = run(s);
    ASSERT_EQ(plan.actions.size(), 1u);
    EXPECT_EQ(plan.actions[0]["type"], "spawn");
}

TEST(Planner, ExpandsIntoNeutralLand) {
    auto plan = run(baseState());
    ASSERT_EQ(countType(plan, "attack"), 1);
    EXPECT_TRUE(plan.actions[0]["target"].is_null());
    EXPECT_GT(plan.actions[0]["troops"].get<double>(), 0);
}

TEST(Planner, ExpandRespectsCooldown) {
    fw::Memory mem;
    json s = baseState();
    run(s, mem);
    s["tick"] = 1002;
    EXPECT_EQ(countType(run(s, mem), "attack"), 0);
}

TEST(Planner, NeverAttacksAllies) {
    json s = baseState();
    s["neutralBorder"] = 0;
    s["me"]["troops"] = 90000;
    json ally = player(7, "ally", "NATION", 300, 1000, 50);
    ally["allied"] = true;
    ally["friendly"] = true;
    s["players"] = json::array({ally});
    for (const auto& a : run(s).actions) EXPECT_NE(a.value("target", json()), "ally");
}

TEST(Planner, AttacksWeakBotWhileLandRemains) {
    json s = baseState();
    s["me"]["troops"] = 90000;
    s["players"] = json::array({player(9, "bot9", "BOT", 200, 2000, 30), player(8, "nat8", "NATION", 200, 2000, 30)});
    auto plan = run(s);
    bool hitBot = false, hitNation = false;
    for (const auto& a : plan.actions) {
        if (a["type"] != "attack") continue;
        hitBot |= a["target"] == "bot9";
        hitNation |= a["target"] == "nat8";
    }
    EXPECT_TRUE(hitBot);
    EXPECT_FALSE(hitNation);
}

TEST(Planner, KeepsHomeReserveUnderAttack) {
    json s = baseState();
    s["me"]["troops"] = 40000;
    s["me"]["incoming"] = json::array({{{"id", "a"}, {"attacker", 5}, {"troops", 38000}}});
    auto plan = run(s);
    double sent = 0;
    for (const auto& a : plan.actions)
        if (a["type"] == "attack") sent += a["troops"].get<double>();
    EXPECT_EQ(sent, 0);  // reserve = 1.1 * 38000 > 40000, nothing left to commit
}

TEST(Planner, AcceptsAllianceFromStrongPlayer) {
    json s = baseState();
    json big = player(4, "big", "NATION", 5000, 200000, 15);  // leader, but on our border and 4x our troops
    big["requestingAlliance"] = true;
    s["players"] = json::array({big});
    auto plan = run(s);
    ASSERT_EQ(countType(plan, "alliance_reply"), 1);
    for (const auto& a : plan.actions)
        if (a["type"] == "alliance_reply") EXPECT_TRUE(a["accept"].get<bool>());
}

json withFactories(json s, int n) {
    for (int i = 0; i < n; ++i)
        s["me"]["units"].push_back({{"id", 100 + i}, {"type", "Factory"}, {"tile", 20 + i}, {"level", 1}});
    return s;
}

TEST(Planner, BuildsCityWhenAffordable) {
    json s = withFactories(baseState(), 2);
    s["me"]["gold"] = 200000;
    auto plan = run(s);
    bool city = false;
    for (const auto& a : plan.actions) city |= a["type"] == "build" && a["unit"] == "City";
    EXPECT_TRUE(city);
}

json armedState() {
    json s = baseState();
    s["me"]["gold"] = 1000000;
    s["me"]["tiles"] = 100;  // below silo threshold, so no new-silo saving
    s["me"]["units"] = json::array({{{"id", 1}, {"type", "Missile Silo"}, {"tile", 10}, {"level", 1}},
                                    {{"id", 2}, {"type", "City"}, {"tile", 11}, {"level", 1}}});
    s["players"] = json::array({player(3, "foe", "NATION", 4000, 80000, 20)});
    s["enemyStructures"] = json::array({{{"owner", 3}, {"type", "City"}, {"tile", 500}, {"level", 1}},
                                        {{"owner", 3}, {"type", "Missile Silo"}, {"tile", 600}, {"level", 1}}});
    return s;
}

TEST(Planner, NoFirstStrikeByDefault) {
    EXPECT_EQ(countType(run(armedState()), "nuke"), 0);
}

TEST(Planner, RetaliatesAgainstSiloOfWhoeverNukedUs) {
    json s = armedState();
    s["nukeSources"] = json::array({3});
    auto plan = run(s);
    bool nuked = false;
    for (const auto& a : plan.actions)
        if (a["type"] == "nuke") {
            nuked = true;
            EXPECT_EQ(a["tile"], 600);
            EXPECT_EQ(a["unit"], "Atom Bomb");
        }
    EXPECT_TRUE(nuked);
}

TEST(Planner, NukesRunawayLeader) {
    json s = armedState();
    s["players"][0]["tiles"] = 40000;  // 40% of the land
    EXPECT_EQ(countType(run(s), "nuke"), 1);
}

TEST(Planner, OpeningIsFactoriesThenCity) {
    json s = baseState();
    s["me"]["gold"] = 500000;
    auto plan = run(s);
    ASSERT_EQ(countType(plan, "build"), 1);
    for (const auto& a : plan.actions)
        if (a["type"] == "build") EXPECT_EQ(a["unit"], "Factory");

    plan = run(withFactories(s, 2));
    ASSERT_EQ(countType(plan, "build"), 1);
    for (const auto& a : plan.actions)
        if (a["type"] == "build") EXPECT_EQ(a["unit"], "City");
}

TEST(Planner, UsesRealCostsFromGame) {
    json s = withFactories(baseState(), 2);
    s["me"]["gold"] = 50000;  // below the 125k fallback estimate
    s["costs"] = {{"City", 40000}};
    bool city = false;
    for (const auto& a : run(s).actions) city |= a["type"] == "build" && a["unit"] == "City";
    EXPECT_TRUE(city);
}

TEST(Planner, FactoryAfterCityCap) {
    json s = baseState();
    s["me"]["gold"] = 500000;
    s["me"]["tiles"] = 299;  // city cap 1, below the silo threshold
    s["me"]["units"] = json::array({{{"id", 2}, {"type", "City"}, {"tile", 11}, {"level", 1}}});
    for (const auto& a : run(s).actions)
        if (a["type"] == "build") EXPECT_EQ(a["unit"], "Factory");
}

TEST(Planner, HoldsOffensivesWhenBleeding) {
    fw::Memory mem;
    json s = baseState();
    s["me"]["outgoing"] = json::array({{{"id", "x"}, {"target", 0}, {"troops", 1000}}});
    s["me"]["troops"] = 90000;
    for (int t = 1000; t <= 1060; t += 10) {
        s["tick"] = t;
        s["me"]["troops"] = 90000 - (t - 1000) * 800;  // falls to 42000
        run(s, mem);
    }
    s["tick"] = 1070;
    s["me"]["outgoing"] = json::array({{{"id", "x"}, {"target", 0}, {"troops", 10}}});
    auto plan = run(s, mem);
    EXPECT_EQ(countType(plan, "attack"), 0);
}

TEST(Planner, DeclinesAllianceWithDistantLeader) {
    json s = baseState();
    json big = player(4, "big", "NATION", 5000, 200000, 0);
    big["requestingAlliance"] = true;
    s["players"] = json::array({big});
    for (const auto& a : run(s).actions)
        if (a["type"] == "alliance_reply") EXPECT_FALSE(a["accept"].get<bool>());
}

TEST(Planner, EmbargoesAttacker) {
    json s = baseState();
    s["players"] = json::array({player(5, "foe", "NATION", 300, 20000, 10)});
    s["me"]["incoming"] = json::array({{{"id", "a"}, {"attacker", 5}, {"troops", 1000}}});
    EXPECT_EQ(countType(run(s), "embargo"), 1);
}

TEST(Planner, PrefersDistractedTarget) {
    json s = baseState();
    s["neutralBorder"] = 0;
    s["me"]["troops"] = 90000;
    json a = player(6, "calm", "NATION", 400, 4000, 30);
    json b = player(7, "busy", "NATION", 400, 4000, 30);
    b["underAttack"] = 2;
    s["players"] = json::array({a, b});
    auto plan = run(s);
    bool hitBusy = false;
    for (const auto& x : plan.actions)
        if (x["type"] == "attack") hitBusy |= x["target"] == "busy";
    EXPECT_TRUE(hitBusy);
}

TEST(Config, LoadsRepoConfig) {
    auto cfg = fw::loadConfig("config.json");
    EXPECT_DOUBLE_EQ(cfg.expand.ratio, 0.20);
    EXPECT_FALSE(cfg.strategic.firstStrike);
    EXPECT_FALSE(cfg.diplomacy.allowBetrayal);
}
