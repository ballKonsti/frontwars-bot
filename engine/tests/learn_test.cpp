#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <thread>
#include <filesystem>
#include <random>

#include "config.h"
#include "learn/learner.h"
#include "learn/mlp.h"
#include "state.h"
#include "strategy/planner.h"

using nlohmann::json;

namespace {

std::string tempDir(const char* name) {
    auto dir = std::filesystem::temp_directory_path() / (std::string("fw_test_") + name + "_" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    return dir.string();
}

json learnState(int tick, int tiles, double troops) {
    json bot = {{"small", 9}, {"id", "bot9"}, {"name", "bot9"}, {"type", "BOT"}, {"tiles", 200}, {"troops", 2000},
                {"gold", 0}, {"isTraitor", false}, {"allied", false}, {"friendly", false},
                {"requestingAlliance", false}, {"border", 30}, {"underAttack", 0}};
    json nation = bot;
    nation["small"] = 8;
    nation["id"] = nation["name"] = "nat8";
    nation["type"] = "FAKEHUMAN";
    nation["troops"] = 60000;
    json ally = bot;
    ally["small"] = 7;
    ally["id"] = ally["name"] = "ally7";
    ally["type"] = "FAKEHUMAN";
    ally["allied"] = ally["friendly"] = true;
    return {{"ready", true}, {"spawned", true}, {"inSpawn", false}, {"tick", tick}, {"landTiles", 100000},
            {"me", {{"small", 1}, {"id", "me"}, {"tiles", tiles}, {"troops", troops}, {"maxTroops", 100000},
                    {"troopRate", 300}, {"gold", 300000}, {"alive", true}, {"borderLen", 120}, {"coastal", false},
                    {"outgoing", json::array()}, {"incoming", json::array()}, {"units", json::array()}}},
            {"neutralBorder", 40}, {"players", json::array({bot, nation, ally})},
            {"enemyStructures", json::array()}, {"boatTargets", json::array()}};
}

fw::Config quietConfig() {
    fw::Config cfg;
    cfg.learning.epsilon = 0;
    return cfg;
}

}  // namespace

TEST(Mlp, FitsSmallRegression) {
    fw::Mlp net({2, 16, 8, 1}, 3);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1, 1);
    std::vector<std::vector<float>> xs(256);
    std::vector<float> ys(256);
    for (size_t i = 0; i < xs.size(); ++i) {
        xs[i] = {u(rng), u(rng)};
        ys[i] = std::fabs(xs[i][0]) - 0.5f * xs[i][1];
    }
    std::vector<const std::vector<float>*> ptrs;
    for (auto& x : xs) ptrs.push_back(&x);
    float first = net.train(ptrs, ys, 1e-2f), last = first;
    for (int k = 0; k < 600; ++k) last = net.train(ptrs, ys, 1e-2f);
    EXPECT_LT(last, 0.1f * first);
    EXPECT_LT(last, 0.01f);
}

TEST(Mlp, SaveLoadRoundTrip) {
    auto dir = tempDir("mlp");
    std::filesystem::create_directories(dir);
    fw::Mlp a({3, 8, 4, 1}, 5), b({3, 8, 4, 1}, 99), wrong({4, 8, 4, 1}, 1);
    std::vector<float> x = {0.3f, -0.2f, 0.9f};
    ASSERT_TRUE(a.save(dir + "/a.nn"));
    ASSERT_TRUE(b.load(dir + "/a.nn"));
    EXPECT_FLOAT_EQ(a.predict(x), b.predict(x));
    EXPECT_FALSE(wrong.load(dir + "/a.nn"));
}

TEST(Learner, UntrainedFollowsRules) {
    auto cfg = quietConfig();
    fw::Learner learner(cfg, tempDir("rules"));
    auto s = fw::parseState(learnState(1000, 500, 50000));
    fw::Memory ruleMem, mem;
    fw::Plan rules = fw::decide(s, cfg, ruleMem);  // no learner: pure rules
    fw::Plan plan = fw::decide(s, cfg, mem, &learner);
    // An untrained net scores ~0 everywhere, so the prior makes it take the rules' first troop move.
    auto firstAttack = [](const fw::Plan& p) {
        for (const auto& a : p.actions)
            if (a["type"] == "attack") return a["target"];
        return json("none");
    };
    EXPECT_NE(firstAttack(rules), json("none"));
    EXPECT_EQ(firstAttack(plan), firstAttack(rules));
}

TEST(Learner, OptionsRespectMasks) {
    auto cfg = quietConfig();
    fw::Learner learner(cfg, tempDir("masks"));
    auto s = fw::parseState(learnState(1000, 500, 50000));
    fw::Plan plan;
    plan.troopBudget = 30000;
    plan.goldBudget = 130000;
    for (const auto& o : learner.troopOptions(s, plan, false)) {
        EXPECT_LE(o.troops, plan.troopBudget + 1e-6);
        if (!o.action.is_null()) EXPECT_NE(o.action.value("target", json()), "ally7");
    }
    for (const auto& o : learner.goldOptions(s, plan)) EXPECT_LE(o.gold, plan.goldBudget);
    EXPECT_EQ(learner.troopOptions(s, plan, /*overextended=*/true).size(), 1u);  // only wait
}

TEST(Learner, NoWaitingOnFullTroops) {
    auto cfg = quietConfig();
    fw::Learner learner(cfg, tempDir("full"));
    fw::Plan plan;
    plan.troopBudget = 1e9;
    auto full = learner.troopOptions(fw::parseState(learnState(1000, 500, 95000)), plan, false);
    for (const auto& o : full) EXPECT_FALSE(o.action.is_null()) << o.label;
    auto half = learner.troopOptions(fw::parseState(learnState(1000, 500, 50000)), plan, false);
    EXPECT_TRUE(half[0].action.is_null());
}

// Plays a scripted game: land grows by `growth` per decision, then the game ends.
static void playEpisode(fw::Learner& learner, const fw::Config& cfg, double growth, bool won, bool eliminated) {
    fw::Memory mem;
    double tiles = 500;
    for (int k = 0; k < 60; ++k) {
        fw::decide(fw::parseState(learnState(1000 + 20 * k, int(tiles), 50000)), cfg, mem, &learner);
        tiles *= growth;
    }
    learner.gameOver(won, eliminated);
}

TEST(Learner, EpisodesBecomeSamples) {
    auto cfg = quietConfig();
    auto dir = tempDir("episodes");
    fw::Learner learner(cfg, dir);
    playEpisode(learner, cfg, 1.03, true, false);
    playEpisode(learner, cfg, 0.97, false, true);
    auto rep = learner.train(50);
    EXPECT_EQ(rep.episodes, 2);
    EXPECT_EQ(rep.samples[fw::Learner::kTroop], 120u);
    EXPECT_EQ(rep.samples[fw::Learner::kGold], 120u);
    EXPECT_EQ(learner.samples(fw::Learner::kTroop), 120u);
}

TEST(Learner, WinningGamesScoreHigher) {
    auto cfg = quietConfig();
    cfg.learning.maxEpisodes = 100;
    auto dir = tempDir("outcome");
    fw::Learner learner(cfg, dir);
    for (int i = 0; i < 3; ++i) {
        playEpisode(learner, cfg, 1.03, true, false);
        playEpisode(learner, cfg, 0.97, false, true);
    }
    auto rep = learner.train(1500);
    EXPECT_EQ(rep.episodes, 6);
    // Same decision, but from a growing (winning) position vs a shrinking (losing) one.
    fw::Plan plan;
    plan.troopBudget = 1e9;
    auto best = [&](int tiles) {
        float top = -1e9f;
        for (const auto& o : learner.troopOptions(fw::parseState(learnState(1600, tiles, 50000)), plan, false))
            top = std::max(top, learner.score(fw::Learner::kTroop, o));
        return top;
    };
    EXPECT_GT(best(1200), best(200) + 0.5f);
}

TEST(Learner, CutShortGameDropsItsLastDecisions) {
    auto cfg = quietConfig();
    auto dir = tempDir("cut");
    {
        fw::Learner learner(cfg, dir);
        fw::Memory mem;
        for (int k = 0; k < 100; ++k)  // 2000 ticks, never ended
            fw::decide(fw::parseState(learnState(1000 + 20 * k, 500 + k, 50000)), cfg, mem, &learner);
    }
    fw::Learner learner(cfg, dir);
    auto rep = learner.train(10);
    EXPECT_EQ(rep.episodes, 1);
    EXPECT_GT(rep.samples[fw::Learner::kTroop], 40u);
    EXPECT_LT(rep.samples[fw::Learner::kTroop], 100u);
}

TEST(Learner, LearnsToPreferRewardedOption) {
    auto cfg = quietConfig();
    cfg.learning.prior = 0;
    fw::Learner learner(cfg, tempDir("prefer"));
    auto j = learnState(1000, 500, 90000);
    j["neutralBorder"] = 0;
    auto s = fw::parseState(j);
    fw::Plan budget;
    budget.troopBudget = 1e9;
    auto opts = learner.troopOptions(s, budget, false);
    for (int rep = 0; rep < 40; ++rep)
        for (const auto& o : opts)
            learner.addSample(fw::Learner::kTroop, o.x, o.label.rfind("attack bot9", 0) == 0 ? 1.f : -1.f);
    learner.trainSteps(fw::Learner::kTroop, 400);

    fw::Memory mem;
    fw::Plan plan;
    plan.troopBudget = 1e9;
    fw::Ctx c{s, cfg, mem, plan};
    learner.troopPhase(c);
    ASSERT_EQ(plan.actions.size(), 1u);
    EXPECT_EQ(plan.actions[0]["target"], "bot9");
}

TEST(Learner, PicksSpawnAndReconsidersOnce) {
    auto cfg = quietConfig();
    fw::Learner learner(cfg, tempDir("spawn"));
    json cands = json::array();
    for (int i = 0; i < 5; ++i)
        cands.push_back({{"tile", 100 + i}, {"h", 0.1 * (i + 1)}, {"f", json::array({0.5, 0.5, 0.5, 0, 0, 0.8, 0.3, 0.2, 0.2})}});
    json pre = {{"ready", true}, {"spawned", false}, {"inSpawn", true}, {"tick", 5}, {"landTiles", 100000},
                {"spawnCandidates", cands}};
    fw::Memory mem;
    auto plan = fw::decide(fw::parseState(pre), cfg, mem, &learner);
    ASSERT_EQ(plan.actions.size(), 1u);
    EXPECT_EQ(plan.actions[0]["type"], "spawn");
    EXPECT_EQ(plan.actions[0]["tile"], 104);  // untrained: the rule's favourite

    // Late in the spawn phase a clearly better site shows up: move once, then stay.
    auto late = learnState(70, 50, 20000);
    late["inSpawn"] = true;
    cands.push_back({{"tile", 999}, {"h", 0.9}, {"f", json::array({0.9, 0.9, 0.9, 0, 0, 0.9, 0.8, 0.1, 0.1})}});
    late["spawnCandidates"] = cands;
    plan = fw::decide(fw::parseState(late), cfg, mem, &learner);
    ASSERT_EQ(plan.actions.size(), 1u);
    EXPECT_EQ(plan.actions[0]["tile"], 999);
    late["tick"] = 80;
    EXPECT_TRUE(fw::decide(fw::parseState(late), cfg, mem, &learner).actions.empty());
}

TEST(Learner, StudiesWhilePlaying) {
    auto cfg = quietConfig();
    cfg.learning.studyHorizon = 100;
    auto dir = tempDir("study");
    {
        fw::Learner seed(cfg, dir);
        for (int i = 0; i < 2; ++i) {
            playEpisode(seed, cfg, 1.03, true, false);
            playEpisode(seed, cfg, 0.97, false, true);
        }
    }
    fw::Learner learner(cfg, dir);
    fw::Plan budget;
    budget.troopBudget = 1e9;
    auto probe = learner.troopOptions(fw::parseState(learnState(1600, 1200, 50000)), budget, false)[1];
    float before = learner.score(fw::Learner::kTroop, probe);
    learner.startStudy();
    fw::Memory mem;
    for (int k = 0; k < 40; ++k) {
        fw::decide(fw::parseState(learnState(1000 + 20 * k, 500 + 10 * k, 50000)), cfg, mem, &learner);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    learner.stopStudy();
    EXPECT_NE(learner.score(fw::Learner::kTroop, probe), before);  // the acting nets moved during the game
}

TEST(Learner, PersistsAcrossRuns) {
    auto cfg = quietConfig();
    auto dir = tempDir("persist");
    std::vector<float> x;
    float before = 0;
    {
        fw::Learner a(cfg, dir);
        x.assign(a.inputs(fw::Learner::kGold), 0.3f);
        for (int i = 0; i < 80; ++i) a.addSample(fw::Learner::kGold, x, 0.5f);
        a.trainSteps(fw::Learner::kGold, 200);
        fw::Option o;
        o.x = x;
        before = a.score(fw::Learner::kGold, o);
        a.save();
    }
    fw::Learner b(cfg, dir);
    b.load();
    fw::Option o;
    o.x = x;
    EXPECT_FLOAT_EQ(b.score(fw::Learner::kGold, o), before);
    EXPECT_EQ(b.samples(fw::Learner::kGold), 80u);
    EXPECT_LT(b.priorWeight(fw::Learner::kGold), cfg.learning.prior);
}
