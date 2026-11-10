#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <thread>
#include <filesystem>
#include <fstream>
#include <random>

#include "config.h"
#include "learn/learner.h"
#include "learn/episodes.h"
#include "learn/mlp.h"
#include "learn/model.h"
#include "learn/trainer.h"
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
    cfg.learning.size = "small";  // fast tests; the layers are the same at every size
    cfg.learning.ensemble = 3;
    return cfg;
}

// Plays a scripted game: land grows by `growth` per decision, then the game ends.
void playEpisode(fw::Learner& learner, const fw::Config& cfg, double growth, bool won, bool eliminated, int steps = 60) {
    fw::Memory mem;
    double tiles = 500;
    for (int k = 0; k < steps; ++k) {
        fw::decide(fw::parseState(learnState(1000 + 20 * k, int(tiles), 50000)), cfg, mem, &learner);
        tiles *= growth;
    }
    learner.gameOver(won, eliminated);
}

fw::nn::Shape tinyShape() {
    fw::nn::Shape s;
    s.self = 5, s.ent = 4, s.opt[0] = 3, s.opt[1] = 2;
    s.D = 12, s.E = 6, s.heads = 2, s.blocks = 2, s.aux = fw::nn::kAux;
    return s;
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


// The hand-written backward pass must agree with finite differences for every parameter tensor.
TEST(Model, GradientsMatchFiniteDifferences) {
    fw::nn::Model model(tinyShape(), 3);
    model.detachValue = false;  // so the loss below is an ordinary function of the weights
    std::mt19937 rng(5);
    std::normal_distribution<float> n(0, 1);
    // Untrained heads start near zero; shake every weight so no path is trivially flat.
    for (auto* p : model.params())
        for (auto& w : p->w) w += 0.3f * n(rng);
    std::vector<float> self(5), ents(3 * 4), opt(3), aux = {0.4f, 0.7f, 1.0f, 0.0f, 0.2f};
    for (auto* v : {&self, &ents, &opt})
        for (auto& x : *v) x = n(rng);
    fw::nn::Input in{self.data(), ents.data(), 3};
    fw::nn::Target tg;
    tg.net = 0, tg.opt = opt.data(), tg.target = 1, tg.G = 0.3f, tg.aux = aux.data(), tg.auxMask = 0b11111;
    auto loss = [&]() {
        model.zeroGrad();
        auto l = model.accumulate(in, tg);
        return double(l.value) + l.advantage + l.aux;
    };
    loss();
    std::vector<std::vector<float>> grads;
    for (auto* p : model.params()) grads.push_back(p->g);

    int checked = 0, bad = 0;
    auto params = model.params();
    for (size_t t = 0; t < params.size(); ++t) {
        std::uniform_int_distribution<size_t> pick(0, params[t]->w.size() - 1);
        for (int rep = 0; rep < 4; ++rep) {
            const size_t i = pick(rng);
            const float keep = params[t]->w[i], h = 2e-3f;
            params[t]->w[i] = keep + h;
            const double up = loss();
            params[t]->w[i] = keep - h;
            const double down = loss();
            params[t]->w[i] = keep;
            const double numeric = (up - down) / (2 * h);
            // A ReLU switching inside the +-h window makes a finite difference meaningless; allow a few.
            if (std::fabs(grads[t][i] - numeric) > 0.02 + 0.06 * std::fabs(numeric)) ++bad;
            ++checked;
        }
    }
    EXPECT_GT(checked, 60);
    EXPECT_LE(bad, checked / 40);
}

TEST(Model, HandlesNoVisiblePlayers) {
    fw::nn::Model model(tinyShape(), 3);
    std::vector<float> self(5, 0.2f), opt(2, 0.5f);
    fw::nn::Target tg;
    tg.net = 1, tg.opt = opt.data(), tg.G = 1.0f;
    auto l = model.accumulate({self.data(), nullptr, 0}, tg);
    EXPECT_TRUE(std::isfinite(l.value + l.advantage));
    model.step(1e-3f, 0, 1);
    model.encode({self.data(), nullptr, 0});
    EXPECT_TRUE(std::isfinite(model.value()));
}

// The return depends on the position (self[0]), on the move (opt[0]) and on the player it is aimed at
// (ents[.][1]). Given an honest baseline for the position, the advantage head must learn the rest.
TEST(Model, SplitsPositionFromMove) {
    fw::nn::Model model(tinyShape(), 9);
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> u(-1, 1);
    struct S {
        std::vector<float> self, ents, opt;
        int target;
        float position, move;
    };
    std::vector<S> data(400);
    for (auto& s : data) {
        s.self.resize(5), s.ents.resize(2 * 4), s.opt.resize(3);
        for (auto* v : {&s.self, &s.ents, &s.opt})
            for (auto& x : *v) x = u(rng);
        s.target = rng() % 2;
        s.position = 1.5f * s.self[0];
        s.move = 0.8f * s.opt[0] + 0.5f * s.ents[s.target * 4 + 1];
    }
    for (int step = 0; step < 2500; ++step) {
        for (int b = 0; b < 32; ++b) {
            const S& s = data[rng() % 300];  // the last 100 are held out
            fw::nn::Target tg;
            tg.net = 0, tg.opt = s.opt.data(), tg.target = s.target, tg.G = s.position + s.move;
            tg.hasBaseline = true, tg.baseline = s.position;
            model.accumulate({s.self.data(), s.ents.data(), 2}, tg);
        }
        model.step(2e-3f, 0, 32);
    }
    double ve = 0, ae = 0, rr = 0;
    for (size_t i = 300; i < data.size(); ++i) {
        const S& s = data[i];
        model.encode({s.self.data(), s.ents.data(), 2});
        ve += std::pow(model.value() - s.position, 2);
        ae += std::pow(model.advantage(0, s.opt.data(), s.target) - s.move, 2);
        rr += s.move * s.move;
    }
    EXPECT_LT(ve / 100, 0.4);  // V sees the position but not the move, on games it never trained on
    EXPECT_LT(ae / rr, 0.3);   // the head explains most of what the move contributes
}

TEST(Model, SaveLoadRoundTrip) {
    auto dir = tempDir("model");
    std::filesystem::create_directories(dir);
    fw::nn::Model a(tinyShape(), 5), b(tinyShape(), 99);
    auto other = tinyShape();
    other.D = 16;
    fw::nn::Model wrong(other, 1);
    std::vector<float> self(5, 0.3f), ents(4, -0.2f), opt(3, 0.9f);
    ASSERT_TRUE(a.save(dir + "/a.bin"));
    ASSERT_TRUE(b.load(dir + "/a.bin"));
    a.encode({self.data(), ents.data(), 1});
    b.encode({self.data(), ents.data(), 1});
    EXPECT_FLOAT_EQ(a.value(), b.value());
    EXPECT_FLOAT_EQ(a.advantage(0, opt.data(), 0), b.advantage(0, opt.data(), 0));
    EXPECT_FALSE(wrong.load(dir + "/a.bin"));
}

TEST(Episodes, RoundTripAndTargets) {
    auto cfg = quietConfig();
    auto dir = tempDir("eps");
    const fw::Dims dims{3, 2, 2, 2, 2};
    auto write = [&](bool end, bool won, bool eliminated) {
        fw::EpisodeWriter w;
        EXPECT_TRUE(w.open(dir, dims));
        for (int k = 0; k < 100; ++k) {
            fw::Decision d;
            d.net = k % 2;
            d.tick = 1000 + 20 * k;
            const double snap[fw::kSnap] = {500.0 + 10 * k, 100000, 50000, 100000, 1000, 1000, k == 50 ? 900.0 : 0.0, 100.0 * k};
            std::copy(snap, snap + fw::kSnap, d.snap);
            d.self = {0.1f, 0.2f, 0.3f};
            d.ents = {1, 2, 3, 4};
            d.nEnts = 2;
            d.opt = {0.5f, 0.25f};
            d.target = 1;
            w.write(d, "attack \"x\" 20%");
        }
        const double endSnap[fw::kSnap] = {eliminated ? 0.0 : 1500.0, 100000, 0, 100000, 0, 0, 0, 0};
        if (end) w.end(3000, endSnap, won, eliminated);
    };
    write(true, true, false);
    fw::Episode win, lose, cut;
    std::vector<std::string> files;
    auto newest = [&]() {
        std::string best;
        for (const auto& e : std::filesystem::directory_iterator(dir))
            if (std::find(files.begin(), files.end(), e.path().string()) == files.end()) best = e.path().string();
        files.push_back(best);
        return best;
    };
    ASSERT_TRUE(fw::readEpisode(newest(), dims, win));
    write(true, false, true);
    ASSERT_TRUE(fw::readEpisode(newest(), dims, lose));
    write(false, false, false);
    ASSERT_TRUE(fw::readEpisode(newest(), dims, cut));
    EXPECT_FALSE(fw::readEpisode(files[0], fw::Dims{4, 2, 2, 2, 2}, cut));  // other input sizes: skipped
    ASSERT_TRUE(fw::readEpisode(files[2], dims, cut));

    ASSERT_EQ(win.steps.size(), 100u);
    EXPECT_TRUE(win.ended && win.won);
    EXPECT_EQ(win.steps[3].target, 1);
    EXPECT_FLOAT_EQ(win.steps[3].ents[2], 3);

    fw::computeTargets(win, cfg.learning);
    fw::computeTargets(lose, cfg.learning);
    fw::computeTargets(cut, cfg.learning);
    EXPECT_GT(win.steps[90].G, lose.steps[90].G + 5);  // same position, opposite outcomes
    EXPECT_GT(win.steps[90].G, win.steps[10].G);       // the win is closer
    // Forecasts: attacked within 30 s (step 50 was under attack), survival, and unknowns.
    EXPECT_TRUE(win.steps[45].auxMask >> 3 & 1);
    EXPECT_FLOAT_EQ(win.steps[45].aux[3], 1.0f);
    EXPECT_FLOAT_EQ(win.steps[60].aux[3], 0.0f);
    EXPECT_FLOAT_EQ(lose.steps[95].aux[2], 0.0f);  // eliminated within 2 minutes
    EXPECT_FLOAT_EQ(lose.steps[10].aux[2], 1.0f);
    EXPECT_GT(win.steps[10].aux[4], 0.0f);         // gold was earned in the following minute
    // A cut-short game: late decisions have no known result.
    EXPECT_TRUE(cut.steps[10].usable);
    EXPECT_FALSE(cut.steps[95].usable);
    EXPECT_FALSE(cut.steps[95].auxMask >> 2 & 1);
    // ...unless something values the latest position for it.
    fw::computeTargets(cut, cfg.learning, [](const fw::Decision&) { return 4.0; });
    EXPECT_TRUE(cut.steps[80].usable);
    EXPECT_FALSE(cut.steps[95].usable);
    EXPECT_GT(cut.steps[80].G, 2.0f);
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

TEST(Learner, SeesEveryPlayerAndAimsOptionsAtThem) {
    auto cfg = quietConfig();
    fw::Learner learner(cfg, tempDir("ents"));
    auto s = fw::parseState(learnState(1000, 500, 50000));
    auto ft = learner.features(s, false);
    const auto dims = fw::learnerDims();
    EXPECT_EQ(int(ft.self.size()), dims.self);
    EXPECT_EQ(ft.nEnts, 3);
    EXPECT_EQ(int(ft.ents.size()), 3 * dims.ent);
    fw::Plan plan;
    plan.troopBudget = 1e9;
    plan.goldBudget = 1e9;
    for (const auto& o : learner.troopOptions(s, plan, false)) {
        EXPECT_EQ(int(o.x.size()), dims.optTroop) << o.label;
        if (o.label.rfind("attack bot9", 0) == 0) EXPECT_EQ(ft.smalls[o.target], 9);
        if (o.label.rfind("expand", 0) == 0) EXPECT_EQ(o.target, -1);
    }
    for (const auto& o : learner.goldOptions(s, plan)) EXPECT_EQ(int(o.x.size()), dims.optGold) << o.label;
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

TEST(Learner, FollowsResearchAdvice) {
    auto cfg = quietConfig();
    auto dir = tempDir("advice");
    std::filesystem::create_directories(dir);
    const std::string file = dir + "/advice.json";
    auto write = [&](const json& items) { std::ofstream(file) << json{{"items", items}}.dump(); };
    json boat = {{"id", "a1"}, {"strength", 1.0}, {"when", {{"phase", "early"}, {"neutralLand", true}}},
                 {"prefer", {{"decision", "troop"}, {"kind", "expand"}, {"ratioMax", 0.1}}}};
    json typo = {{"id", "a2"}, {"prefer", {{"decision", "troop"}, {"kind", "charge"}}}};
    json off = boat;
    off["id"] = "a3", off["enabled"] = false, off["prefer"]["kind"] = "wait";
    write(json::array({boat, typo, off}));

    std::vector<std::string> errors;
    auto advice = fw::Advice::load(file, &errors);
    EXPECT_EQ(advice.size(), 1u);  // the typo is rejected, the disabled item skipped
    ASSERT_EQ(errors.size(), 1u);
    EXPECT_NE(errors[0].find("charge"), std::string::npos);

    cfg.learning.adviceWeight = 5;  // outweighs the rules' bonus
    fw::Learner learner(cfg, dir, 7, file);
    learner.load();
    fw::Memory mem;
    auto plan = fw::decide(fw::parseState(learnState(1000, 500, 50000)), cfg, mem, &learner);
    bool small = false;
    for (const auto& a : plan.actions)
        small |= a["type"] == "attack" && a["target"].is_null() && std::fabs(a["troops"].get<double>() - 5000) < 1;
    EXPECT_TRUE(small);  // the rules would send 20%; the advice asked for a 10% push into free land
    // Outside the advice's condition (mid game) nothing changes.
    fw::Memory mem2;
    plan = fw::decide(fw::parseState(learnState(5000, 500, 50000)), cfg, mem2, &learner);
    for (const auto& a : plan.actions)
        if (a["type"] == "attack" && a["target"].is_null()) EXPECT_GT(a["troops"].get<double>(), 5001);
}

TEST(Learner, EvalGamesRecordNothing) {
    auto cfg = quietConfig();
    cfg.learning.record = false;
    auto dir = tempDir("norecord");
    fw::Learner learner(cfg, dir);
    playEpisode(learner, cfg, 1.03, true, false, 10);
    EXPECT_FALSE(std::filesystem::exists(dir + "/episodes"));
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

// The whole loop: games are recorded, the trainer learns from the files, a playing engine picks the
// new weights up mid-game and values winning positions above losing ones.
TEST(Trainer, LearnsFromRecordedGamesAndEnginesReload) {
    auto cfg = quietConfig();
    cfg.learning.maxEpisodes = 100;
    auto dir = tempDir("trainer");
    {
        fw::Learner recorder(cfg, dir);
        for (int i = 0; i < 3; ++i) {
            playEpisode(recorder, cfg, 1.03, true, false);
            playEpisode(recorder, cfg, 0.97, false, true);
        }
    }
    fw::Learner player(cfg, dir);
    player.load();
    EXPECT_EQ(player.brain().seen, 0u);

    fw::Trainer trainer(cfg, dir);
    EXPECT_EQ(trainer.refresh(), 6u * 120u);
    EXPECT_EQ(trainer.status().episodes, 6);
    EXPECT_EQ(trainer.status().samples[fw::Learner::kTroop], 360u);
    trainer.trainSteps(300);
    trainer.save();

    // The player notices the new weights at its next reload check.
    fw::Memory mem;
    fw::decide(fw::parseState(learnState(5000, 800, 50000)), cfg, mem, &player);
    EXPECT_EQ(player.reloads(), 1);
    EXPECT_EQ(player.brain().seen, 720u);
    EXPECT_LT(player.priorWeight(fw::Learner::kTroop), cfg.learning.prior);

    auto value = [&](int tiles) {
        auto ft = player.features(fw::parseState(learnState(1600, tiles, 50000)), false);
        auto& m = player.brain().members[0];
        m.encode({ft.self.data(), ft.ents.data(), ft.nEnts});
        return m.value();
    };
    EXPECT_GT(value(1200), value(200) + 0.5f);  // growing (winning) position vs shrinking (losing) one
}

TEST(Trainer, StudiesGamesStillRunning) {
    auto cfg = quietConfig();
    cfg.learning.studyHorizon = 100;
    auto dir = tempDir("running");
    fw::Learner recorder(cfg, dir);
    playEpisode(recorder, cfg, 1.03, true, false);
    playEpisode(recorder, cfg, 0.97, false, true);
    fw::Trainer trainer(cfg, dir);
    trainer.refresh();
    trainer.trainSteps(50);
    // A third game is in progress: no end line yet.
    fw::Memory mem;
    for (int k = 0; k < 40; ++k)
        fw::decide(fw::parseState(learnState(1000 + 20 * k, 500 + 10 * k, 50000)), cfg, mem, &recorder);
    const size_t before = trainer.status().samples[fw::Learner::kTroop];
    trainer.refresh();
    EXPECT_EQ(trainer.status().running, 1);
    const size_t added = trainer.status().samples[fw::Learner::kTroop] - before;
    EXPECT_GT(added, 25u);  // everything older than studyHorizon (100 ticks = 5 decisions)...
    EXPECT_LT(added, 40u);  // ...but not the newest decisions
}
