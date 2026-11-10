// fw_engine: reads one game-state JSON per line on stdin, writes one {"actions":[...],"log":[...]} line per state.
// A {"gameOver":true,"won":bool,"eliminated":bool} line ends the game for the learner (no reply).
// When stdin closes, the networks are retrained on every recorded game and saved.
//   fw_engine <config>                play
//   fw_engine --train <config> [N]    only retrain (N minibatch steps) and print how well the nets fit
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "config.h"
#include "learn/learner.h"
#include "state.h"
#include "strategy/planner.h"

namespace {

void trainAndSave(fw::Learner& learner, int steps) {
    fw::TrainReport r = learner.train(steps);
    learner.save();
    std::cerr << "learner: trained on " << r.episodes << " games, " << r.samples[0] << " troop / " << r.samples[1]
              << " gold / " << r.samples[2] << " spawn decisions";
    for (int net : {0, 1})
        if (r.valAfter[net] >= 0)
            std::cerr << "; " << (net ? "gold" : "troop") << " held-out error " << r.valBefore[net] << " -> "
                      << r.valAfter[net] << " (1.0 = knows nothing)";
    std::cerr << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    fw::Config cfg;
    std::filesystem::path configDir = ".";
    bool trainOnly = argc > 1 && std::string(argv[1]) == "--train";
    if (trainOnly) {
        --argc;
        ++argv;
    }
    if (argc > 1) {
        configDir = std::filesystem::path(argv[1]).parent_path();
        try {
            cfg = fw::loadConfig(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "config: " << e.what() << ", using defaults\n";
        }
    }

    std::unique_ptr<fw::Learner> learner;
    if (cfg.learning.enabled) {
        std::filesystem::path dir = cfg.learning.modelDir;
        if (dir.is_relative()) dir = configDir / dir;
        learner = std::make_unique<fw::Learner>(cfg, dir.string(), std::random_device{}());
        learner->load();
        std::cerr << "learner: " << learner->samples(fw::Learner::kTroop) << " troop / "
                  << learner->samples(fw::Learner::kGold) << " gold decisions of experience in " << dir.string()
                  << "\n";
    }
    if (trainOnly) {
        if (!learner) {
            std::cerr << "learning is disabled in the config\n";
            return 1;
        }
        trainAndSave(*learner, argc > 2 ? std::atoi(argv[2]) : 10 * cfg.learning.trainSteps);
        return 0;
    }

    if (learner) learner->startStudy();
    fw::Memory mem;
    std::string line;
    while (std::getline(std::cin, line)) {
        nlohmann::json out{{"actions", nlohmann::json::array()}, {"log", nlohmann::json::array()}};
        try {
            auto j = nlohmann::json::parse(line);
            if (j.value("gameOver", false)) {
                if (learner) learner->gameOver(j.value("won", false), j.value("eliminated", false));
                continue;
            }
            fw::Plan plan = fw::decide(fw::parseState(j), cfg, mem, learner.get());
            out["actions"] = plan.actions;
            out["log"] = plan.log;
        } catch (const std::exception& e) {
            out["log"].push_back(std::string("engine error: ") + e.what());
        }
        std::cout << out.dump() << '\n' << std::flush;
    }
    if (learner && cfg.learning.record) {
        trainAndSave(*learner, cfg.learning.trainSteps);
    }
    return 0;
}
