// fw_engine: reads one game-state JSON per line on stdin, writes one {"actions":[...],"log":[...]} line per state.
// A {"gameOver":true,"won":bool,"eliminated":bool} line ends the game for the learner (no reply).
// Learning happens in fw_trainer; this process only plays, records its decisions, and picks up new
// weights while the game runs.
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "config.h"
#include "learn/learner.h"
#include "state.h"
#include "strategy/planner.h"

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    fw::Config cfg;
    std::filesystem::path configDir = ".";
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
        std::cerr << "learner: " << learner->brain().seen << " decisions of experience, "
                  << learner->brain().members[0].parameters() << " weights x " << learner->brain().members.size()
                  << " (" << cfg.learning.size << ") in " << learner->brain().dir << "\n";
    }
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
    if (learner) std::cerr << "learner: picked up new weights " << learner->reloads() << " times during the game\n";
    return 0;
}
