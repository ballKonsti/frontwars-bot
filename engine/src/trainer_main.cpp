// fw_trainer: learns from the episode files while games are being played.
//   fw_trainer <config>              train until stdin closes (the bot runner holds it open) or SIGTERM
//   fw_trainer <config> --once N     N minibatch steps, print how well the nets fit, exit
//   fw_trainer <config> --bench      steps/second for this network size
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include "config.h"
#include "learn/trainer.h"

namespace {

std::atomic<bool> stop{false};
void onSignal(int) { stop = true; }

void print(const fw::TrainStatus& s, fw::Trainer& t) {
    std::cerr << "trainer: " << s.episodes << " games (" << s.running << " running), " << s.samples[0] << " troop / "
              << s.samples[1] << " gold / " << s.samples[2] << " spawn decisions, " << s.steps << " steps at "
              << int(s.stepsPerSecond) << "/s, " << t.brain().members[0].parameters() << " weights x "
              << t.brain().members.size();
    if (s.valueError >= 0)
        std::cerr << "; held-out: position error " << s.valueError << ", move error " << s.moveError << ", move pairs "
                  << int(100 * s.movePairs + 0.5) << "% (50% = chance)";
    std::cerr << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: fw_trainer <config> [--once STEPS | --bench]\n";
        return 2;
    }
    fw::Config cfg;
    try {
        cfg = fw::loadConfig(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "config: " << e.what() << "\n";
        return 1;
    }
    std::filesystem::path dir = cfg.learning.modelDir;
    if (dir.is_relative()) dir = std::filesystem::path(argv[1]).parent_path() / dir;
    fw::Trainer trainer(cfg, dir.string(), std::random_device{}());
    const std::string mode = argc > 2 ? argv[2] : "";

    if (mode == "--once" || mode == "--bench") {
        const size_t n = trainer.refresh();
        const int steps = mode == "--bench" ? 100 : argc > 3 ? std::atoi(argv[3]) : 2000;
        if (n < 64) std::cerr << "trainer: only " << n << " usable decisions; play more games first\n";
        for (int done = 0; done < steps && n >= 64; done += 100) trainer.trainSteps(std::min(100, steps - done));
        print(trainer.evaluate(), trainer);
        if (mode == "--once") trainer.save();
        return 0;
    }

    std::signal(SIGTERM, onSignal);
    std::signal(SIGINT, onSignal);
    // The runner keeps our stdin open; when it goes away (even by crashing) we stop too.
    std::thread([] {
        std::string line;
        while (std::getline(std::cin, line)) {}
        stop = true;
    }).detach();

    using clock = std::chrono::steady_clock;
    auto lastRefresh = clock::now() - std::chrono::hours(1), lastSave = clock::now(), lastEval = clock::now();
    size_t usable = 0;
    while (!stop) {
        const auto now = clock::now();
        if (now - lastRefresh > std::chrono::seconds(10)) {
            usable = trainer.refresh();
            lastRefresh = now;
        }
        if (usable < 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        trainer.trainSteps(20);
        if (now - lastEval > std::chrono::seconds(60)) {
            print(trainer.evaluate(), trainer);
            lastEval = now;
        }
        if (now - lastSave > std::chrono::seconds(5)) {
            trainer.save();
            lastSave = now;
        }
    }
    // Games that just ended have their outcome now: take one last look before saving.
    if (trainer.refresh() >= 64) trainer.trainSteps(100);
    trainer.evaluate();
    trainer.save();
    print(trainer.status(), trainer);
    return 0;
}
