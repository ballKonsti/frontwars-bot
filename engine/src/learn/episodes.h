#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "config.h"
#include "learn/model.h"

namespace fw {

constexpr int kEpisodeVersion = 4;
constexpr int kSnap = 8;  // tiles, landTiles, troops, maxTroops, gold, wealth, incoming troops, gold earned so far

// Input sizes an episode file was written with; files with other sizes are skipped.
struct Dims {
    int self = 0, ent = 0, optTroop = 0, optGold = 0, spawn = 0;
    bool operator==(const Dims&) const = default;
};

// One recorded decision: what the bot saw, the option it took, and (after computeTargets) what came of it.
struct Decision {
    int net = 0;  // 0 troop, 1 gold, 2 spawn
    int tick = 0;
    double snap[kSnap] = {};
    std::vector<float> self, ents, opt;  // spawn decisions only use opt
    int nEnts = 0;
    int target = -1;  // index into ents of the player the option is aimed at
    bool rule = false;

    float G = 0;  // discounted future gain in position value (raw units)
    float aux[nn::kAux] = {};
    unsigned auxMask = 0;
    bool usable = true;  // false: too close to the end of a cut-short game to know its result
};

struct Episode {
    std::vector<Decision> steps;
    bool ended = false, won = false, eliminated = false;
    double endSnap[kSnap] = {};
    int endTick = 0;
};

// Position value of a snapshot: land share, wealth and army, weighted by the learning config.
double potential(const double snap[kSnap], const LearningConfig& lc);

// Appends decisions to <dir>/ep-<time>-<pid>-<n>.jsonl, flushing every line so a running game can be
// read by the trainer and a killed one loses nothing.
class EpisodeWriter {
public:
    ~EpisodeWriter() { close(); }
    bool open(const std::string& dir, const Dims& dims);
    bool isOpen() const { return file_ != nullptr; }
    void write(const Decision& d, const std::string& label);
    void end(int tick, const double snap[kSnap], bool won, bool eliminated);
    void close();

private:
    std::FILE* file_ = nullptr;
};

// False if the file is missing, from another version, or written with other input sizes.
// A half-written last line (game still running) is ignored.
bool readEpisode(const std::string& path, const Dims& dims, Episode& out);

// Fills G, aux and usable for every step.
// G_i = (phi_{i+1} - phi_i) + gamma^dt * G_{i+1}; the last term is the outcome for finished games.
// For a game still running, `bootstrap` (if given) estimates the value of its latest position, and
// only decisions at least studyHorizon old are usable; without it, at least horizon/2 old.
void computeTargets(Episode& ep, const LearningConfig& lc,
                    const std::function<double(const Decision&)>& bootstrap = nullptr);

}  // namespace fw
