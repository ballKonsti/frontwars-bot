#include "learn/episodes.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>
#include <unistd.h>

namespace fw {

using nlohmann::json;

double potential(const double s[kSnap], const LearningConfig& lc) {
    return lc.wLand * std::sqrt(std::max(0.0, s[0]) / std::max(1.0, s[1])) +
           lc.wGold * std::log1p(std::max(0.0, s[5]) / 1e5) +
           lc.wTroops * std::clamp(s[2] / std::max(1.0, s[3]), 0.0, 1.5) +
           lc.wCap * std::log1p(std::max(0.0, s[3]) / 1e5);
}

namespace {

void writeFloats(std::FILE* f, const char* key, const std::vector<float>& v) {
    std::fprintf(f, ",\"%s\":[", key);
    for (size_t i = 0; i < v.size(); ++i) std::fprintf(f, i ? ",%.4g" : "%.4g", double(v[i]));
    std::fputc(']', f);
}

void writeSnap(std::FILE* f, const double s[kSnap]) {
    std::fputs("\"s\":[", f);
    for (int i = 0; i < kSnap; ++i) std::fprintf(f, i ? ",%.0f" : "%.0f", s[i]);
    std::fputc(']', f);
}

json dimsJson(const Dims& d) { return json::array({d.self, d.ent, d.optTroop, d.optGold, d.spawn}); }

double share(const double s[kSnap]) { return std::max(0.0, s[0]) / std::max(1.0, s[1]); }

}  // namespace

bool EpisodeWriter::open(const std::string& dir, const Dims& dims) {
    namespace fs = std::filesystem;
    close();
    std::error_code ec;
    fs::create_directories(dir, ec);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
    static int serial = 0;  // several games can start within one millisecond in tests
    fs::path file = fs::path(dir) / ("ep-" + std::to_string(ms) + "-" + std::to_string(::getpid()) + "-" +
                                     std::to_string(serial++) + ".jsonl");
    file_ = std::fopen(file.c_str(), "w");
    if (!file_) return false;
    std::fprintf(file_, "{\"v\":%d,\"dims\":%s}\n", kEpisodeVersion, dimsJson(dims).dump().c_str());
    std::fflush(file_);
    return true;
}

void EpisodeWriter::write(const Decision& d, const std::string& label) {
    if (!file_) return;
    std::fprintf(file_, "{\"n\":%d,\"t\":%d,", d.net, d.tick);
    writeSnap(file_, d.snap);
    if (!d.self.empty()) writeFloats(file_, "self", d.self);
    if (d.nEnts) {
        writeFloats(file_, "ents", d.ents);
        std::fprintf(file_, ",\"ne\":%d", d.nEnts);
    }
    writeFloats(file_, "opt", d.opt);
    std::fprintf(file_, ",\"tg\":%d,\"r\":%d,\"a\":%s}\n", d.target, d.rule ? 1 : 0, json(label).dump().c_str());
    std::fflush(file_);
}

void EpisodeWriter::end(int tick, const double snap[kSnap], bool won, bool eliminated) {
    if (!file_) return;
    std::fprintf(file_, "{\"end\":true,\"t\":%d,", tick);
    writeSnap(file_, snap);
    std::fprintf(file_, ",\"won\":%s,\"eliminated\":%s}\n", won ? "true" : "false", eliminated ? "true" : "false");
    close();
}

void EpisodeWriter::close() {
    if (file_) std::fclose(file_);
    file_ = nullptr;
}

bool readEpisode(const std::string& path, const Dims& dims, Episode& out) {
    out = Episode{};
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    bool header = false;
    const int optSize[3] = {dims.optTroop, dims.optGold, dims.spawn};
    while (std::getline(in, line)) {
        json j = json::parse(line, nullptr, false);
        if (j.is_discarded() || !j.is_object()) break;  // half-written last line of a running game
        if (!header) {
            if (j.value("v", 0) != kEpisodeVersion || j.value("dims", json::array()) != dimsJson(dims)) return false;
            header = true;
            continue;
        }
        if (!j.contains("s") || j["s"].size() != kSnap) break;
        double snap[kSnap];
        for (int k = 0; k < kSnap; ++k) snap[k] = j["s"][k].get<double>();
        if (j.value("end", false)) {
            out.ended = true;
            out.won = j.value("won", false);
            out.eliminated = j.value("eliminated", false);
            out.endTick = j.value("t", 0);
            std::copy(snap, snap + kSnap, out.endSnap);
            break;
        }
        Decision d;
        d.net = j.value("n", 0);
        d.tick = j.value("t", 0);
        std::copy(snap, snap + kSnap, d.snap);
        if (d.net < 0 || d.net > 2) break;
        d.opt = j.value("opt", std::vector<float>{});
        if (int(d.opt.size()) != optSize[d.net]) break;
        if (d.net != 2) {
            d.self = j.value("self", std::vector<float>{});
            d.ents = j.value("ents", std::vector<float>{});
            d.nEnts = j.value("ne", 0);
            if (int(d.self.size()) != dims.self || d.nEnts < 0 || d.nEnts > nn::kMaxEnts ||
                int(d.ents.size()) != d.nEnts * dims.ent)
                break;
        }
        d.target = j.value("tg", -1);
        if (d.target >= d.nEnts) d.target = -1;
        d.rule = j.value("r", 0) != 0;
        out.steps.push_back(std::move(d));
    }
    return header;
}

void computeTargets(Episode& ep, const LearningConfig& lc, const std::function<double(const Decision&)>& bootstrap) {
    auto& st = ep.steps;
    if (st.empty()) return;
    const double gammaTick = std::exp(-1.0 / std::max(1, lc.horizon));
    const int lastTick = ep.ended ? std::max(ep.endTick, st.back().tick) : st.back().tick;

    // Returns, backwards.
    double nextPhi, G;
    int nextTick;
    if (ep.ended) {
        nextPhi = potential(ep.endSnap, lc);
        nextTick = lastTick;
        // Lost games still differ: ending on 30% of the map is worth more than ending on 5%.
        G = (ep.won ? lc.terminalWin : ep.eliminated ? lc.terminalLoss : 0.0) + lc.terminalShare * share(ep.endSnap);
    } else {
        nextPhi = potential(st.back().snap, lc);
        nextTick = st.back().tick;
        G = bootstrap ? bootstrap(st.back()) : 0.0;
    }
    const int unknownWindow = ep.ended ? 0 : bootstrap ? lc.studyHorizon : lc.horizon / 2;
    for (size_t i = st.size(); i-- > 0;) {
        Decision& d = st[i];
        const double phi = potential(d.snap, lc);
        G = (nextPhi - phi) + std::pow(gammaTick, std::max(0, nextTick - d.tick)) * G;
        nextPhi = phi;
        nextTick = d.tick;
        d.G = float(G);
        d.usable = ep.ended || lastTick - d.tick >= unknownWindow;
    }

    // Auxiliary forecasts. `at(tick)` = first step at or after that tick, or -1 if the record ends earlier.
    auto at = [&](int tick) -> int {
        auto it = std::lower_bound(st.begin(), st.end(), tick, [](const Decision& d, int t) { return d.tick < t; });
        return it == st.end() ? -1 : int(it - st.begin());
    };
    const bool died = ep.ended && ep.eliminated;
    for (size_t i = 0; i < st.size(); ++i) {
        Decision& d = st[i];
        d.auxMask = 0;
        auto set = [&](int k, double v) {
            d.aux[k] = float(v);
            d.auxMask |= 1u << k;
        };
        const int horizons[2] = {300, 1200};
        for (int k = 0; k < 2; ++k) {  // land share in 30 s / 2 min
            int j = at(d.tick + horizons[k]);
            if (j >= 0) set(k, 3.0 * std::sqrt(share(st[j].snap)));
            else if (died) set(k, 0.0);
            else if (ep.ended) set(k, 3.0 * std::sqrt(share(ep.endSnap)));
        }
        if (at(d.tick + 1200) >= 0 || (ep.ended && !ep.eliminated)) set(2, 1.0);  // survive 2 min
        else if (died) set(2, 0.0);
        if (int j = at(d.tick + 300); j >= 0 || ep.ended) {  // attacked within 30 s
            bool attacked = false;
            for (size_t k = i + 1; k < st.size() && st[k].tick <= d.tick + 300; ++k) attacked |= st[k].snap[6] > 0;
            set(3, attacked ? 1.0 : 0.0);
        }
        if (int j = at(d.tick + 600); j >= 0)  // gold earned in the next minute
            set(4, std::log1p(std::max(0.0, st[j].snap[7] - d.snap[7]) / 1e5));
    }
}

}  // namespace fw
