#include "learn/trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>

namespace fw {

namespace {

constexpr size_t kMinSamples = 64;
constexpr size_t kMinSpawnSamples = 12;

void meanSd(const std::vector<const Decision*>& set, double& mean, double& sd) {
    mean = 0;
    for (const Decision* d : set) mean += d->G;
    mean /= double(std::max<size_t>(1, set.size()));
    double var = 0;
    for (const Decision* d : set) var += (d->G - mean) * (d->G - mean);
    sd = std::sqrt(var / double(std::max<size_t>(1, set.size()))) + 1e-6;
}

}  // namespace

Trainer::Trainer(const Config& cfg, std::string modelDir, uint32_t seed)
    : cfg_(cfg), lc_(cfg.learning), dir_(std::move(modelDir)), brain_(cfg.learning, dir_, seed), rng_(seed) {
    brain_.load();
}

size_t Trainer::refresh() {
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<fs::path> paths;
    for (const auto& e : fs::directory_iterator(fs::path(dir_) / "episodes", ec))
        if (e.path().extension() == ".jsonl") paths.push_back(e.path());
    std::sort(paths.begin(), paths.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename().string() > b.filename().string();  // names start with a millisecond timestamp
    });
    const size_t keep = size_t(std::max(1, lc_.maxEpisodes));
    for (size_t i = keep; i < paths.size(); ++i) fs::remove(paths[i], ec);
    if (paths.size() > keep) paths.resize(keep);

    // Parse only what changed since last time.
    const Dims dims = learnerDims();
    std::map<std::string, Cached> next;
    for (const auto& p : paths) {
        const std::string key = p.string();
        const auto time = fs::last_write_time(p, ec);
        const auto size = fs::file_size(p, ec);
        auto it = files_.find(key);
        if (it != files_.end() && it->second.time == time && it->second.size == size) {
            next[key] = std::move(it->second);
            continue;
        }
        Cached c;
        c.time = time;
        c.size = size;
        c.ok = readEpisode(key, dims, c.ep) && c.ep.steps.size() >= 2;
        next[key] = std::move(c);
    }
    files_ = std::move(next);

    // A game still running has no outcome yet: its latest position is valued by the nets themselves.
    const bool trained = brain_.seen >= kMinSamples;
    auto bootstrap = [&](const Decision& d) {
        if (d.net == Learner::kSpawn || d.self.empty()) return 0.0;
        double v = 0;
        for (auto& m : brain_.members) {
            m.encode({d.self.data(), d.ents.data(), d.nEnts});
            v += m.value();
        }
        return v / double(brain_.members.size()) * brain_.sd + brain_.mean;
    };

    train_.clear();
    val_.clear();
    spawn_.clear();
    fold_.clear();
    const size_t folds = std::max<size_t>(1, brain_.members.size());
    status_.episodes = status_.running = 0;
    const size_t cap = size_t(std::max(1, lc_.replay));
    const bool holdOut = files_.size() >= 8;
    // Newest first, so the cap drops the oldest games.
    for (auto it = files_.rbegin(); it != files_.rend(); ++it) {
        Cached& c = it->second;
        if (!c.ok) continue;
        if (!c.ep.ended) ++status_.running;
        // Before the nets know anything, a running game is used without that estimate (its last
        // decisions are then left out until their result is in).
        computeTargets(c.ep, lc_, c.ep.ended || !trained ? nullptr : std::function<double(const Decision&)>(bootstrap));
        ++status_.episodes;
        // Held-out games are chosen by name, so a game never moves between the two sets.
        const bool isVal = holdOut && c.ep.ended && std::hash<std::string>{}(it->first) % 8 == 7;
        for (const Decision& d : c.ep.steps) {
            if (!d.usable) continue;
            if (d.net == Learner::kSpawn) spawn_.push_back(&d);
            else if (train_.size() + val_.size() < cap) {
                (isVal ? val_ : train_).push_back(&d);
                if (!isVal) fold_.push_back(uint8_t((std::hash<std::string>{}(it->first) / 8) % folds));
            }
        }
    }
    status_.samples[0] = status_.samples[1] = 0;
    for (auto* set : {&train_, &val_})
        for (const Decision* d : *set) ++status_.samples[d->net];
    status_.samples[2] = spawn_.size();
    status_.heldOut = val_.size();
    // Returns are standardised, so scores mean the same however the rewards are weighted.
    if (train_.size() >= kMinSamples) {
        meanSd(train_, brain_.mean, brain_.sd);
        brain_.seen = train_.size() + val_.size();
    }
    if (spawn_.size() >= kMinSpawnSamples) {
        meanSd(spawn_, brain_.spawnMean, brain_.spawnSd);
        brain_.spawnSeen = spawn_.size();
    }
    return train_.size();
}

// Cross-fitting. Every game belongs to one fold, and member k never trains on fold k. That gives each
// member its own 80% of the games (so their disagreement measures ignorance), and it gives every
// decision an honest baseline: the position value from the one member that has not seen its game.
// The advantage heads learn "return minus that baseline" - what the move added beyond what the position
// promised. With the model's own V as baseline, V would memorise its games and leave nothing to learn.
void Trainer::trainMember(int member, int steps, uint32_t seed, const std::vector<nn::Model>* frozen) {
    nn::Model& model = brain_.members[member];
    const size_t size = train_.size();
    const int batch = int(std::min<size_t>(std::max(1, lc_.batch), size));
    const int folds = int(brain_.members.size());
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, size - 1);
    std::vector<nn::Model> judges;  // this thread's own copies: a model's caches are not shareable
    if (frozen) judges = *frozen;
    for (int k = 0; k < steps; ++k) {
        for (int i = 0; i < batch; ++i) {
            size_t idx = pick(rng);
            for (int tries = 0; folds > 1 && fold_[idx] == member && tries < 64; ++tries) idx = pick(rng);
            const Decision& d = *train_[idx];
            const nn::Input in{d.self.data(), d.ents.data(), d.nEnts};
            nn::Target t;
            t.net = d.net;
            t.opt = d.opt.data();
            t.target = d.target;
            t.G = normG(d);
            t.aux = d.aux;
            t.auxMask = d.auxMask;
            if (!judges.empty()) {
                judges[fold_[idx]].encode(in);
                t.hasBaseline = true;
                t.baseline = judges[fold_[idx]].value();
            }
            model.accumulate(in, t);
        }
        model.step(float(lc_.lr), float(lc_.weightDecay), batch);
    }
}

void Trainer::trainSpawn(int steps) {
    if (spawn_.size() < kMinSpawnSamples) return;
    const size_t batch = std::min<size_t>(32, spawn_.size());
    std::uniform_int_distribution<size_t> pick(0, spawn_.size() - 1);
    std::vector<const std::vector<float>*> xs(batch);
    std::vector<float> ys(batch);
    for (auto& net : brain_.spawn)
        for (int k = 0; k < steps; ++k) {
            for (size_t i = 0; i < batch; ++i) {
                const Decision& d = *spawn_[pick(rng_)];
                xs[i] = &d.opt;
                ys[i] = float((d.G - brain_.spawnMean) / brain_.spawnSd);
            }
            net.train(xs, ys, 0.3f * float(lc_.lr), 2.0f);
        }
}

void Trainer::trainSteps(int steps) {
    if (train_.size() < kMinSamples) return;
    const auto t0 = std::chrono::steady_clock::now();
    // The judges are a snapshot of the ensemble, refreshed every call (a few seconds of training).
    const std::vector<nn::Model> frozen = brain_.members;
    const bool crossFit = frozen.size() > 1;
    std::vector<std::thread> threads;
    for (int m = 0; m < int(brain_.members.size()); ++m)
        threads.emplace_back(&Trainer::trainMember, this, m, steps, uint32_t(rng_()), crossFit ? &frozen : nullptr);
    for (auto& t : threads) t.join();
    // The spawn net sees one decision per game: a step now and then is plenty.
    trainSpawn(std::max(1, steps / 20));
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    status_.steps += steps;
    status_.stepsPerSecond = secs > 0 ? steps / secs : 0;
}

TrainStatus Trainer::evaluate() {
    status_.valueError = status_.moveError = status_.movePairs = -1;
    if (val_.size() < 64 || brain_.seen < kMinSamples) return status_;
    // Ensemble predictions on held-out decisions.
    std::vector<double> resid(val_.size()), adv(val_.size());
    double ve = 0, ae = 0, rr = 0;
    for (size_t i = 0; i < val_.size(); ++i) {
        const Decision& d = *val_[i];
        double v = 0, a = 0;
        for (auto& m : brain_.members) {
            m.encode({d.self.data(), d.ents.data(), d.nEnts});
            v += m.value();
            a += m.advantage(d.net, d.opt.data(), d.target);
        }
        v /= double(brain_.members.size());
        a /= double(brain_.members.size());
        const double g = normG(d);
        resid[i] = g - v;
        adv[i] = a;
        ve += (v - g) * (v - g);
        ae += (a - resid[i]) * (a - resid[i]);
        rr += resid[i] * resid[i];
    }
    status_.valueError = ve / double(val_.size());  // G is standardised: its variance is ~1
    status_.moveError = rr > 0 ? ae / rr : -1;
    std::uniform_int_distribution<size_t> pick(0, val_.size() - 1);
    int right = 0, total = 0;
    for (int k = 0; k < 4000; ++k) {
        size_t i = pick(rng_), j = pick(rng_);
        if (i == j || adv[i] == adv[j] || resid[i] == resid[j]) continue;
        ++total;
        right += (adv[i] > adv[j]) == (resid[i] > resid[j]);
    }
    status_.movePairs = total ? double(right) / total : -1;
    return status_;
}

void Trainer::save() {
    const TrainStatus& s = status_;
    brain_.save({{"size", lc_.size},
                 {"episodes", s.episodes},
                 {"running", s.running},
                 {"troopDecisions", s.samples[0]},
                 {"goldDecisions", s.samples[1]},
                 {"spawnDecisions", s.samples[2]},
                 {"heldOut", s.heldOut},
                 {"valueError", s.valueError},
                 {"moveError", s.moveError},
                 {"movePairs", s.movePairs},
                 {"steps", s.steps},
                 {"stepsPerSecond", s.stepsPerSecond},
                 {"priorTroop", std::max(std::min(lc_.prior, lc_.priorFloor),
                                         lc_.prior * std::pow(0.5, double(brain_.seen) / std::max(1.0, lc_.priorHalfLife)))},
                 {"updated", std::chrono::duration_cast<std::chrono::seconds>(
                                 std::chrono::system_clock::now().time_since_epoch()).count()}});
}

}  // namespace fw
