#include "learn/learner.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <unistd.h>

namespace fw {

namespace {

constexpr int kEntFeatures = 18;
constexpr int kTroopOptionFeatures = 18;
constexpr int kGoldOptionFeatures = 12;
constexpr int kSpawnPageFeatures = 9;   // computed by the page per candidate
constexpr int kSpawnFeatures = kSpawnPageFeatures + 2;
constexpr double kRatios[] = {0.10, 0.20, 0.35, 0.50, 0.75};
constexpr double kBoatRatio = 0.30;
constexpr int kMaxAttackTargets = 10;
constexpr int kMaxBoatTargets = 3;
constexpr int kReloadEveryTicks = 20;

const char* kGoldKinds[] = {"save",         "City",         "Factory",      "Port",   "Defense Post",
                            "upgrade City", "Missile Silo", "SAM Launcher", "Warship"};
constexpr int kNumGoldKinds = sizeof kGoldKinds / sizeof *kGoldKinds;

float clampf(double v, double lo, double hi) { return float(std::min(hi, std::max(lo, v))); }
float logRatio(double a, double b) { return clampf(std::log((a + 1.0) / (b + 1.0)) / 3.0, -1.0, 1.0); }

std::vector<float> stateFeatures(const State& s, bool overextended, double income) {
    const Me& m = s.me;
    const Player* leader = s.leader();
    int hostileNeighbors = 0, botNeighbors = 0, allies = 0, cityLevels = 0;
    double strongest = 0, neighborTroops = 0, outTroops = 0;
    for (const auto& p : s.players) {
        if (p.allied) ++allies;
        if (p.border == 0 || p.friendly || p.allied) continue;
        if (p.isBot()) {
            ++botNeighbors;
        } else {
            ++hostileNeighbors;
            strongest = std::max(strongest, p.troops);
            neighborTroops += p.troops;
        }
    }
    for (const auto& u : m.units)
        if (u.type == "City") cityLevels += u.level;
    for (const auto& a : m.outgoing) outTroops += a.troops;

    std::vector<float> f;
    f.reserve(40);
    f.push_back(clampf(s.landShare() * 4, 0, 4));
    f.push_back(float(std::log1p(m.tiles) / 12));
    f.push_back(clampf(m.troops / std::max(1.0, m.maxTroops), 0, 1.5));
    f.push_back(float(std::log1p(m.maxTroops) / 16));
    f.push_back(float(std::log1p(std::max(0.0, m.troopRate)) / 10));
    f.push_back(float(std::log1p(m.gold) / 17));
    f.push_back(s.neutralBorder > 0 ? 1.f : 0.f);
    f.push_back(float(std::log1p(s.neutralBorder) / 7));
    f.push_back(clampf(m.incomingTroops() / std::max(1.0, m.troops), 0, 3) / 3);
    f.push_back(clampf(m.outgoing.size() / 3.0, 0, 1));
    f.push_back(clampf(m.count("City") / 10.0, 0, 2));
    f.push_back(clampf(m.count("Factory") / 10.0, 0, 2));
    f.push_back(clampf(m.count("Port") / 5.0, 0, 2));
    f.push_back(clampf(m.count("Defense Post") / 5.0, 0, 2));
    f.push_back(clampf(m.count("Missile Silo") / 3.0, 0, 2));
    f.push_back(clampf(m.count("SAM Launcher") / 3.0, 0, 2));
    f.push_back(m.coastal ? 1.f : 0.f);
    f.push_back(overextended ? 1.f : 0.f);
    f.push_back(float(std::log1p(m.borderLen) / 9));
    f.push_back(leader ? logRatio(leader->tiles, m.tiles) : 0.f);
    f.push_back(s.nukesInbound > 0 ? 1.f : 0.f);
    f.push_back(clampf(s.tick / 30000.0, 0, 1));
    // v2: how dangerous the neighbourhood is, and what we already have in motion.
    f.push_back(clampf(hostileNeighbors / 5.0, 0, 2));
    f.push_back(clampf(botNeighbors / 5.0, 0, 2));
    f.push_back(hostileNeighbors ? logRatio(strongest, m.troops) : -1.f);
    f.push_back(hostileNeighbors ? logRatio(neighborTroops, m.troops) : -1.f);
    f.push_back(clampf(allies / 3.0, 0, 2));
    f.push_back(leader ? clampf(2.0 * leader->tiles / s.landTiles, 0, 2) : 0.f);
    f.push_back(clampf(m.count("Transport") / 3.0, 0, 1));
    f.push_back(clampf(outTroops / std::max(1.0, m.maxTroops), 0, 1));
    f.push_back(clampf(m.incoming.size() / 3.0, 0, 1));
    f.push_back(clampf(m.count("Warship") / 3.0, 0, 1));
    f.push_back(clampf(cityLevels / 20.0, 0, 2));
    f.push_back(float(std::log1p(std::max(0.0, income) * 600) / 15));  // gold per minute
    return f;
}


double estimatedCost(double base, int owned, const EconomyConfig& cfg) { return base * (1.0 + cfg.costGrowth * owned); }

int selfFeatureCount() {
    static const int n = int(stateFeatures(State{}, false, 0).size());
    return n;
}

}  // namespace

Dims learnerDims() { return {selfFeatureCount(), kEntFeatures, kTroopOptionFeatures, kGoldOptionFeatures, kSpawnFeatures}; }

nn::Shape modelShape(const LearningConfig& lc) {
    nn::Shape s;
    const Dims d = learnerDims();
    s.self = d.self;
    s.ent = d.ent;
    s.opt[0] = d.optTroop;
    s.opt[1] = d.optGold;
    s.aux = nn::kAux;
    if (lc.size == "small") {
        s.D = 64, s.E = 32, s.heads = 2, s.blocks = 1;
    } else if (lc.size == "medium") {
        s.D = 128, s.E = 48, s.heads = 4, s.blocks = 2;
    } else {
        s.D = 192, s.E = 64, s.heads = 4, s.blocks = 3;
    }
    return s;
}

// ---------------------------------------------------------------- brain (the weights)

Brain::Brain(const LearningConfig& lc, const std::string& modelDir, uint32_t seed)
    : dir((std::filesystem::path(modelDir) / ("v4-" + lc.size)).string()) {
    const int k = std::max(1, lc.ensemble);
    const nn::Shape shape = modelShape(lc);
    for (int m = 0; m < k; ++m) {
        members.emplace_back(shape, seed + 7 * m);
        spawn.emplace_back(std::vector<int>{kSpawnFeatures, 24, 12, 1}, seed + 1000 + 7 * m);
    }
}

bool Brain::load() {
    namespace fs = std::filesystem;
    std::ifstream in(fs::path(dir) / "status.json");
    auto j = in ? nlohmann::json::parse(in, nullptr, false) : nlohmann::json();
    if (!j.is_object() || j.value("v", 0) != kEpisodeVersion) return false;
    // Load into copies first: a trainer may be replacing files while we read.
    auto m2 = members;
    auto s2 = spawn;
    for (size_t m = 0; m < m2.size(); ++m) {
        if (!m2[m].load((fs::path(dir) / ("member." + std::to_string(m) + ".bin")).string())) return false;
        s2[m].load((fs::path(dir) / ("spawn." + std::to_string(m) + ".nn")).string());
    }
    members = std::move(m2);
    spawn = std::move(s2);
    seen = j.value("seen", size_t(0));
    spawnSeen = j.value("spawnSeen", size_t(0));
    mean = j.value("mean", 0.0);
    sd = j.value("sd", 1.0);
    spawnMean = j.value("spawnMean", 0.0);
    spawnSd = j.value("spawnSd", 1.0);
    return true;
}

void Brain::save(const nlohmann::json& extra) const {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);
    const std::string suffix = ".tmp" + std::to_string(::getpid());
    auto put = [&](const std::string& name, auto&& write) {
        const fs::path file = fs::path(dir) / name, tmp = file.string() + suffix;
        if (write(tmp.string())) fs::rename(tmp, file, ec);
    };
    for (size_t m = 0; m < members.size(); ++m) {
        put("member." + std::to_string(m) + ".bin", [&](const std::string& p) { return members[m].save(p); });
        put("spawn." + std::to_string(m) + ".nn", [&](const std::string& p) { return spawn[m].save(p); });
    }
    nlohmann::json status = extra.is_object() ? extra : nlohmann::json::object();
    status.update({{"v", kEpisodeVersion}, {"ensemble", members.size()}, {"seen", seen}, {"spawnSeen", spawnSeen},
                   {"mean", mean}, {"sd", sd}, {"spawnMean", spawnMean}, {"spawnSd", spawnSd},
                   {"parameters", members.empty() ? 0 : members[0].parameters()}});
    put("status.json", [&](const std::string& p) {  // written last: engines reload when it changes
        std::ofstream out(p);
        out << status.dump() << '\n';
        return bool(out);
    });
}

// ---------------------------------------------------------------- features

int Features::index(int small) const {
    for (size_t i = 0; i < smalls.size(); ++i)
        if (smalls[i] == small) return int(i);
    return -1;
}

// Our own state plus one feature row per visible player: every neighbour (widest border first), then
// the biggest others, up to nn::kMaxEnts.
Features Learner::features(const State& s, bool overextended) const {
    Features ft;
    ft.self = stateFeatures(s, overextended, income_);
    const Me& m = s.me;
    const double myDensity = m.troops / std::max(1, m.tiles);
    const Player* leader = s.leader();
    std::vector<const Player*> order;
    for (const auto& p : s.players) order.push_back(&p);
    std::sort(order.begin(), order.end(), [](const Player* a, const Player* b) {
        return a->border != b->border ? a->border > b->border : a->tiles > b->tiles;
    });
    if (int(order.size()) > nn::kMaxEnts) order.resize(nn::kMaxEnts);
    for (const Player* p : order) {
        int silos = 0, posts = 0;
        for (const auto& u : s.enemyStructures)
            if (u.owner == p->small) {
                silos += u.type == "Missile Silo";
                posts += u.type == "Defense Post";
            }
        bool hostile = std::any_of(m.incoming.begin(), m.incoming.end(),
                                   [&](const Incoming& a) { return a.attacker == p->small; });
        auto& f = ft.ents;
        f.push_back(float(std::log1p(p->tiles) / 12));
        f.push_back(clampf(4.0 * p->tiles / s.landTiles, 0, 4));
        f.push_back(logRatio(p->troops, m.troops));
        f.push_back(logRatio(p->troops / std::max(1, p->tiles), myDensity));
        f.push_back(clampf(double(p->border) / std::max(1, m.borderLen), 0, 1));
        f.push_back(p->border > 0);
        f.push_back(p->isBot());
        f.push_back(!p->isBot() && p->type != "HUMAN");
        f.push_back(p->allied);
        f.push_back(p->friendly);
        f.push_back(hostile);
        f.push_back(m.attackOn(p->small) != nullptr);
        f.push_back(p->underAttack > 0);
        f.push_back(p == leader);
        f.push_back(p->isTraitor);
        f.push_back(float(std::log1p(p->gold) / 17));
        f.push_back(clampf(silos / 3.0, 0, 2));
        f.push_back(clampf(posts / 5.0, 0, 2));
        ft.smalls.push_back(p->small);
    }
    ft.nEnts = int(ft.smalls.size());
    return ft;
}

Learner::Learner(const Config& cfg, std::string modelDir, uint32_t seed)
    : cfg_(cfg), lc_(cfg.learning), dir_(std::move(modelDir)), brain_(cfg.learning, dir_, seed), rng_(seed) {}

Learner::~Learner() = default;

void Learner::load() {
    brain_.load();
    namespace fs = std::filesystem;
    std::error_code ec;
    statusTime_ = fs::last_write_time(fs::path(brain_.dir) / "status.json", ec).time_since_epoch().count();
}

// The trainer publishes new weights every few seconds; pick them up while we play.
void Learner::reloadIfNewer() {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto t = fs::last_write_time(fs::path(brain_.dir) / "status.json", ec).time_since_epoch().count();
    if (ec || t == statusTime_) return;
    if (brain_.load()) {
        statusTime_ = t;
        ++reloads_;
    }
}

// ---------------------------------------------------------------- options

std::vector<Option> Learner::troopOptions(const State& s, const Plan& plan, bool overextended) const {
    const Features ft = features(s, overextended);
    const Me& m = s.me;
    const double myDensity = m.troops / std::max(1, m.tiles);
    const Player* leader = s.leader();

    auto make = [&](const char* kind, double ratio, const Player* p, const BoatTarget* b) {
        Option o;
        o.x.reserve(kTroopOptionFeatures);
        if (p) o.target = ft.index(p->small);
        for (const char* k : {"wait", "expand", "attack", "boat"}) o.x.push_back(std::strcmp(k, kind) == 0);
        o.x.push_back(float(ratio));
        const double sent = ratio * m.troops;
        if (p) {
            bool hostile = std::any_of(m.incoming.begin(), m.incoming.end(),
                                       [&](const Incoming& a) { return a.attacker == p->small; });
            o.x.push_back(float(std::log1p(p->tiles) / 12));
            o.x.push_back(logRatio(p->troops / std::max(1, p->tiles), myDensity));
            o.x.push_back(logRatio(p->troops, m.troops));
            o.x.push_back(clampf(double(p->border) / std::max(1, m.borderLen), 0, 1));
            o.x.push_back(p->isBot());
            o.x.push_back(p->underAttack > 0);
            o.x.push_back(hostile);
            o.x.push_back(p == leader);
            o.x.push_back(0.f);
            o.x.push_back(m.attackOn(p->small) != nullptr);
            o.x.push_back(logRatio(sent, p->troops));
            o.x.push_back(clampf(4.0 * p->tiles / s.landTiles, 0, 2));
            o.sane = p->isBot() || hostile || p->troops < 1.5 * m.troops;
            // Never explored at random: a big push into a stronger player who isn't even attacking us.
            o.reckless = !p->isBot() && !hostile && ratio >= 0.5 && p->troops > m.troops;
        } else {
            for (int i = 0; i < 8; ++i) o.x.push_back(0.f);
            bool neutral = std::strcmp(kind, "wait") != 0;
            o.x.push_back(neutral);
            o.x.push_back(neutral && m.attackOn(0) != nullptr);
            o.x.push_back(0.f);
            o.x.push_back(0.f);
            o.sane = neutral;
        }
        o.x.push_back(b ? clampf(b->dist / 1000.0, 0, 2) : 0.f);
        o.ratio = ratio;
        o.troops = sent;
        return o;
    };
    auto affordable = [&](double troops) { return troops >= 1 && troops <= plan.troopBudget + 1e-6; };

    std::vector<Option> out;
    Option wait = make("wait", 0, nullptr, nullptr);
    wait.label = "wait";
    wait.troops = 0;
    out.push_back(std::move(wait));
    if (s.inSpawn || overextended) return out;

    if (s.neutralBorder > 0 && m.troops >= cfg_.expand.minTroopFrac * m.maxTroops)
        for (double r : kRatios) {
            Option o = make("expand", r, nullptr, nullptr);
            if (!affordable(o.troops)) continue;
            o.label = "expand " + std::to_string(int(r * 100)) + "%";
            o.action = {{"type", "attack"}, {"target", nullptr}};
            out.push_back(std::move(o));
        }

    int running = int(std::count_if(m.outgoing.begin(), m.outgoing.end(), [](const Attack& a) { return a.target; }));
    std::vector<const Player*> targets;
    for (const auto& p : s.players)
        if (p.border > 0 && !p.friendly && !p.allied && (running < cfg_.military.maxConcurrent || m.attackOn(p.small)))
            targets.push_back(&p);
    std::sort(targets.begin(), targets.end(), [](const Player* a, const Player* b) { return a->border > b->border; });
    if (int(targets.size()) > kMaxAttackTargets) targets.resize(kMaxAttackTargets);
    for (const Player* p : targets)
        for (double r : kRatios) {
            Option o = make("attack", r, p, nullptr);
            if (!affordable(o.troops)) continue;
            o.label = "attack " + p->name + " " + std::to_string(int(r * 100)) + "%";
            o.action = {{"type", "attack"}, {"target", p->id}};
            out.push_back(std::move(o));
        }

    if (cfg_.naval.enabled && m.coastal && m.count("Transport") < cfg_.naval.maxBoats) {
        int n = 0;
        for (const auto& b : s.boatTargets) {
            const Player* owner = b.owner ? s.player(b.owner) : nullptr;
            if (b.owner && (!owner || owner->friendly || owner->allied)) continue;
            Option o = make("boat", kBoatRatio, owner, &b);
            if (!affordable(o.troops)) continue;
            o.label = std::string("boat to ") + (owner ? owner->name : "free land");
            o.action = {{"type", "boat"}, {"dst", b.tile}};
            o.action["target"] = owner ? nlohmann::json(owner->id) : nlohmann::json(nullptr);
            out.push_back(std::move(o));
            if (++n >= kMaxBoatTargets) break;
        }
    }

    // Full troops regenerate nothing, so sitting on them is strictly worse than any sane move.
    if (m.troops >= lc_.fullTroops * m.maxTroops &&
        std::any_of(out.begin() + 1, out.end(), [](const Option& o) { return o.sane; }))
        out.erase(out.begin());
    return out;
}

std::vector<Option> Learner::goldOptions(const State& s, const Plan& plan) const {
    const Features ft = features(s, plan.overextended);
    const Me& m = s.me;
    const auto& ec = cfg_.economy;
    const auto& sc = cfg_.strategic;
    const int cities = m.count("City"), factories = m.count("Factory"), ports = m.count("Port"),
              posts = m.count("Defense Post"), silos = m.count("Missile Silo"), sams = m.count("SAM Launcher"),
              warships = m.count("Warship");
    auto cost = [&](const char* unit, double base, int owned) { return s.cost(unit, estimatedCost(base, owned, ec)); };

    const Incoming* worst = nullptr;
    for (const auto& a : m.incoming)
        if (a.attacker != 0 && (!worst || a.troops > worst->troops)) worst = &a;
    const Unit* lowCity = nullptr;
    for (const auto& u : m.units)
        if (u.type == "City" && (!lowCity || u.level < lowCity->level)) lowCity = &u;

    std::vector<Option> out;
    auto add = [&](int kind, double cost, int owned, nlohmann::json action, int near = 0) {
        if (kind != 0 && cost > plan.goldBudget) return;
        Option o;
        if (near) o.target = ft.index(near);
        for (int k = 0; k < kNumGoldKinds; ++k) o.x.push_back(k == kind);
        o.x.push_back(clampf(cost / (m.gold + 1.0), 0, 1));
        o.x.push_back(clampf(owned / 10.0, 0, 2));
        o.x.push_back(float(std::log1p(cost) / 17));
        o.label = kGoldKinds[kind];
        o.gold = cost;
        o.action = std::move(action);
        out.push_back(std::move(o));
    };
    auto build = [](const char* unit, const char* site) {
        return nlohmann::json{{"type", "build"}, {"unit", unit}, {"site", site}};
    };

    add(0, 0, 0, nullptr);
    if (s.inSpawn) return out;
    out.reserve(kNumGoldKinds);
    add(1, cost("City", ec.cityCost, cities), cities, build("City", "interior"));
    add(2, cost("Factory", ec.factoryCost, factories), factories, build("Factory", "interior"));
    if (m.coastal) add(3, cost("Port", ec.portCost, ports), ports, build("Port", "coast-far"));
    // Post toward whoever attacks us, else toward the strongest non-friendly neighbour.
    int postNear = worst ? worst->attacker : 0;
    if (!postNear)
        for (const auto& p : s.players)
            if (p.border > 0 && !p.friendly && (!postNear || p.troops > s.player(postNear)->troops)) postNear = p.small;
    if (postNear) {
        auto a = build("Defense Post", "border");
        a["near"] = postNear;
        add(4, cost("Defense Post", ec.postCost, posts), posts, std::move(a), postNear);
    }
    if (lowCity)
        add(5, cost("City", ec.cityCost, cities), lowCity->level,
            {{"type", "upgrade"}, {"unit", "City"}, {"unitId", lowCity->id}});
    int siloCap = std::clamp(cities / std::max(1, sc.citiesPerSilo), 1, std::max(1, sc.maxSilos));
    if (sc.enabled && silos < siloCap && m.tiles >= sc.minTilesForSilo) {
        auto a = build("Missile Silo", "interior");
        a["mountainWeight"] = sc.mountainWeight;
        add(6, s.cost("Missile Silo", sc.siloCost), silos, std::move(a));
    }
    if (sc.enabled && sams < 1 + cities / std::max(1, sc.citiesPerSam)) {
        auto a = build("SAM Launcher", "interior");
        a["mountainWeight"] = sc.mountainWeight;
        add(7, s.cost("SAM Launcher", sc.samCost), sams, std::move(a));
    }
    if (ports > 0 && warships < ec.maxWarships)
        add(8, cost("Warship", ec.warshipCost, warships), warships, build("Warship", "coast"));
    return out;
}




// ---------------------------------------------------------------- deciding

double Learner::priorWeight(int net) const {
    if (net == kSpawn)  // one decision per game: its rule bonus fades over games, not thousands of samples
        return std::max(lc_.priorFloor, lc_.prior * std::pow(0.5, double(brain_.spawnSeen) / 200.0));
    double w = lc_.prior * std::pow(0.5, double(brain_.seen) / std::max(1.0, lc_.priorHalfLife));
    return std::max(w, std::min(lc_.prior, lc_.priorFloor));
}

// Ensemble mean and spread of every option's score.
void Learner::scoreAll(int net, const Features& ft, const std::vector<Option>& opts, std::vector<double>& mean,
                       std::vector<double>& sd) {
    const size_t n = opts.size();
    mean.assign(n, 0.0);
    sd.assign(n, 0.0);
    const size_t k = net == kSpawn ? brain_.spawn.size() : brain_.members.size();
    for (size_t m = 0; m < k; ++m) {
        if (net != kSpawn) brain_.members[m].encode({ft.self.data(), ft.ents.data(), ft.nEnts});
        for (size_t i = 0; i < n; ++i) {
            double v = net == kSpawn ? brain_.spawn[m].predict(opts[i].x)
                                     : brain_.members[m].advantage(net, opts[i].x.data(), opts[i].target);
            mean[i] += v;
            sd[i] += v * v;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        mean[i] /= double(k);
        sd[i] = std::sqrt(std::max(0.0, sd[i] / double(k) - mean[i] * mean[i]));
    }
}

float Learner::score(int net, const Features& ft, const Option& o) {
    std::vector<double> mean, sd;
    scoreAll(net, ft, {o}, mean, sd);
    return float(mean[0]);
}

const Option& Learner::choose(int net, const Features& ft, std::vector<Option>& opts) {
    const double prior = priorWeight(net);
    std::vector<double> mean, sd, v(opts.size());
    scoreAll(net, ft, opts, mean, sd);
    size_t best = 0;
    for (size_t i = 0; i < opts.size(); ++i) {
        // Members that disagree don't know yet: an unfamiliar move is marked down by their spread,
        // while the rules' move keeps its bonus. So the nets overrule the rules only when they agree.
        v[i] = opts[i].rulePick ? mean[i] + prior : mean[i] - lc_.caution * sd[i];
        if (advice_) v[i] += advice_(net, opts[i]);
        if (v[i] > v[best]) best = i;
    }
    // Exploration samples by score, so clearly bad moves stay rare even while exploring.
    const size_t seen = net == kSpawn ? brain_.spawnSeen * 100 : brain_.seen;
    double eps = lc_.explore ? lc_.epsilon / (1.0 + seen * lc_.epsilonDecay) : 0.0;
    if (opts.size() > 1 && std::uniform_real_distribution<double>(0, 1)(rng_) < eps) {
        const double temp = std::max(1e-3, lc_.temperature);
        std::vector<double> w(opts.size());
        for (size_t i = 0; i < opts.size(); ++i) w[i] = opts[i].reckless ? 0.0 : std::exp((v[i] - v[best]) / temp);
        if (std::any_of(w.begin(), w.end(), [](double x) { return x > 0; }))
            best = std::discrete_distribution<size_t>(w.begin(), w.end())(rng_);
    }
    lastPick_ = mean[best];
    lastSpread_ = sd[best];
    return opts[best];
}

namespace {

// What the hand-written rules would do this tick, without touching the real plan or cooldowns.
nlohmann::json rulePick(Ctx& c, std::initializer_list<void (*)(Ctx&)> modules,
                        std::initializer_list<const char*> types) {
    Memory mem = c.mem;
    Plan plan;
    plan.troopBudget = c.plan.troopBudget;
    plan.goldBudget = c.plan.goldBudget;
    plan.overextended = c.plan.overextended;
    Ctx scratch{c.s, c.cfg, mem, plan};
    for (auto* m : modules) m(scratch);
    for (const auto& a : plan.actions)
        for (const char* t : types)
            if (a.value("type", "") == t) return a;
    return nullptr;
}

std::string fmt(double v) {
    std::ostringstream o;
    o.precision(2);
    o << std::fixed << v;
    return o.str();
}

}  // namespace

void Learner::troopPhase(Ctx& c) {
    const State& s = c.s;
    if (!c.mem.ready("nn:troop", s.tick, lc_.troopCooldown)) return;
    c.mem.mark("nn:troop", s.tick);

    auto opts = troopOptions(s, c.plan, c.plan.overextended);
    auto rule = rulePick(c, {militaryModule, expandModule, navalModule}, {"attack", "boat"});
    std::string ruleLabel = "wait";
    if (rule.is_null()) {
        if (opts[0].action.is_null()) opts[0].rulePick = true;
    } else {
        double r = s.me.troops > 0 ? rule.value("troops", 0.0) / s.me.troops : 0;
        Option* match = nullptr;
        for (auto& o : opts) {
            if (o.action.is_null() || o.action["type"] != rule["type"]) continue;
            if (rule["type"] == "boat" ? o.action["dst"] != rule["dst"] : o.action["target"] != rule["target"]) continue;
            if (!match || std::abs(o.ratio - r) < std::abs(match->ratio - r)) match = &o;
        }
        if (match) {
            match->rulePick = true;
            ruleLabel = match->label;
        }
    }

    const Features ft = features(s, c.plan.overextended);
    const Option& pick = choose(kTroop, ft, opts);
    record(kTroop, ft, pick, s);
    if (pick.action.is_null()) return;
    double troops = c.plan.commit(pick.troops);
    if (troops < 1) return;
    auto a = pick.action;
    a["troops"] = troops;
    c.plan.actions.push_back(std::move(a));
    c.plan.log.push_back("nn: " + pick.label + " (" + fmt(lastPick_) + ", rule: " + ruleLabel + ")");
}

void Learner::goldPhase(Ctx& c) {
    const State& s = c.s;
    if (!c.mem.ready("nn:gold", s.tick, lc_.goldCooldown)) return;
    c.mem.mark("nn:gold", s.tick);

    auto opts = goldOptions(s, c.plan);
    auto rule = rulePick(c, {strategicBuildModule, economyModule}, {"build", "upgrade"});
    std::string want = rule.is_null() ? "save"
                       : rule["type"] == "upgrade" ? "upgrade City"
                                                   : rule.value("unit", "");
    for (auto& o : opts) o.rulePick = o.label == want;

    const Features ft = features(s, c.plan.overextended);
    const Option& pick = choose(kGold, ft, opts);
    record(kGold, ft, pick, s);
    if (pick.action.is_null() || !c.plan.spend(pick.gold)) return;
    c.plan.actions.push_back(pick.action);
    c.plan.log.push_back("nn: " + pick.label + " (" + fmt(lastPick_) + ", rule: " + want + ")");
}



// Before the game begins: score every candidate spawn site the page found. Late in the spawn phase,
// when the others have shown where they start, look again once and move if somewhere else is better.
bool Learner::spawnPhase(Ctx& c) {
    const State& s = c.s;
    if (!s.inSpawn || s.spawnCandidates.empty()) return false;
    const bool reconsider = s.spawned;
    if (reconsider && (s.tick < c.cfg.spawn.reconsiderTick || c.mem.lastTick.count("respawn"))) return true;
    if (!reconsider && !c.mem.ready("spawn", s.tick, 20)) return true;

    std::vector<Option> opts;
    size_t ruleBest = 0;
    for (size_t i = 0; i < s.spawnCandidates.size(); ++i) {
        const auto& cand = s.spawnCandidates[i];
        if (int(cand.f.size()) != kSpawnPageFeatures) continue;
        Option o;
        o.x = cand.f;
        o.x.push_back(float(cand.heur));
        o.x.push_back(clampf(s.tick / 100.0, 0, 2));
        o.label = "spawn";
        o.action = {{"type", "spawn"}, {"tile", cand.tile}};
        o.ratio = cand.heur;
        opts.push_back(std::move(o));
        if (opts.back().ratio > opts[ruleBest].ratio) ruleBest = opts.size() - 1;
    }
    if (opts.empty()) return false;
    opts[ruleBest].rulePick = true;
    const Option& pick = choose(kSpawn, Features{}, opts);
    c.mem.mark(reconsider ? "respawn" : "spawn", s.tick);
    // Moving is only worth it for a clearly better site.
    if (reconsider && pick.ratio < 1.15 * c.mem.spawnHeur && !(lastPick_ > c.mem.spawnScore + 0.25)) return true;
    c.mem.spawnHeur = pick.ratio;
    c.mem.spawnScore = lastPick_;
    record(kSpawn, Features{}, pick, s);
    c.plan.actions.push_back(pick.action);
    c.plan.log.push_back(std::string(reconsider ? "nn: moving spawn" : "nn: spawning") + " (" + fmt(lastPick_) + ", " +
                         std::to_string(opts.size()) + " sites" + (pick.rulePick ? "" : ", not the rule's pick") + ")");
    return true;
}

// ---------------------------------------------------------------- recording

void Learner::snapshot(const State& s, double out[kSnap]) const {
    const Me& m = s.me;
    double structures = 0;
    for (const auto& u : m.units) {
        if (u.type == "City") structures += 125000.0 * u.level;
        else if (u.type == "Factory" || u.type == "Port") structures += 125000;
        else if (u.type == "Defense Post") structures += 50000;
        else if (u.type == "Warship") structures += 250000;
        else if (u.type == "Missile Silo") structures += 1000000;
        else if (u.type == "SAM Launcher") structures += 1500000;
    }
    out[0] = m.alive ? m.tiles : 0;
    out[1] = s.landTiles;
    out[2] = m.alive ? m.troops : 0;
    out[3] = std::max(1.0, m.maxTroops);
    out[4] = m.gold;
    out[5] = m.gold + structures;
    out[6] = m.incomingTroops();
    out[7] = earned_;
}

void Learner::record(int net, const Features& ft, const Option& pick, const State& s) {
    if (!lc_.record) return;
    if (!writer_.isOpen() && !writer_.open((std::filesystem::path(dir_) / "episodes").string(), learnerDims())) return;
    Decision d;
    d.net = net;
    d.tick = s.tick;
    snapshot(s, d.snap);
    d.self = ft.self;
    d.ents = ft.ents;
    d.nEnts = ft.nEnts;
    d.opt = pick.x;
    d.target = pick.target;
    d.rule = pick.rulePick;
    writer_.write(d, pick.label);
}

void Learner::observe(const State& s) {
    if (s.tick < lastTick_) {  // a new game started under the same engine
        writer_.close();
        earned_ = income_ = 0;
        lastGold_ = -1;
    }
    if (s.spawned) {
        // Gold going up is income; going down is spending.
        if (lastGold_ >= 0 && s.me.gold > lastGold_ && s.tick > lastTick_) {
            const double gain = s.me.gold - lastGold_;
            earned_ += gain;
            income_ += 0.02 * (gain / (s.tick - lastTick_) - income_);
        }
        lastGold_ = s.me.gold;
        lastState_ = s;
    }
    if (s.tick / kReloadEveryTicks != lastTick_ / kReloadEveryTicks) reloadIfNewer();
    lastTick_ = s.tick;
}

void Learner::gameOver(bool won, bool eliminated) {
    if (!writer_.isOpen()) return;
    double snap[kSnap];
    snapshot(lastState_, snap);
    if (eliminated) snap[0] = snap[2] = 0;
    writer_.end(lastTick_, snap, won, eliminated);
}

}  // namespace fw
