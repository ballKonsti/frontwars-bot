#include "learn/learner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include <unistd.h>

namespace fw {

namespace {

constexpr int kFeatureVersion = 3;  // bump when features change: older episodes and weights are skipped
constexpr int kTroopOptionFeatures = 18;
constexpr int kGoldOptionFeatures = 12;
constexpr int kSpawnPageFeatures = 9;   // computed by the page per candidate
constexpr int kSpawnFeatures = kSpawnPageFeatures + 2;
constexpr double kRatios[] = {0.10, 0.20, 0.35, 0.50, 0.75};
constexpr double kBoatRatio = 0.30;
constexpr int kMaxAttackTargets = 10;
constexpr int kMaxBoatTargets = 3;
constexpr float kHuber = 2.0f;
constexpr size_t kMinSamples[] = {64, 64, 12};  // per net, before it is trained at all

const char* kGoldKinds[] = {"save",         "City",         "Factory",      "Port",   "Defense Post",
                            "upgrade City", "Missile Silo", "SAM Launcher", "Warship"};
constexpr int kNumGoldKinds = sizeof kGoldKinds / sizeof *kGoldKinds;
const char* kNetNames[] = {"troop", "gold", "spawn"};

float clampf(double v, double lo, double hi) { return float(std::min(hi, std::max(lo, v))); }
float logRatio(double a, double b) { return clampf(std::log((a + 1.0) / (b + 1.0)) / 3.0, -1.0, 1.0); }

std::vector<float> stateFeatures(const State& s, bool overextended) {
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
    return f;
}

double estimatedCost(double base, int owned, const EconomyConfig& cfg) { return base * (1.0 + cfg.costGrowth * owned); }

// {tiles, landTiles, troops, maxTroops, gold, wealth}; wealth = gold + what our structures cost.
void snapshot(const State& s, double out[6]) {
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
}


int stateFeatureCount() {
    static const int n = int(stateFeatures(State{}, false).size());
    return n;
}

}  // namespace

Learner::Learner(const Config& cfg, std::string modelDir, uint32_t seed)
    : cfg_(cfg), lc_(cfg.learning), dir_(std::move(modelDir)), rng_(seed) {
    const int h = std::max(8, lc_.hidden);
    const int k = std::max(1, lc_.ensemble);
    const int in[kNets] = {stateFeatureCount() + kTroopOptionFeatures, stateFeatureCount() + kGoldOptionFeatures,
                           kSpawnFeatures};
    for (int net = 0; net < kNets; ++net) {
        // The spawn net sees one decision per game, so it stays small.
        std::vector<int> sizes = net == kSpawn ? std::vector<int>{in[net], 24, 12, 1}
                                               : std::vector<int>{in[net], h, h, h / 2, 1};
        for (int m = 0; m < k; ++m) nets_[net].members.emplace_back(sizes, seed + 101 * net + 7 * m);
        nets_[net].act = nets_[net].members;
    }
}

Learner::~Learner() {
    stopStudy();
    closeEpisode();
}

// ---------------------------------------------------------------- options

std::vector<Option> Learner::troopOptions(const State& s, const Plan& plan, bool overextended) const {
    const auto base = stateFeatures(s, overextended);
    const Me& m = s.me;
    const double myDensity = m.troops / std::max(1, m.tiles);
    const Player* leader = s.leader();

    auto make = [&](const char* kind, double ratio, const Player* p, const BoatTarget* b) {
        Option o;
        o.x = base;
        o.x.reserve(base.size() + kTroopOptionFeatures);
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
    const auto base = stateFeatures(s, plan.overextended);
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
    auto add = [&](int kind, double cost, int owned, nlohmann::json action) {
        if (kind != 0 && cost > plan.goldBudget) return;
        Option o;
        o.x = base;
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
        add(4, cost("Defense Post", ec.postCost, posts), posts, std::move(a));
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
    double w = lc_.prior * std::pow(0.5, double(nets_[net].seen) / std::max(1.0, lc_.priorHalfLife));
    return std::max(w, std::min(lc_.prior, lc_.priorFloor));
}

// Ensemble mean and spread for one input. Callers hold actMu_.
void Learner::predict(int net, const std::vector<float>& x, double& mean, double& sd) const {
    const auto& act = nets_[net].act;
    double sum = 0, sq = 0;
    for (const auto& m : act) {
        double v = m.predict(x);
        sum += v;
        sq += v * v;
    }
    mean = sum / double(act.size());
    sd = std::sqrt(std::max(0.0, sq / double(act.size()) - mean * mean));
}

float Learner::score(int net, const Option& o) const {
    std::lock_guard<std::mutex> lock(actMu_);
    double mean, sd;
    predict(net, o.x, mean, sd);
    return float(mean);
}

const Option& Learner::choose(int net, std::vector<Option>& opts) {
    const Net& n = nets_[net];
    // The spawn net sees one decision per game, so its rule bonus fades over games, not thousands of samples.
    const double prior = net == kSpawn ? std::max(lc_.priorFloor, lc_.prior * std::pow(0.5, double(n.seen) / 200.0))
                                       : priorWeight(net);
    std::vector<double> v(opts.size()), mean(opts.size());
    size_t best = 0;
    {
        std::lock_guard<std::mutex> lock(actMu_);
        for (size_t i = 0; i < opts.size(); ++i) {
            double sd;
            predict(net, opts[i].x, mean[i], sd);
            // Members that disagree don't know yet: an unfamiliar move is marked down by their spread,
            // while the rules' move keeps its bonus. So the nets overrule the rules only when they agree.
            v[i] = opts[i].rulePick ? mean[i] + prior : mean[i] - lc_.caution * sd;
            if (v[i] > v[best]) best = i;
        }
    }
    lastBest_[net] = mean[best];
    // Exploration samples by score, so clearly bad moves stay rare even while exploring.
    double eps = lc_.explore ? lc_.epsilon / (1.0 + n.seen * lc_.epsilonDecay) : 0.0;
    if (opts.size() > 1 && std::uniform_real_distribution<double>(0, 1)(rng_) < eps) {
        const double temp = std::max(1e-3, lc_.temperature);
        std::vector<double> w(opts.size());
        for (size_t i = 0; i < opts.size(); ++i) w[i] = std::exp((v[i] - v[best]) / temp);
        best = std::discrete_distribution<size_t>(w.begin(), w.end())(rng_);
    }
    lastPick_ = mean[best];
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

    const Option& pick = choose(kTroop, opts);
    record(kTroop, pick, s);
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

    const Option& pick = choose(kGold, opts);
    record(kGold, pick, s);
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
    const Option& pick = choose(kSpawn, opts);
    c.mem.mark(reconsider ? "respawn" : "spawn", s.tick);
    // Moving is only worth it for a clearly better site.
    if (reconsider && pick.ratio < 1.15 * c.mem.spawnHeur && !(lastPick_ > c.mem.spawnScore + 0.25)) return true;
    c.mem.spawnHeur = pick.ratio;
    c.mem.spawnScore = lastPick_;
    record(kSpawn, pick, s);
    c.plan.actions.push_back(pick.action);
    c.plan.log.push_back(std::string(reconsider ? "nn: moving spawn" : "nn: spawning") + " (" + fmt(lastPick_) + ", " +
                         std::to_string(opts.size()) + " sites" + (pick.rulePick ? "" : ", not the rule's pick") + ")");
    return true;
}

// ---------------------------------------------------------------- episodes
//
// One JSON object per line:
//   {"v":3,"dims":[troopInputs,goldInputs,spawnInputs]}                  header
//   {"n":net,"t":tick,"s":[snapshot],"x":[inputs],"a":"label","r":0|1}   a decision (r: it was the rule's pick)
//   {"end":true,"t":tick,"s":[snapshot],"won":bool,"eliminated":bool}    outcome
// A file without an end line is a game that was cut short (or is still being played).

namespace {

void writeSnap(std::FILE* f, const double snap[6]) {
    std::fprintf(f, "\"s\":[%.0f,%.0f,%.0f,%.0f,%.0f,%.0f]", snap[0], snap[1], snap[2], snap[3], snap[4], snap[5]);
}

}  // namespace

void Learner::record(int net, const Option& pick, const State& s) {
    if (!lc_.record) return;
    if (!episode_) {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::path dir = fs::path(dir_) / "episodes";
        fs::create_directories(dir, ec);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch()).count();
        static int serial = 0;  // several games can start within one millisecond in tests
        fs::path file = dir / ("ep-" + std::to_string(ms) + "-" + std::to_string(::getpid()) + "-" +
                               std::to_string(serial++) + ".jsonl");
        episode_ = std::fopen(file.c_str(), "w");
        if (!episode_) return;
        std::fprintf(episode_, "{\"v\":%d,\"dims\":[%d,%d,%d]}\n", kFeatureVersion, inputs(kTroop), inputs(kGold),
                     inputs(kSpawn));
    }
    double snap[6];
    snapshot(s, snap);
    std::fprintf(episode_, "{\"n\":%d,\"t\":%d,", net, s.tick);
    writeSnap(episode_, snap);
    std::fputs(",\"x\":[", episode_);
    for (size_t i = 0; i < pick.x.size(); ++i) std::fprintf(episode_, i ? ",%.4g" : "%.4g", double(pick.x[i]));
    std::fprintf(episode_, "],\"a\":%s,\"r\":%d}\n", nlohmann::json(pick.label).dump().c_str(), pick.rulePick ? 1 : 0);
    std::fflush(episode_);

    if (studying_ && net != kSpawn) studyLive(net, pick.x, s.tick, potential(snap));
}

void Learner::closeEpisode() {
    if (episode_) std::fclose(episode_);
    episode_ = nullptr;
    trail_.clear();
}

void Learner::observe(const State& s) {
    if (s.tick < lastTick_) closeEpisode();  // a new game started under the same engine
    lastTick_ = s.tick;
    if (s.spawned) lastState_ = s;
}

void Learner::gameOver(bool won, bool eliminated) {
    if (!episode_) return;
    double snap[6];
    snapshot(lastState_, snap);
    if (eliminated) snap[0] = snap[2] = 0;
    std::fprintf(episode_, "{\"end\":true,\"t\":%d,", lastTick_);
    writeSnap(episode_, snap);
    std::fprintf(episode_, ",\"won\":%s,\"eliminated\":%s}\n", won ? "true" : "false", eliminated ? "true" : "false");
    closeEpisode();
}

// ---------------------------------------------------------------- learning

double Learner::potential(const double s[6]) const {
    return lc_.wLand * std::sqrt(std::max(0.0, s[0]) / std::max(1.0, s[1])) + lc_.wGold * std::log1p(std::max(0.0, s[5]) / 1e5) +
           lc_.wTroops * std::clamp(s[2] / std::max(1.0, s[3]), 0.0, 1.5) + lc_.wCap * std::log1p(std::max(0.0, s[3]) / 1e5);
}

// Reads every episode, newest first, until `replay` decisions per net are collected. Each decision's
// target is the discounted sum of later changes in position value, plus the discounted outcome.
// With holdOut, every 8th episode goes to `val` once there are enough of them.
int Learner::loadEpisodes(std::vector<Sample> train[kNets], std::vector<Sample> val[kNets], bool holdOut) {
    namespace fs = std::filesystem;
    using nlohmann::json;
    std::error_code ec;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(fs::path(dir_) / "episodes", ec))
        if (e.path().extension() == ".jsonl") files.push_back(e.path());
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename().string() > b.filename().string();  // names start with a millisecond timestamp
    });
    const size_t keep = size_t(std::max(1, lc_.maxEpisodes));
    for (size_t i = keep; i < files.size(); ++i) fs::remove(files[i], ec);
    if (files.size() > keep) files.resize(keep);

    struct Step {
        int net, tick;
        double phi;
        std::vector<float> x;
    };
    const double gammaTick = std::exp(-1.0 / std::max(1, lc_.horizon));
    const size_t cap = size_t(std::max(1, lc_.replay));
    holdOut = holdOut && files.size() >= 8;
    const json dims = json::array({inputs(kTroop), inputs(kGold), inputs(kSpawn)});
    int used = 0;
    for (size_t fi = 0; fi < files.size(); ++fi) {
        std::ifstream in(files[fi]);
        std::string line;
        std::vector<Step> steps;
        bool ok = false, ended = false, won = false, eliminated = false;
        double endPhi = 0;
        int endTick = 0;
        while (std::getline(in, line)) {
            json j = json::parse(line, nullptr, false);
            if (j.is_discarded() || !j.is_object()) break;  // half-written last line of a running game
            if (j.contains("v")) {
                ok = j.value("v", 0) == kFeatureVersion && j.value("dims", json::array()) == dims;
                if (!ok) break;
                continue;
            }
            if (!ok || !j.contains("s") || j["s"].size() != 6) break;
            double snap[6];
            for (int k = 0; k < 6; ++k) snap[k] = j["s"][k].get<double>();
            if (j.value("end", false)) {
                ended = true;
                won = j.value("won", false);
                eliminated = j.value("eliminated", false);
                endPhi = potential(snap);
                endTick = j.value("t", 0);
                break;
            }
            Step st{j.value("n", 0), j.value("t", 0), potential(snap), j["x"].get<std::vector<float>>()};
            if (st.net < 0 || st.net >= kNets || int(st.x.size()) != inputs(st.net)) break;
            steps.push_back(std::move(st));
        }
        if (!ok || steps.size() < 2) continue;
        ++used;

        // Backwards: G_i = (phi_{i+1} - phi_i) + gamma^dt * G_{i+1}.
        double nextPhi, G;
        int nextTick;
        if (ended) {
            nextPhi = endPhi;
            nextTick = std::max(endTick, steps.back().tick);
            G = won ? lc_.terminalWin : eliminated ? lc_.terminalLoss : 0.0;
        } else {
            nextPhi = steps.back().phi;
            nextTick = steps.back().tick;
            G = 0;
        }
        const int lastTick = steps.back().tick;
        const bool isVal = holdOut && fi % 8 == 7;
        for (size_t i = steps.size(); i-- > 0;) {
            Step& st = steps[i];
            G = (nextPhi - st.phi) + std::pow(gammaTick, std::max(0, nextTick - st.tick)) * G;
            nextPhi = st.phi;
            nextTick = st.tick;
            // A cut-short game says little about its last decisions: their future was never seen.
            if (!ended && lastTick - st.tick < lc_.horizon / 2) continue;
            auto& dst = isVal ? val[st.net] : train[st.net];
            if (train[st.net].size() + val[st.net].size() < cap) dst.push_back({std::move(st.x), float(G)});
        }
        if (train[0].size() + val[0].size() >= cap && train[1].size() + val[1].size() >= cap) break;
    }
    return used;
}

void Learner::addSample(int net, std::vector<float> x, float y) {
    nets_[net].data.push_back({std::move(x), y});
    ++nets_[net].seen;
}

void Learner::publish(int net) {
    std::lock_guard<std::mutex> lock(actMu_);
    nets_[net].act = nets_[net].members;
}

// Minibatch steps for one ensemble member. Each member trains on its own 80% of the data (and started
// from its own random weights), which is what makes their disagreement a measure of ignorance.
void Learner::trainMember(int net, int member, int steps, float lrStart, float lrEnd, uint32_t seed) {
    Net& n = nets_[net];
    const size_t size = n.data.size();
    const size_t batch = std::min<size_t>(std::max(1, lc_.batch), size);
    if (batch == 0) return;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, size - 1);
    const bool subset = size >= 200 && n.members.size() > 1;
    std::vector<const std::vector<float>*> xs(batch);
    std::vector<float> ys(batch);
    for (int k = 0; k < steps; ++k) {
        for (size_t i = 0; i < batch; ++i) {
            size_t idx = pick(rng);
            while (subset && (idx * 2654435761u + uint32_t(member) * 40503u) % 5 == 0) idx = pick(rng);
            xs[i] = &n.data[idx].x;
            ys[i] = n.data[idx].y;
        }
        float lr = lrEnd + (lrStart - lrEnd) * float(steps - k) / float(steps);
        n.members[member].train(xs, ys, lr, kHuber);
    }
}

void Learner::trainSteps(int net, int steps) {
    Net& n = nets_[net];
    if (n.data.empty()) return;
    // Step size decays over the run so the fit settles instead of chasing the last minibatches.
    std::vector<std::thread> threads;
    for (int m = 0; m < int(n.members.size()); ++m)
        threads.emplace_back(&Learner::trainMember, this, net, m, steps, float(lc_.lr), 0.1f * float(lc_.lr),
                             uint32_t(rng_()));
    for (auto& t : threads) t.join();
    publish(net);
}

namespace {

// Targets are standardised, so scores are "standard deviations better than an average decision"
// and the prior, caution and temperature mean the same thing however the rewards are weighted.
void meanSd(const std::vector<float>& ys, double& mean, double& sd) {
    mean = 0;
    for (float y : ys) mean += y;
    mean /= double(std::max<size_t>(1, ys.size()));
    double var = 0;
    for (float y : ys) var += (y - mean) * (y - mean);
    sd = std::sqrt(var / double(std::max<size_t>(1, ys.size()))) + 1e-6;
}

}  // namespace

TrainReport Learner::train(int steps) {
    stopStudy();
    TrainReport rep;
    std::vector<Sample> tr[kNets], val[kNets];
    rep.episodes = loadEpisodes(tr, val, true);
    for (int net = 0; net < kNets; ++net) {
        Net& n = nets_[net];
        rep.samples[net] = tr[net].size() + val[net].size();
        if (tr[net].size() < kMinSamples[net]) continue;
        std::vector<float> ys;
        for (const auto& s : tr[net]) ys.push_back(s.y);
        meanSd(ys, n.mean, n.sd);
        for (auto* set : {&tr[net], &val[net]})
            for (auto& s : *set) s.y = float((s.y - n.mean) / n.sd);
        auto valLoss = [&]() {
            if (val[net].size() < 32) return -1.0;
            double se = 0;
            for (const auto& s : val[net]) {
                double p = 0;
                for (const auto& m : n.members) p += m.predict(s.x);
                double e = p / double(n.members.size()) - s.y;
                se += e * e;
            }
            return se / double(val[net].size());
        };
        rep.valBefore[net] = valLoss();
        n.data = std::move(tr[net]);
        // Few samples (the spawn net early on): few steps, or it would just memorise them.
        trainSteps(net, int(std::min<size_t>(size_t(steps), 4 * n.data.size())));
        rep.valAfter[net] = valLoss();
        n.seen = rep.samples[net];
        n.data.clear();
        n.data.shrink_to_fit();
    }
    return rep;
}

// ---------------------------------------------------------------- studying during the game
//
// A background thread keeps training while we play: first on every recorded game, then also on this
// game's own decisions as soon as their short-term result is known. The acting copies of the nets are
// refreshed a few times a second, so what happened a minute ago already shapes the next decision.

void Learner::startStudy() {
    if (studying_ || !lc_.study || !lc_.record) return;
    stop_ = false;
    studying_ = true;
    study_ = std::thread(&Learner::studyLoop, this);
}

void Learner::stopStudy() {
    if (!studying_) return;
    stop_ = true;
    study_.join();
    studying_ = false;
    for (int net = 0; net < kNets; ++net) {
        publish(net);
        nets_[net].data.clear();
    }
}

// Main thread: a decision was just made. Decisions older than studyHorizon now have a known short-term
// result; the rest of their future is estimated by the nets' own value of the current position.
void Learner::studyLive(int net, const std::vector<float>& x, int tick, double phi) {
    trail_.push_back({net, tick, phi, x, false});
    const double gammaTick = std::exp(-1.0 / std::max(1, lc_.horizon));
    const Trail& now = trail_.back();
    for (auto& t : trail_) {
        if (t.done || now.tick - t.tick < lc_.studyHorizon) continue;
        t.done = true;
        // Same recursion as loadEpisodes, cut off at `now`: G = partial + discount * G_now.
        double partial = 0, discount = 1;
        const Trail* prev = &t;
        for (const auto& u : trail_) {
            if (u.tick <= t.tick || &u == &t) continue;
            partial += discount * (u.phi - prev->phi);
            discount *= std::pow(gammaTick, u.tick - prev->tick);
            prev = &u;
        }
        std::lock_guard<std::mutex> lock(liveMu_);
        live_.push_back({t.net, std::move(t.x), partial, discount, lastBest_[t.net]});
    }
    while (!trail_.empty() && trail_.front().done) trail_.pop_front();
}

void Learner::studyLoop() {
    std::vector<Sample> tr[kNets], none[kNets];
    loadEpisodes(tr, none, false);
    bool fitted[kNets] = {false, false, false};
    for (int net = 0; net < kNets; ++net) {
        Net& n = nets_[net];
        if (tr[net].size() < kMinSamples[net]) continue;
        std::vector<float> ys;
        for (const auto& s : tr[net]) ys.push_back(s.y);
        meanSd(ys, n.mean, n.sd);
        for (auto& s : tr[net]) s.y = float((s.y - n.mean) / n.sd);
        n.data = std::move(tr[net]);
        fitted[net] = true;
    }
    std::mt19937 rng(std::random_device{}());
    const float lr = 0.3f * float(lc_.lr);
    for (long it = 1; !stop_; ++it) {
        {
            std::lock_guard<std::mutex> lock(liveMu_);
            for (auto& l : live_) {
                Net& n = nets_[l.net];
                if (!fitted[l.net]) continue;  // no scale for its targets yet
                // G = partial + discount * G_now, with G_now = value * sd + mean; then standardise G.
                double y = (l.partial + l.discount * (l.value * n.sd + n.mean) - n.mean) / n.sd;
                n.data.push_back({std::move(l.x), clampf(y, -6, 6)});
            }
            live_.clear();
        }
        bool any = false;
        for (int net : {kTroop, kGold}) {
            if (!fitted[net]) continue;
            any = true;
            for (int m = 0; m < int(nets_[net].members.size()); ++m) trainMember(net, m, 1, lr, lr, uint32_t(rng()));
        }
        if (it % 40 == 0)
            for (int net : {kTroop, kGold})
                if (fitted[net]) publish(net);
        // Leave the CPU to the game itself: a short nap between steps.
        std::this_thread::sleep_for(std::chrono::milliseconds(any ? 4 : 200));
    }
}

// ---------------------------------------------------------------- persistence

void Learner::save() const {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir_, ec);
    const std::string suffix = ".tmp" + std::to_string(::getpid());
    nlohmann::json stats{{"v", kFeatureVersion}, {"ensemble", nets_[0].members.size()}};
    for (int net = 0; net < kNets; ++net) {
        const Net& n = nets_[net];
        for (size_t m = 0; m < n.members.size(); ++m) {
            const fs::path nn = fs::path(dir_) / (std::string(kNetNames[net]) + "." + std::to_string(m) + ".nn");
            const fs::path tmp = nn.string() + suffix;
            if (n.members[m].save(tmp.string())) fs::rename(tmp, nn, ec);
        }
        stats[kNetNames[net]] = {{"samples", n.seen}, {"mean", n.mean}, {"sd", n.sd}};
    }
    const fs::path file = fs::path(dir_) / "stats.json";
    const fs::path tmp = file.string() + suffix;
    {
        std::ofstream out(tmp);
        out << stats.dump() << '\n';
    }
    fs::rename(tmp, file, ec);
}

void Learner::load() {
    namespace fs = std::filesystem;
    std::ifstream in(fs::path(dir_) / "stats.json");
    auto j = in ? nlohmann::json::parse(in, nullptr, false) : nlohmann::json();
    if (!j.is_object() || j.value("v", 0) != kFeatureVersion) return;  // weights from other features: start fresh
    for (int net = 0; net < kNets; ++net) {
        Net& n = nets_[net];
        bool all = true;
        for (size_t m = 0; m < n.members.size(); ++m)
            all = n.members[m].load((fs::path(dir_) / (std::string(kNetNames[net]) + "." + std::to_string(m) + ".nn")).string()) && all;
        if (!all || !j.contains(kNetNames[net])) continue;
        const auto& st = j[kNetNames[net]];
        n.seen = st.value("samples", size_t(0));
        n.mean = st.value("mean", 0.0);
        n.sd = st.value("sd", 1.0);
        publish(net);
    }
}

}  // namespace fw
