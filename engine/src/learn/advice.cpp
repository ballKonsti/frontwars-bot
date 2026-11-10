#include "learn/advice.h"

#include <algorithm>
#include <fstream>
#include <set>

namespace fw {

namespace {

const std::set<std::string> kWhenKeys = {"phase", "troopFillMin", "troopFillMax", "landShareMin", "landShareMax",
                                         "neutralLand", "underAttack", "citiesMin", "citiesMax", "goldMin"};
const std::set<std::string> kPreferKeys = {"decision", "kind", "ratioMin", "ratioMax", "target"};
const std::set<std::string> kPhases = {"", "early", "mid", "late"};
const std::set<std::string> kTroopKinds = {"wait", "expand", "attack", "boat"};
const std::set<std::string> kGoldKinds = {"save", "City", "Factory", "Port", "Defense Post", "upgrade City",
                                          "Missile Silo", "SAM Launcher", "Warship"};
const std::set<std::string> kTargets = {"", "bot", "nation", "weaker", "stronger", "attacker"};

int triState(const nlohmann::json& j, const char* key) {
    auto it = j.find(key);
    return it == j.end() || !it->is_boolean() ? -1 : it->get<bool>() ? 1 : 0;
}

}  // namespace

Advice Advice::parse(const nlohmann::json& j, std::vector<std::string>* errors) {
    Advice a;
    auto fail = [&](const std::string& id, const std::string& why) {
        if (errors) errors->push_back((id.empty() ? "item" : id) + ": " + why);
    };
    if (!j.is_object() || !j.contains("items") || !j["items"].is_array()) return a;
    for (const auto& it : j["items"]) {
        if (!it.is_object()) continue;
        AdviceItem x;
        x.id = it.value("id", "");
        x.text = it.value("text", "");
        if (!it.value("enabled", true)) continue;
        try {
            x.strength = std::clamp(it.value("strength", 0.5), -1.0, 1.0);
            const auto when = it.value("when", nlohmann::json::object());
            const auto prefer = it.value("prefer", nlohmann::json::object());
            std::string bad;
            for (const auto& [k, v] : when.items())
                if (!kWhenKeys.count(k)) bad = "unknown condition \"" + k + "\"";
            for (const auto& [k, v] : prefer.items())
                if (!kPreferKeys.count(k)) bad = "unknown preference \"" + k + "\"";
            x.phase = when.value("phase", "");
            x.troopFillMin = when.value("troopFillMin", 0.0);
            x.troopFillMax = when.value("troopFillMax", 9.0);
            x.landShareMin = when.value("landShareMin", 0.0);
            x.landShareMax = when.value("landShareMax", 9.0);
            x.goldMin = when.value("goldMin", 0.0);
            x.neutralLand = triState(when, "neutralLand");
            x.underAttack = triState(when, "underAttack");
            x.citiesMin = when.value("citiesMin", 0);
            x.citiesMax = when.value("citiesMax", 1 << 30);
            const std::string decision = prefer.value("decision", "");
            x.kind = prefer.value("kind", "");
            x.target = prefer.value("target", "");
            x.ratioMin = prefer.value("ratioMin", 0.0);
            x.ratioMax = prefer.value("ratioMax", 9.0);
            x.net = decision == "gold" ? 1 : 0;
            if (decision != "troop" && decision != "gold") bad = "decision must be \"troop\" or \"gold\"";
            else if (!kPhases.count(x.phase)) bad = "unknown phase \"" + x.phase + "\"";
            else if (!(x.net ? kGoldKinds : kTroopKinds).count(x.kind)) bad = "unknown kind \"" + x.kind + "\"";
            else if (!kTargets.count(x.target)) bad = "unknown target \"" + x.target + "\"";
            if (!bad.empty()) {
                fail(x.id, bad);
                continue;
            }
        } catch (const std::exception& e) {
            fail(x.id, e.what());
            continue;
        }
        a.items_.push_back(std::move(x));
    }
    return a;
}

Advice Advice::load(const std::string& path, std::vector<std::string>* errors) {
    std::ifstream in(path);
    if (!in) return {};
    auto j = nlohmann::json::parse(in, nullptr, false);
    if (j.is_discarded()) {
        if (errors) errors->push_back("advice file is not valid JSON");
        return {};
    }
    return parse(j, errors);
}

double Advice::bonus(const State& s, int net, const AdviceOption& o, double scale) const {
    if (items_.empty()) return 0;
    const Me& m = s.me;
    const double fill = m.troops / std::max(1.0, m.maxTroops);
    const double share = s.landShare();
    const char* phase = s.tick < 3000 ? "early" : s.tick < 9000 ? "mid" : "late";
    const int cities = m.count("City");
    const bool attacked = !m.incoming.empty();
    double sum = 0;
    for (const auto& x : items_) {
        if (x.net != net || x.kind != o.kind) continue;
        if (!x.phase.empty() && x.phase != phase) continue;
        if (fill < x.troopFillMin || fill > x.troopFillMax || share < x.landShareMin || share > x.landShareMax) continue;
        if (m.gold < x.goldMin || cities < x.citiesMin || cities > x.citiesMax) continue;
        if (x.neutralLand >= 0 && (s.neutralBorder > 0) != (x.neutralLand == 1)) continue;
        if (x.underAttack >= 0 && attacked != (x.underAttack == 1)) continue;
        if (o.ratio < x.ratioMin - 1e-9 || o.ratio > x.ratioMax + 1e-9) continue;
        if (!x.target.empty()) {
            const Player* p = o.target;
            if (!p) continue;
            const bool attacker = std::any_of(m.incoming.begin(), m.incoming.end(),
                                              [&](const Incoming& a) { return a.attacker == p->small; });
            if (x.target == "bot" && !p->isBot()) continue;
            if (x.target == "nation" && p->isBot()) continue;
            if (x.target == "weaker" && p->troops >= m.troops) continue;
            if (x.target == "stronger" && p->troops < m.troops) continue;
            if (x.target == "attacker" && !attacker) continue;
        }
        sum += x.strength * scale;
    }
    return sum;
}

}  // namespace fw
