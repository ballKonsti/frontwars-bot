#include "state.h"

#include <algorithm>
#include <cctype>

namespace fw {

using nlohmann::json;

int Me::count(const std::string& type) const {
    return int(std::count_if(units.begin(), units.end(), [&](const Unit& u) { return u.type == type; }));
}

double Me::incomingTroops() const {
    double sum = 0;
    for (const auto& a : incoming) sum += a.troops;
    return sum;
}

const Attack* Me::attackOn(int target) const {
    for (const auto& a : outgoing)
        if (a.target == target) return &a;
    return nullptr;
}

bool Player::isBot() const {
    std::string t = type;
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return std::tolower(c); });
    return t == "bot";
}

const Player* State::player(int small) const {
    for (const auto& p : players)
        if (p.small == small) return &p;
    return nullptr;
}

const Player* State::leader() const {
    const Player* top = nullptr;
    for (const auto& p : players)
        if (p.tiles > me.tiles && (!top || p.tiles > top->tiles)) top = &p;
    return top;
}

namespace {

template <typename T>
T get(const json& j, const char* key, T fallback) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return fallback;
    return it->get<T>();
}

// Attack targets arrive as small IDs, but tolerate strings and null (neutral).
int smallRef(const json& j, const char* key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return 0;
    if (it->is_number()) return it->get<int>();
    if (it->is_string()) {
        try {
            return std::stoi(it->get<std::string>());
        } catch (...) {
            return 0;
        }
    }
    return 0;
}

}  // namespace

State parseState(const json& j) {
    State s;
    s.ready = get(j, "ready", false);
    s.spawned = get(j, "spawned", false);
    s.inSpawn = get(j, "inSpawn", false);
    s.tick = get(j, "tick", 0);
    s.mapW = get(j, "mapW", 0);
    s.mapH = get(j, "mapH", 0);
    s.landTiles = std::max(1, get(j, "landTiles", 1));
    s.neutralBorder = get(j, "neutralBorder", 0);
    s.nukesInbound = get(j, "nukesInbound", 0);

    if (auto it = j.find("me"); it != j.end() && it->is_object()) {
        const json& m = *it;
        Me& me = s.me;
        me.small = get(m, "small", 0);
        me.id = get<std::string>(m, "id", "");
        me.tiles = get(m, "tiles", 0);
        me.troops = get(m, "troops", 0.0);
        me.maxTroops = get(m, "maxTroops", 0.0);
        me.troopRate = get(m, "troopRate", 0.0);
        me.gold = get(m, "gold", 0.0);
        me.isTraitor = get(m, "isTraitor", false);
        me.alive = get(m, "alive", true);
        me.coastal = get(m, "coastal", false);
        me.borderLen = get(m, "borderLen", 0);
        for (const auto& a : m.value("outgoing", json::array()))
            me.outgoing.push_back({get<std::string>(a, "id", ""), smallRef(a, "target"), get(a, "troops", 0.0)});
        for (const auto& a : m.value("incoming", json::array()))
            me.incoming.push_back({get<std::string>(a, "id", ""), smallRef(a, "attacker"), get(a, "troops", 0.0)});
        for (const auto& u : m.value("units", json::array()))
            me.units.push_back({get(u, "id", 0), get<std::string>(u, "type", ""), get(u, "tile", 0), get(u, "level", 1)});
    }

    for (const auto& p : j.value("players", json::array())) {
        Player q;
        q.small = get(p, "small", 0);
        q.id = get<std::string>(p, "id", "");
        q.name = get<std::string>(p, "name", "");
        q.type = get<std::string>(p, "type", "");
        q.tiles = get(p, "tiles", 0);
        q.troops = get(p, "troops", 0.0);
        q.gold = get(p, "gold", 0.0);
        q.isTraitor = get(p, "isTraitor", false);
        q.allied = get(p, "allied", false);
        q.friendly = get(p, "friendly", false);
        q.requestingAlliance = get(p, "requestingAlliance", false);
        q.embargoed = get(p, "embargoed", false);
        q.border = get(p, "border", 0);
        q.underAttack = get(p, "underAttack", 0);
        s.players.push_back(std::move(q));
    }

    for (const auto& u : j.value("enemyStructures", json::array()))
        s.enemyStructures.push_back(
            {get(u, "owner", 0), get<std::string>(u, "type", ""), get(u, "tile", 0), get(u, "level", 1)});

    if (auto it = j.find("costs"); it != j.end() && it->is_object())
        for (const auto& [unit, v] : it->items())
            if (v.is_number()) s.costs[unit] = v.get<double>();

    for (const auto& n : j.value("nukeSources", json::array()))
        if (n.is_number()) s.nukeSources.push_back(n.get<int>());

    for (const auto& b : j.value("boatTargets", json::array()))
        s.boatTargets.push_back({get(b, "tile", 0), get(b, "owner", 0), get(b, "dist", 0)});

    for (const auto& c : j.value("spawnCandidates", json::array())) {
        SpawnCandidate sc{get(c, "tile", 0), get(c, "h", 0.0), {}};
        for (const auto& v : c.value("f", json::array()))
            if (v.is_number()) sc.f.push_back(v.get<float>());
        s.spawnCandidates.push_back(std::move(sc));
    }

    return s;
}

}  // namespace fw
