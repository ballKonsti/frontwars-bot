#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace fw {

// Mirrors the JSON produced by bot/src/page-agent.js state().
// Troops are internal units (HUD shows troops / 10). Player references are small IDs; 0 is neutral land.

struct Attack {
    std::string id;
    int target = 0;
    double troops = 0;
};

struct Incoming {
    std::string id;
    int attacker = 0;
    double troops = 0;
};

struct Unit {
    int id = 0;
    std::string type;
    int tile = 0;
    int level = 1;
};

struct Me {
    int small = 0;
    std::string id;
    int tiles = 0;
    double troops = 0;
    double maxTroops = 0;
    double troopRate = 0;
    double gold = 0;
    bool isTraitor = false;
    bool alive = true;
    bool coastal = false;
    int borderLen = 0;
    std::vector<Attack> outgoing;
    std::vector<Incoming> incoming;
    std::vector<Unit> units;

    int count(const std::string& type) const;
    double incomingTroops() const;
    const Attack* attackOn(int target) const;
};

struct Player {
    int small = 0;
    std::string id;
    std::string name;
    std::string type;
    int tiles = 0;
    double troops = 0;
    double gold = 0;
    bool isTraitor = false;
    bool allied = false;
    bool friendly = false;
    bool requestingAlliance = false;
    bool embargoed = false;  // we already embargo them
    int border = 0;
    int underAttack = 0;     // attacks currently hitting them (from anyone)

    bool isBot() const;
};

struct Structure {
    int owner = 0;
    std::string type;
    int tile = 0;
    int level = 1;
};

struct BoatTarget {
    int tile = 0;
    int owner = 0;
    int dist = 0;
};

struct SpawnCandidate {
    int tile = 0;
    double heur = 0;        // the page's rule-of-thumb score, 1 = best candidate
    std::vector<float> f;   // free/owned land around it at three radii, edge distance, position
};

struct State {
    bool ready = false;
    bool spawned = false;
    bool inSpawn = false;
    int tick = 0;
    int mapW = 0;
    int mapH = 0;
    int landTiles = 1;
    Me me;
    int neutralBorder = 0;
    std::vector<Player> players;
    std::vector<Structure> enemyStructures;
    int nukesInbound = 0;
    std::vector<int> nukeSources;  // owners of nukes aimed at our land
    std::vector<BoatTarget> boatTargets;
    std::vector<SpawnCandidate> spawnCandidates;  // only during the spawn phase
    std::unordered_map<std::string, double> costs;  // real current build costs, when the page sent them

    const Player* player(int small) const;
    double cost(const std::string& unit, double estimate) const {
        auto it = costs.find(unit);
        return it != costs.end() && it->second > 0 ? it->second : estimate;
    }
    // Player holding the most land, if they out-hold us; nullptr when we lead.
    const Player* leader() const;
    double landShare() const { return landTiles > 0 ? double(me.tiles) / landTiles : 0; }
};

State parseState(const nlohmann::json& j);

}  // namespace fw
