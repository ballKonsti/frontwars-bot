#pragma once

#include <string>

namespace fw {

// All tuning knobs. Missing keys in config.json keep these defaults.
// Fractions are of current troops unless stated otherwise; cooldowns are game ticks (10 ticks ~ 1 s).

struct SpawnConfig {
    double edgeWeight = 1.0;  // prefer corners/edges: fewer exposed fronts
    int reconsiderTick = 60;  // late in the spawn phase, look again now that others have picked (learner only)
};

struct SustainConfig {
    // The "interest bar": if troops fell by more than maxDrop over the window while we were attacking,
    // stop opening new offensives until they recover.
    int window = 60;
    double maxDrop = 0.25;
};

struct ExpandConfig {
    double ratio = 0.20;         // troops per neutral-land push; strong players send many 20% pushes
    double minTroopFrac = 0.05;  // don't push below this fraction of maxTroops
    int cooldown = 5;
};

struct MilitaryConfig {
    double ratio = 0.20;              // troops committed per attack on a player
    double minTroopFrac = 0.20;       // only start player attacks when this full
    double advantage = 2.0;           // 2:1 over the defenders we expect to meet
    double botAdvantage = 0.6;        // bots defend poorly
    int minGainTiles = 80;            // an attack should be able to take at least this many tiles
    double distractedBonus = 1.5;     // prefer targets already fighting someone else
    double finishBonus = 2.0;         // prefer targets we can wipe out (conquest pays gold)
    double reserveFrac = 0.10;        // always keep this fraction of maxTroops home
    double incomingReserve = 1.1;     // ...or this multiple of troops attacking us, whichever is larger
    int maxConcurrent = 3;
    int cooldown = 10;
};

struct EconomyConfig {
    // Fallback cost estimates; normally the page sends the game's real costs.
    double cityCost = 125000;
    double factoryCost = 125000;
    double portCost = 125000;
    double postCost = 50000;
    double warshipCost = 250000;
    double costGrowth = 1.0;    // estimated cost multiplier per structure already owned
    int openingFactories = 2;   // factories before the first city
    int tilesPerCity = 250;     // don't build more cities than tiles / this
    int citiesPerFactory = 2;   // factories have diminishing returns: one per N cities
    int citiesPerPost = 3;      // standing defense posts per N cities (0 = only when attacked)
    int portsPerCity = 3;       // one port per N cities
    int maxWarships = 3;
    double goldReserve = 0;
    int cooldown = 20;
};

struct NavalConfig {
    bool enabled = true;
    double ratio = 0.30;
    double minTroopFrac = 0.40;
    int maxBoats = 3;
    int maxNeutralBorder = 40;  // still ship troops overseas while land expansion is down to this
    double maxTargetTroops = 0.5;   // land on players with at most this * our troops...
    double islandTargetTroops = 1.2; // ...or this, when there is nothing left to take over land
    int cooldown = 60;
};

struct DiplomacyConfig {
    bool enabled = true;
    double acceptIfStrongerFrac = 0.7;  // accept requests from players with >= this * our troops
    bool acceptNeighbors = true;        // accept non-bot neighbours we are not currently attacking
    double requestIfStrongerFrac = 1.2; // ask strong neighbours for alliances
    double leaderAllyFrac = 2.0;        // only ally the #1 player if they border us and have this * our troops
    bool embargoEnemies = true;         // stop trading with players we're at war with
    bool allowBetrayal = false;
    int requestCooldown = 300;
};

struct StrategicConfig {
    bool enabled = true;
    double siloCost = 1000000;
    double samCost = 1500000;
    double atomCost = 750000;
    double hydrogenCost = 5000000;
    int minTilesForSilo = 300;    // strong players have a silo within the first minute
    int minCitiesForSilo = 3;     // ...but a silo costs ~8 cities: get an economy first instead of hoarding for it
    int citiesPerSilo = 3;        // more silos as we grow; spread out they keep a second-strike option
    int maxSilos = 12;
    int citiesPerSam = 2;
    // Deterrence: nuking first makes us everyone's target. Without firstStrike we only hit players who
    // nuked us, or a runaway leader holding more than leaderShare of the land.
    bool firstStrike = false;
    double leaderShare = 0.35;
    double mountainWeight = 1.0;  // place silos/SAMs on high ground
    int nukeCooldown = 80;
    int cooldown = 40;
};

// Neural decision layer (src/learn). Two value nets score candidate troop and gold decisions.
// Every decision is written to an episode file; after each game the nets are retrained on all episodes
// to predict the discounted gain in "position" (land share, wealth, army) plus the win/loss outcome.
struct LearningConfig {
    bool enabled = true;
    std::string modelDir = "models";  // relative to the config file
    int hidden = 128;               // troop/gold nets: inputs -> hidden -> hidden -> hidden/2 -> 1
    int ensemble = 5;               // independent copies of each net; their disagreement = uncertainty
    double caution = 0.5;           // moves the rules wouldn't make are marked down by this * disagreement
    bool study = true;              // keep training in the background while the game runs
    int studyHorizon = 300;         // ticks until a decision of the running game is studied
    double lr = 1e-3;
    int batch = 64;
    int replay = 200000;            // newest decisions (per net) used for training
    int maxEpisodes = 400;          // older episode files are deleted
    int trainSteps = 3000;          // minibatch steps after each game
    int horizon = 1500;             // discount horizon in ticks: gains this far ahead count 1/e
    // Position value: wLand*sqrt(land share) + wGold*log(1 + wealth/100k) + wTroops*troops/max
    // + wCap*log(1 + maxTroops/100k). Wealth = gold + what our structures cost.
    double wLand = 10.0;
    double wGold = 1.0;
    double wTroops = 0.5;
    double wCap = 0.5;
    double terminalWin = 10.0;
    double terminalLoss = -10.0;
    double epsilon = 0.15;          // chance to sample a move by score instead of taking the best
    double epsilonDecay = 5e-5;     // eps = epsilon / (1 + samples * epsilonDecay)
    double temperature = 0.3;       // softmax temperature of that sampling (scores are in std units)
    double prior = 1.0;             // bonus for the rule-based choice while the nets are young
    double priorHalfLife = 20000;   // samples until that bonus halves
    double priorFloor = 0.1;        // ...but never below this: the rules stay the tie-breaker
    double fullTroops = 0.9;        // at this troop fill, waiting is not an option if a sane move exists
    int troopCooldown = 10;
    int goldCooldown = 20;
};

struct Config {
    SpawnConfig spawn;
    SustainConfig sustain;
    ExpandConfig expand;
    MilitaryConfig military;
    EconomyConfig economy;
    NavalConfig naval;
    DiplomacyConfig diplomacy;
    StrategicConfig strategic;
    LearningConfig learning;
};

Config loadConfig(const std::string& path);

}  // namespace fw
