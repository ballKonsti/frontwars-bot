#include "config.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace fw {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(SpawnConfig, edgeWeight, reconsiderTick)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(SustainConfig, window, maxDrop)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ExpandConfig, ratio, minTroopFrac, cooldown)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(MilitaryConfig, ratio, minTroopFrac, advantage, botAdvantage, minGainTiles,
                                                distractedBonus, finishBonus, reserveFrac, incomingReserve, maxConcurrent, cooldown)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(EconomyConfig, cityCost, factoryCost, portCost, postCost, warshipCost,
                                                costGrowth, openingFactories, tilesPerCity, citiesPerFactory,
                                                citiesPerPost, portsPerCity, maxWarships, goldReserve, cooldown)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(NavalConfig, enabled, ratio, minTroopFrac, maxBoats, maxNeutralBorder,
                                                maxTargetTroops, islandTargetTroops,
                                                cooldown)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(DiplomacyConfig, enabled, acceptIfStrongerFrac, acceptNeighbors,
                                                requestIfStrongerFrac, leaderAllyFrac, embargoEnemies, allowBetrayal,
                                                requestCooldown)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(StrategicConfig, enabled, siloCost, samCost, atomCost, hydrogenCost,
                                                minTilesForSilo, minCitiesForSilo, citiesPerSilo, maxSilos, citiesPerSam, firstStrike, leaderShare,
                                                mountainWeight, nukeCooldown, cooldown)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(LearningConfig, enabled, modelDir, hidden, ensemble, caution, study, record, explore,
                                                studyHorizon, lr, batch, replay, maxEpisodes,
                                                trainSteps, horizon, wLand, wGold, wTroops, wCap, terminalWin,
                                                terminalLoss, epsilon, epsilonDecay, temperature, prior, priorHalfLife,
                                                priorFloor, fullTroops, troopCooldown, goldCooldown)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Config, spawn, sustain, expand, military, economy, naval, diplomacy,
                                                strategic, learning)

Config loadConfig(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config " + path);
    return nlohmann::json::parse(in, nullptr, true, /*ignore_comments=*/true).get<Config>();
}

}  // namespace fw
