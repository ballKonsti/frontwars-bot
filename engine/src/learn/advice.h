#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "state.h"

namespace fw {

// One piece of research turned into something the engine can act on: "in this situation, prefer this
// kind of move". Matching options get `strength` added to their score, the same way the rules' own
// pick gets its bonus, so the bot tries the advice and the nets learn from how it goes.
//
// advice.json: {"items":[{"id":"...", "enabled":true, "strength":0.5, "text":"...",
//   "when":{"phase":"early|mid|late","troopFillMin":0..1,"troopFillMax":0..1,"landShareMin":0..1,
//           "landShareMax":0..1,"neutralLand":bool,"underAttack":bool,"citiesMin":n,"citiesMax":n,"goldMin":n},
//   "prefer":{"decision":"troop|gold","kind":"wait|expand|attack|boat|save|City|Factory|Port|Defense Post|
//             upgrade City|Missile Silo|SAM Launcher|Warship","ratioMin":0..1,"ratioMax":0..1,
//             "target":"bot|nation|weaker|stronger|attacker"}}]}
struct AdviceItem {
    std::string id, text;
    double strength = 0;  // -1..1; negative = avoid
    // when
    std::string phase;
    double troopFillMin = 0, troopFillMax = 9, landShareMin = 0, landShareMax = 9, goldMin = 0;
    int neutralLand = -1, underAttack = -1;  // -1 = don't care
    int citiesMin = 0, citiesMax = 1 << 30;
    // prefer
    int net = 0;
    std::string kind, target;
    double ratioMin = 0, ratioMax = 9;
};

// What an option looks like to advice.
struct AdviceOption {
    std::string kind;   // wait / expand / attack / boat, or the gold option's name
    double ratio = 0;
    const Player* target = nullptr;
};

class Advice {
public:
    // Parses advice.json. Items with unknown words are rejected (listed in `errors`), never guessed.
    static Advice parse(const nlohmann::json& j, std::vector<std::string>* errors = nullptr);
    static Advice load(const std::string& path, std::vector<std::string>* errors = nullptr);

    bool empty() const { return items_.empty(); }
    size_t size() const { return items_.size(); }
    // Total bonus for one option in this state. `scale` converts strength into score units.
    double bonus(const State& s, int net, const AdviceOption& o, double scale) const;

private:
    std::vector<AdviceItem> items_;
};

}  // namespace fw
