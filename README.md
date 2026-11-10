# frontwars-bot

Auto-player for [FrontWars](https://frontwars.io), for **single player and private lobbies with friends who know a bot is playing**.
It does not join public matchmaking and does not try to get around the site's human checks.

```
Chrome (Playwright) ── page-agent.js ──state JSON──▶ fw_engine (C++)
   reads GameView, sends intents  ◀──actions JSON──  strategy modules, ~5 Hz
```

- `bot/src/page-agent.js`: injected before the game loads. It captures the game's own `GameView` and `Transport`
  objects (see `docs/recon.md`), so state is exact and actions are the same intents the UI sends. No screen reading.
- `bot/src/main.ts`: launches Chrome, starts or waits for a game, and runs the state → engine → actions loop.
  Each game is logged to `logs/*.jsonl`.
- `engine/`: the C++20 decision engine. Modules run each tick in this order:
  diplomacy → strategic (silo/SAM/nukes) → economy (posts/cities/ports/upgrades) → military → expand → naval.
  They share troop and gold budgets, so they can't overspend, and a home reserve is kept back when you're attacked.
- `engine/config.json`: every tuning knob (commented). Edit it and rerun; no rebuild needed.

## Setup
```sh
cd engine && cmake -S . -B build && cmake --build build -j && ctest --test-dir build
cd ../bot && npm install && npx playwright install firefox
```
By default it runs Playwright's Firefox (a one-time download of about 80 MB). Pass `--browser chrome` to use the
system Chrome (`/usr/bin/google-chrome-stable` or `/usr/bin/chromium`) instead. Chrome shows an
"automated test software" bar, but the page now sizes itself to the real window.

## Run
```sh
cd bot
npm run bot                      # single player vs AI nations, one game
npm run bot -- --games 5         # play 5 single-player games in a row (for tuning)
npm run bot -- --mode manual     # you create/join a private lobby in the window; the bot plays once it starts
npm run bot -- --games 40 --parallel 4 --headless   # feed the networks: 4 games at a time, no windows
```
On first run, a cookie-consent banner and a Cloudflare "Quick check" may appear. Handle them yourself in the
window. The bot waits up to 5 minutes for you, and the persistent profile in `.profile-<browser>/` keeps your choices.

## Strategy (defaults)
- **Spawn:** lots of free land, biased toward corners and edges, so there are fewer fronts to defend.
- **Early:** take neutral land with 35% pushes and farm bots. Human-like nations aren't attacked while free land remains.
- **Sustainability:** if troops fall more than 25% within ~6 s while attacking, no new offensives until they recover.
- **Build order:** the first City comes before any other spending. Then Cities, then Factories (one per 2 cities,
  because returns diminish), then Ports at the far ends of the coast (longer trade routes), then City upgrades.
  Defense posts go up only facing a serious attacker.
- **Attacks:** aim for a 2:1 edge over the defenders' troop density. Prefer wide fronts, players attacking us, and
  players busy fighting someone else.
- **Diplomacy:** ally strong neighbours, but not the #1 player unless they border us and are far stronger.
  Embargo anyone we're at war with. Betrayal is off.
- **Nukes (deterrence):** silos on high ground, plus a second silo for second-strike and SAMs once anyone has silos.
  We nuke only players who nuked us, or a leader holding over 35% of the map (`strategic.firstStrike` changes this).

## Learning
The decisions are made by neural networks (`engine/src/learn/`), with the rules above as their teacher and safety net.

- **Three ensembles.** Troop moves (wait, expand, attack any of up to 10 neighbours, boats; each at 10/20/35/50/75%),
  gold (save, City, Factory, Port, Defense Post, City upgrade, Silo, SAM, Warship) and the spawn site (40 candidate
  sites, scored before the game begins and once more late in the spawn phase when the others have picked).
  Each is 5 independent networks. Where they disagree they don't know yet, and that move is marked down
  (`caution`), so the nets overrule the rules only when they agree.
- **What they learn.** For every decision: how much the position improves afterwards, looking about 2.5 minutes ahead
  (`horizon`), plus the result of the game. Position = land share (`wLand`), wealth = gold + structures (`wGold`),
  army (`wTroops`, `wCap`); a win adds `terminalWin`, elimination `terminalLoss`. Raise `wGold` for a greedier bot.
- **Studying during the game.** A background thread trains the whole time: on all recorded games, plus this game's
  own decisions 30 s after they were made (`studyHorizon`). The playing copy of the nets is refreshed several times
  a second.
- **Studying between games.** Every decision is written to `engine/models/episodes/*.jsonl`. When a game ends, all
  nets are retrained on the newest 200k decisions and saved. The engine prints the held-out error: 1.0 means the
  net knows nothing yet, lower is better.
- **Rules as a head start.** The rule-based choice gets a bonus (`prior`) that halves every 20k recorded decisions,
  and at full troops "wait" is not an option. Hard limits stay: no attacking allies, keep the home reserve, only
  spend what we have. Diplomacy and nuke launches stay rule-based.

Feed it by playing. Several games at once, without windows:

```sh
npm run bot -- --games 40 --parallel 4 --headless    # 4 single-player games at a time
../engine/build/fw_engine --train ../engine/config.json   # retrain only, 10x the usual steps
```
`--headless` only works once the profile has passed the consent banner and the site's check in a normal run.
Delete `engine/models/` to start from scratch; `"learning": {"enabled": false}` gives pure rules.

- **Auto-tuning (`--tune`).** Between games, an evolution strategy adjusts the main `config.json` knobs (attack
  advantage, reserves, build ratios, silo timing, exploration). It keeps a change only if it scores better over 2 games.
  Progress is in `logs/tuner.json`, and the config actually used is `engine/config.tuned.json`.
  Score = peak land share + 1 for a win + up to 0.2 for surviving 20 minutes. Not combinable with `--parallel`.

Expect it to need a few hundred games before the nets clearly beat the plain rules.

## Tuning
Each `logs/game-*.jsonl` line holds the tick, your stats, the actions sent and their results.
Change `config.json`, run `--games N`, and compare how fast land share grows.
Betrayal is off by default (`diplomacy.allowBetrayal`).

## When the site updates
Minified code changes don't matter: the agent relies only on instance property names
(`lastUpdate`, `localServer`) and the public `GameView`/`PlayerView` methods. If capture breaks, redo the steps in
`docs/recon.md`.
