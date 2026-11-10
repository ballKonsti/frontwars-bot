# Notes from "FrontWars Gameplay Walkthrough FULL GAME" (no commentary)

Source: `~/Downloads/FrontWars Gameplay Walkthrough FULL GAME [4K 60FPS PC] - No Commentary.mp4` (1080p, 55 min).
- **Game 1** (0:00–17:00): single player, Australia map, Classic economy. The player wins.
- **Game 2** (19:00–55:00): World map. The player ("Guest973", southern Africa) is gone by ~21:00, and the rest is
  spectating the AI nation Spain win. That part is only useful for seeing how strong AI nations play
  (rail networks everywhere, constant nukes).

Read from HUD close-ups taken every 20 s.

## Game 1 timeline
| time | land | troops / max (regen) | gold | City/Fac/Port/Post/Silo/SAM/Warship |
|---|---|---|---|---|
| 0:40 | – | 21.6K / 55K (+1.19K) | 145K | 0/2/0/1/0/0/0 |
| 1:00 | – | 18.7K / 93.5K | 268K | 1/2/0/1/1/1/0 |
| 2:20 | 2.5% (#3) | 23.7K / 108K | 594K | 2/2/2/1/1/2/0, "MIRV INBOUND", buildings destroyed |
| 4:00 | 5.0% (#4) | 595 / 171K | 1.09M | 4/1/1/2/1/2/1, boats out, captured a City |
| 5:40 | 12.1% (#2) | 61.8K / 261K | 3.71M | 6/2/0/4/3/3/2 |
| 7:00 | 32.5% (#1) | 105K / 429K | 6.59M | 10/7/0/7/4/7/2 |
| 9:00 | 67.4% | 152K / 666K | 17.3M | 16/11/1/10/8/9/2 |
| 11:20 | 95.3% | 28.4K / 934K | 32.9M | 22/12/7/11/9/10/2 |
| 15:00 | 99.4% | 997 / 1.42M | 38.2M | 41/20/9/14/11/12/3 |

## What the player does
1. **Attack ratio 20%, the whole game** (41% only for the final mop-up). They send many 20% attacks back to back
   instead of a few big ones.
2. **Troops run low.** They're often at 1–25% of max because everything goes into attacks. Regen per tick is highest
   at roughly 40–60% of max (≈2.3% of max per tick) and drops when nearly empty, but land and conquest gold are
   worth more.
3. **Factories first.** 2 Factories (and a Defense Post) in the first 40 s, the first City at ~1:00. After that,
   Factories ≈ Cities / 2 all game, which matches our `citiesPerFactory = 2`.
4. **Silo and SAM very early** (both by 1:00). Later Silos ≈ SAMs ≈ Cities / 2 – Cities / 4. AI nations nuke a lot
   (a MIRV at 2:20, repeated "your City/Silo/Post was destroyed"). The player keeps rebuilding.
5. **Defense Posts all game,** about one per 2–3 Cities, not only when attacked.
6. **Boats early** (from ~3:40) to islands like Tasmania, while land targets still exist. 2–3 Warships from ~5 min.
7. **Conquest pays:** each eliminated bot or nation gives 30K–850K gold ("Conquered X received 275K gold").
   That plus trains drives gold from 1M (4:00) to 17M (9:00).
8. **Snowball:** they stay #3–4 until ~5:30, then go from 12% → 67% land in 3 minutes.
9. **Builds are cheap early.** Two factories and a post with under 150K gold earned. Our hard-coded cost
   estimates (City 125K, Silo 1M, SAM 1.5M) are wrong for early game, so the bot must read real costs from the game.
