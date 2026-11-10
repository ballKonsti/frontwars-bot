# FrontWars recon (2026-10-20)

Findings from inspecting https://frontwars.io in a live browser.

## Client architecture
- Vite SPA. Entry bundle `assets/CJ93GxYg.js`; game client chunk `Bzbr2u2v.js`;
  simulation runs in a Web Worker (`BLfjY75a.js`) backed by `engine.wasm`.
- The codebase is a descendant of OpenFront: same intent names, same `Turn {turnNumber, intents, hash}` schema,
  same `GameView` / `PlayerView` API on the main thread.
- Single player uses an in-page `LocalServer` (no network); lobbies use a WebSocket via the same `Transport` class.
  The transport has anti-bot checks (`refusedByAntibot`) on remote connections. We do not touch those.

## Getting at state (no screen reading needed)
Minified names change between builds, but **property names on instances survive minification**.
Defining a one-shot setter on `Object.prototype` before the bundle runs captures instances on first assignment:

| property          | captured instance | what it gives us |
|-------------------|-------------------|------------------|
| `lastUpdate`      | `GameView` (has `_players`) | full map + players + units |
| `localServer`     | `Transport` | `send({type:'intent', intent})` |

Useful `GameView` methods: `myPlayer() players() playerBySmallID(n) ownerID(tile) isLand isOceanShore isShore
neighbors(tile) x(t) y(t) ref(x,y) width() height() ticks() inSpawnPhase() units() nukesInFlight() myClientID()`.

Useful `PlayerView` methods: `id() smallID() name() type() isAlive() numTilesOwned() troops() maxTroops()
troopIncreaseRate() gold() (BigInt) allies() isAlliedWith(p) isFriendly(p) isRequestingAllianceWith(p) isTraitor()
outgoingAttacks() incomingAttacks() units() hasSpawned()`, plus async worker queries
`borderTiles() -> {borderTiles}` and `actions(tile) -> {canAttack, buildableUnits:[{type, canBuild: tile|false, canUpgrade, cost}]}`.

Troop values are internal units (HUD shows /10). Gold is a BigInt.

## Intent formats (verified against the zod schemas in the bundle)
All intents also carry `clientID` (= `GameView.myClientID()`).

```
attack              {targetID: playerId|null, troops}        null = neutral land
spawn               {tile}
boat                {targetID|null, troops, dst: tile, src: tile|null}
build_unit          {unit: "City"|"Port"|"Defense Post"|"Missile Silo"|"SAM Launcher"|"Warship"|"Atom Bomb"|"Hydrogen Bomb"|"MIRV"|..., tile}
upgrade_structure   {unit, unitId}
alliance_request    {recipient}
alliance_request_reply {requestor, accept}
break_alliance      {recipient}
donate_troops       {recipient, troops|null}
donate_gold         {recipient, gold}
cancel_attack       {attackID}
move_warship        {unitId, tile}
target_player       {target}
```

Verified live: UI click produced `{"type":"attack","clientID":"…","targetID":"jnewnbr8","troops":57075}`, and
sending the same shape through `Transport.send` started an attack.

## Game rules seen in the worker `initialized` message
spawn phase 100 turns, max 3 boats, alliance request duration 200 / cooldown 300 ticks, traitor 300 ticks
with 0.5 defense debuff, defense post range 30, SAM/silo cooldown 75. Costs seen: City 125k, Port 125k,
Defense Post 50k, Warship 250k, Silo 1M, SAM 1.5M, Atom 750k, Hydrogen 5M, MIRV 35M.
