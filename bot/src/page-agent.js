// Injected into the FrontWars page before any game script runs (Playwright addInitScript).
// Captures the GameView and Transport instances and exposes window.__fwbot with
// state() / exec(actions). Plain JS: it is serialized and evaluated in the page.
(() => {
  if (window.top !== window || !/frontwars\.io$/.test(location.hostname)) return;

  const cap = { gameView: null, transport: null, win: null, costs: null, costsTick: -1e9, spawnC: null, spawnTick: -1e9 };

  // One-shot capture: the first assignment of these properties on an instance
  // lands here; we turn it into a normal data property on that instance.
  function captureOn(prop, test, slot) {
    Object.defineProperty(Object.prototype, prop, {
      configurable: true,
      get() { return undefined; },
      set(v) {
        if (this === Object.prototype) return;
        Object.defineProperty(this, prop, { value: v, writable: true, configurable: true, enumerable: true });
        if (test(this)) cap[slot] = this;
      },
    });
  }
  captureOn('lastUpdate', (o) => '_players' in o, 'gameView');
  captureOn('localServer', (o) => typeof o.send === 'function' || 'socket' in o, 'transport');

  const ACTIONS_CONCURRENCY = 8;
  const rnd = (n) => Math.floor(Math.random() * n);

  function gv() { return cap.gameView; }
  function me() { return cap.gameView?.myPlayer() ?? null; }

  function ready() {
    const g = gv();
    return !!(g && cap.transport && g.ticks() > 0);
  }

  function send(intent) {
    cap.transport.send({ type: 'intent', intent: { clientID: gv().myClientID(), ...intent } });
  }

  async function borderTiles(p) {
    const r = await p.borderTiles();
    return [...(r.borderTiles ?? r)];
  }

  // Ask the worker which of these tiles can host `unit`; returns the first buildable tile.
  async function firstBuildable(p, tiles, unit) {
    for (let i = 0; i < tiles.length; i += ACTIONS_CONCURRENCY) {
      const batch = tiles.slice(i, i + ACTIONS_CONCURRENCY);
      const res = await Promise.all(batch.map((t) => p.actions(t).catch(() => null)));
      for (const r of res) {
        const b = r?.buildableUnits?.find((u) => u.type === unit);
        if (b && b.canBuild !== false) return b.canBuild;
      }
    }
    return null;
  }

  function myUnits(p) {
    return p.units().map((u) => ({ id: u.id(), type: u.type(), tile: u.tile(), level: u.level?.() ?? 1 }));
  }

  // Random owned land tiles inside the bounding box of our border, scored for a given site kind:
  // interior = deep inside, coast = on the ocean, coast-far = on the ocean as far as possible from our
  // other ports (long trade routes). mountainWeight favours high ground (magnitude >= 20 is Mountain).
  function candidateSites(p, border, kind, unit, nearSmall, mountainWeight = 0) {
    const g = gv();
    const mine = p.smallID();
    if (kind === 'border' && nearSmall != null) {
      const out = border.filter((t) => g.neighbors(t).some((n) => g.ownerID(n) === nearSmall));
      out.sort(() => Math.random() - 0.5);
      return out.slice(0, 24);
    }
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const t of border) {
      const x = g.x(t), y = g.y(t);
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    if (!isFinite(x0)) return [];
    const borderSample = border.length > 200 ? Array.from({ length: 200 }, () => border[rnd(border.length)]) : border;
    const same = p.units().filter((u) => u.type() === unit).map((u) => u.tile());
    const scored = [];
    for (let i = 0; i < 400; i++) {
      const t = g.ref(x0 + rnd(x1 - x0 + 1), y0 + rnd(y1 - y0 + 1));
      if (g.ownerID(t) !== mine || !g.isLand(t)) continue;
      const coastal = kind === 'coast' || kind === 'coast-far';
      if (coastal && !g.isOceanShore(t)) continue;
      let dBorder = Infinity;
      for (const b of borderSample) dBorder = Math.min(dBorder, g.manhattanDist(t, b));
      let dSame = kind === 'coast-far' ? 4000 : 60;
      for (const s of same) dSame = Math.min(dSame, g.manhattanDist(t, s));
      const mag = g.magnitude(t);
      const height = mag >= 20 ? 1 : mag >= 10 ? 0.5 : 0;
      const score = (kind === 'interior' ? Math.min(dBorder, 40) : 0) + dSame + 30 * mountainWeight * height;
      scored.push([score, t]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    return scored.slice(0, 24).map((s) => s[1]);
  }

  // Free land around the tile, boosted near map edges/corners (fewer fronts to defend).
  function chooseSpawnTile(edgeWeight = 0) {
    const g = gv();
    const W = g.width(), H = g.height();
    const edgeness = (x, y) => (1 - Math.min(1, (4 * Math.min(x, W - 1 - x)) / W)) +
      (1 - Math.min(1, (4 * Math.min(y, H - 1 - y)) / H));
    let best = null, bestScore = -1;
    for (let i = 0; i < 300; i++) {
      const t = g.ref(rnd(W), rnd(H));
      if (!g.isLand(t) || g.hasOwner(t)) continue;
      let free = 0;
      for (let k = 0; k < 40; k++) {
        const dx = rnd(41) - 20, dy = rnd(41) - 20;
        const x = g.x(t) + dx, y = g.y(t) + dy;
        if (!g.isValidCoord(x, y)) continue;
        const n = g.ref(x, y);
        if (g.isLand(n) && !g.hasOwner(n)) free++;
      }
      const score = free * (1 + (edgeWeight * edgeness(g.x(t), g.y(t))) / 2);
      if (score > bestScore) { bestScore = score; best = t; }
    }
    return best;
  }

  // Candidate spawn sites for the engine's spawn net: the 40 best by the rule of thumb above, each with
  // what surrounds it at three radii. h is that rule's score (1 = a fully free corner).
  function spawnCandidates() {
    const g = gv();
    if (cap.spawnC && g.ticks() - cap.spawnTick < 10 && g.ticks() >= cap.spawnTick) return cap.spawnC;
    const W = g.width(), H = g.height();
    const edgeness = (x, y) => (1 - Math.min(1, (4 * Math.min(x, W - 1 - x)) / W)) +
      (1 - Math.min(1, (4 * Math.min(y, H - 1 - y)) / H));
    // Fractions of sampled tiles within +-r that are free land / owned / land at all.
    const around = (t, r, n) => {
      let free = 0, owned = 0, land = 0;
      for (let k = 0; k < n; k++) {
        const x = g.x(t) + rnd(2 * r + 1) - r, y = g.y(t) + rnd(2 * r + 1) - r;
        if (!g.isValidCoord(x, y)) continue;
        const q = g.ref(x, y);
        if (!g.isLand(q)) continue;
        land++;
        if (g.hasOwner(q)) owned++; else free++;
      }
      return [free / n, owned / n, land / n];
    };
    const pool = [];
    for (let i = 0; i < 300; i++) {
      const t = g.ref(rnd(W), rnd(H));
      if (!g.isLand(t) || g.hasOwner(t)) continue;
      const near = around(t, 20, 40);
      const edge = edgeness(g.x(t), g.y(t));
      pool.push({ t, near, edge, h: (near[0] * (1 + edge / 2)) / 2 });
    }
    pool.sort((a, b) => b.h - a.h);
    const r3 = (v) => Math.round(v * 1000) / 1000;
    cap.spawnC = pool.slice(0, 40).map((c) => {
      const mid = around(c.t, 60, 60), far = around(c.t, 150, 80);
      return { tile: c.t, h: r3(c.h),
        f: [c.near[0], mid[0], far[0], mid[1], far[1], mid[2], c.edge / 2, g.x(c.t) / W, g.y(c.t) / H].map(r3) };
    });
    cap.spawnTick = g.ticks();
    return cap.spawnC;
  }

  // Shore tiles we don't own, ranked by distance from a random point of our coast.
  function boatTargets(p, border) {
    const g = gv();
    const mine = p.smallID();
    const coast = border.filter((t) => g.isOceanShore(t));
    if (!coast.length) return [];
    const W = g.width(), H = g.height();
    const out = [];
    for (let i = 0; i < 600 && out.length < 40; i++) {
      const t = g.ref(rnd(W), rnd(H));
      if (!g.isOceanShore(t) || g.ownerID(t) === mine) continue;
      const from = coast[rnd(coast.length)];
      out.push({ tile: t, owner: g.ownerID(t), dist: g.manhattanDist(t, from) });
    }
    out.sort((a, b) => a.dist - b.dist);
    return out.slice(0, 8);
  }

  // The game emits a Win update once; remember it. Update arrays are keyed by an enum we don't
  // hard-code, so look for the entry shape instead.
  function checkWin(g) {
    if (cap.win) return cap.win;
    const updates = g.lastUpdate?.updates;
    if (!updates) return null;
    for (const list of Object.values(updates)) {
      const w = Array.isArray(list) && list.find((u) => u && 'winner' in u);
      if (w) return (cap.win = { winner: w.winner ?? [], name: w.winnerDisplayName ?? '' });
    }
    return null;
  }

  // Real build costs (they rise with each structure owned). Costs are per player, so any owned tile works.
  async function buildCosts(p, tile) {
    const g = gv();
    if (cap.costs && g.ticks() - cap.costsTick < 10) return cap.costs;
    if (tile == null) return cap.costs ?? {};
    const r = await p.actions(tile).catch(() => null);
    if (r?.buildableUnits) {
      cap.costs = Object.fromEntries(r.buildableUnits.map((u) => [u.type, Number(u.cost)]));
      cap.costsTick = g.ticks();
    }
    return cap.costs ?? {};
  }

  const STRATEGIC = new Set(['City', 'Missile Silo', 'SAM Launcher', 'Port', 'Defense Post']);

  async function state() {
    if (!ready()) return { ready: false };
    const g = gv();
    const p = me();
    const win = checkWin(g);
    const base = { ready: true, tick: g.ticks(), inSpawn: g.inSpawnPhase(), mapW: g.width(), mapH: g.height(),
      landTiles: g.numLandTiles(), gameOver: !!win,
      won: !!win && !!p && (win.winner.includes(p.id()) || win.winner.includes(p.clientID?.())),
      winnerName: win?.name ?? null };
    if (!p || !p.hasSpawned?.()) return { ...base, spawned: false, spawnCandidates: base.inSpawn ? spawnCandidates() : [] };

    const border = await borderTiles(p);
    const mine = p.smallID();
    const nb = new Map(); // smallID -> shared border count
    for (const t of border) {
      for (const n of g.neighbors(t)) {
        if (!g.isLand(n)) continue;
        const o = g.ownerID(n);
        if (o !== mine) nb.set(o, (nb.get(o) ?? 0) + 1);
      }
    }

    const players = [];
    for (const q of g.players()) {
      if (!q.isAlive() || q.smallID() === mine) continue;
      const shared = nb.get(q.smallID()) ?? 0;
      // keep everyone we border plus the biggest players; skip the long tail
      if (!shared && q.numTilesOwned() < 200) continue;
      players.push({
        small: q.smallID(), id: q.id(), name: q.name(), type: String(q.type()),
        tiles: q.numTilesOwned(), troops: q.troops(), gold: Number(q.gold()),
        isTraitor: q.isTraitor(), allied: p.isAlliedWith(q), friendly: p.isFriendly(q),
        requestingAlliance: q.isRequestingAllianceWith?.(p) ?? false,
        embargoed: p.hasEmbargoAgainst?.(q) ?? false,
        border: shared,
        underAttack: q.incomingAttacks().length,
      });
    }

    const enemyStructures = [];
    for (const u of g.units()) {
      const o = u.owner();
      if (!STRATEGIC.has(u.type()) || o.smallID() === mine || p.isFriendly(o)) continue;
      if (!(nb.get(o.smallID()) || o.numTilesOwned() > 2000)) continue;
      enemyStructures.push({ owner: o.smallID(), type: u.type(), tile: u.tile(), level: u.level?.() ?? 1 });
      if (enemyStructures.length >= 60) break;
    }

    const inbound = g.nukesInFlight?.() ?? [];
    const aimedAtUs = inbound.filter((u) => u.targetTile?.() != null && g.ownerID(u.targetTile()) === mine);
    return {
      ...base,
      spawned: true,
      me: {
        small: mine, id: p.id(), tiles: p.numTilesOwned(), troops: p.troops(), maxTroops: p.maxTroops(),
        troopRate: p.troopIncreaseRate(), gold: Number(p.gold()), isTraitor: p.isTraitor(), alive: p.isAlive(),
        outgoing: p.outgoingAttacks().map((a) => ({ id: a.id, target: a.targetID, troops: a.troops })),
        incoming: p.incomingAttacks().map((a) => ({ id: a.id, attacker: a.attackerID, troops: a.troops })),
        units: myUnits(p),
        borderLen: border.length,
        coastal: border.some((t) => g.isOceanShore(t)),
      },
      neutralBorder: nb.get(0) ?? 0,
      costs: await buildCosts(p, border[0]),
      players,
      enemyStructures,
      nukesInbound: aimedAtUs.length,
      nukeSources: [...new Set(aimedAtUs.map((u) => u.owner().smallID()))],
      boatTargets: boatTargets(p, border),
      spawnCandidates: base.inSpawn ? spawnCandidates() : [],
    };
  }

  // Engine actions -> intents. Placement is resolved here because it needs worker queries.
  async function exec(actions) {
    if (!ready()) return [];
    const p = me();
    const results = [];
    for (const a of actions) {
      try {
        switch (a.type) {
          case 'spawn': {
            const t = a.tile ?? chooseSpawnTile(a.edgeWeight ?? 0);
            if (t != null) send({ type: 'spawn', tile: t });
            results.push({ type: a.type, ok: t != null });
            break;
          }
          case 'attack':
            send({ type: 'attack', targetID: a.target ?? null, troops: Math.floor(a.troops) });
            results.push({ type: a.type, ok: true });
            break;
          case 'boat':
            send({ type: 'boat', targetID: a.target ?? null, troops: Math.floor(a.troops), dst: a.dst, src: null });
            results.push({ type: a.type, ok: true });
            break;
          case 'build': {
            let tile = a.tile ?? null;
            if (tile != null) {
              tile = await firstBuildable(p, [tile], a.unit);
            } else {
              const border = await borderTiles(p);
              const sites = candidateSites(p, border, a.site ?? 'interior', a.unit, a.near, a.mountainWeight ?? 0);
              tile = await firstBuildable(p, sites, a.unit);
            }
            if (tile != null) send({ type: 'build_unit', unit: a.unit, tile });
            results.push({ type: a.type, unit: a.unit, ok: tile != null });
            break;
          }
          case 'nuke': {
            // For bombs canBuild is the launching silo; the intent carries the target tile.
            const r = await p.actions(a.tile);
            const b = r?.buildableUnits?.find((u) => u.type === a.unit);
            const ok = !!b && b.canBuild !== false;
            if (ok) send({ type: 'build_unit', unit: a.unit, tile: a.tile });
            results.push({ type: a.type, unit: a.unit, ok });
            break;
          }
          case 'upgrade':
            send({ type: 'upgrade_structure', unit: a.unit, unitId: a.unitId });
            results.push({ type: a.type, ok: true });
            break;
          case 'alliance_request':
            send({ type: 'alliance_request', recipient: a.recipient });
            results.push({ type: a.type, ok: true });
            break;
          case 'alliance_reply':
            send({ type: 'alliance_request_reply', requestor: a.requestor, accept: !!a.accept });
            results.push({ type: a.type, ok: true });
            break;
          case 'break_alliance':
            send({ type: 'break_alliance', recipient: a.recipient });
            results.push({ type: a.type, ok: true });
            break;
          case 'embargo':
            send({ type: 'embargo', targetID: a.target, action: a.action ?? 'start' });
            results.push({ type: a.type, ok: true });
            break;
          case 'cancel_attack':
            send({ type: 'cancel_attack', attackID: a.attackID });
            results.push({ type: a.type, ok: true });
            break;
          default:
            results.push({ type: a.type, ok: false, error: 'unknown action' });
        }
      } catch (e) {
        results.push({ type: a.type, ok: false, error: String(e) });
      }
    }
    return results;
  }

  window.__fwbot = { ready, state, exec, _cap: cap };
})();
