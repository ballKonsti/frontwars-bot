import { createWriteStream, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { parseArgs } from 'node:util';
import type { Page } from 'playwright';
import { launch, newGamePage, startSinglePlayer, waitForGame } from './launcher.js';
import { Engine } from './engine.js';
import { Tuner, scoreGame, type GameStats } from './tuner.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

const { values: args } = parseArgs({
  options: {
    // single: start a single-player game automatically.
    // manual: you open/join a private lobby yourself; the bot takes over once the game starts.
    mode: { type: 'string', default: 'single' },
    engine: { type: 'string', default: path.join(root, 'engine/build/fw_engine') },
    config: { type: 'string', default: path.join(root, 'engine/config.json') },
    interval: { type: 'string', default: '200' },
    // Only useful once the profile has passed the site's checks; otherwise you can't see them.
    headless: { type: 'boolean', default: false },
    games: { type: 'string', default: '1' },
    browser: { type: 'string', default: 'firefox' },
    // Auto-tune config knobs between games (state in logs/tuner.json).
    tune: { type: 'boolean', default: false },
    'max-minutes': { type: 'string', default: '40' },
    // Play this many single-player games at once (one browser window each) to feed the learner faster.
    parallel: { type: 'string', default: '1' },
  },
});

if (args.browser !== 'firefox' && args.browser !== 'chrome') {
  console.error('--browser must be "firefox" or "chrome"');
  process.exit(2);
}

if (args.mode !== 'single' && args.mode !== 'manual') {
  console.error('--mode must be "single" or "manual"');
  process.exit(2);
}

const interval = Number(args.interval);
const maxGameMs = Number(args['max-minutes']) * 60_000;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

mkdirSync(path.join(root, 'logs'), { recursive: true });
// The tuned config sits next to config.json so the learner's relative modelDir resolves the same way.
const tunedConfig = path.join(path.dirname(args.config!), 'config.tuned.json');
const tuner = args.tune ? new Tuner(path.join(root, 'logs', 'tuner.json'), args.config!) : null;

// Firefox and Chrome profiles are incompatible, so each gets its own directory.
const parallel = Math.max(1, Number(args.parallel));
if (parallel > 1 && (args.mode !== 'single' || args.tune)) {
  console.error('--parallel needs --mode single and cannot be combined with --tune');
  process.exit(2);
}

const { context, page: firstPage } = await launch({
  browser: args.browser,
  headless: args.headless!,
  profileDir: path.join(root, `.profile-${args.browser}`),
});

async function playOne(page: Page, gameNo: number): Promise<void> {
  const say = (msg: string) => console.log(parallel > 1 ? `[game ${gameNo}] ${msg}` : msg);
  let configPath = args.config!;
  if (tuner) {
    say(`tuner: playing ${tuner.prepare(tunedConfig)}`);
    configPath = tunedConfig;
  }

  if (args.mode === 'single') await startSinglePlayer(page);
  else say('Waiting for you to start or join a game in the browser window…');
  // Generous timeouts: you may need to pass the site's human check before the game starts.
  await waitForGame(page, args.mode === 'single' ? 5 * 60_000 : 60 * 60_000);

  const engine = new Engine(args.engine!, configPath);
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const logPath = path.join(root, 'logs', `game-${stamp}${parallel > 1 ? `-${gameNo}` : ''}.jsonl`);
  const log = createWriteStream(logPath);
  say(`game ${gameNo} started, logging to ${logPath}`);

  const started = Date.now();
  const stats: GameStats = { peakShare: 0, reached72: false, won: false, survivedMs: 0 };
  let eliminated = false;
  let lastStatus = 0;
  let recent: string[] = [];

  for (;;) {
    const t0 = Date.now();
    const state: any = await page.evaluate(() => (window as any).__fwbot.state()).catch(() => null);
    if (!state?.ready) break;
    if (state.gameOver) {
      stats.won = state.won;
      say(state.won ? 'we won!' : `game over, winner: ${state.winnerName ?? 'someone else'}`);
      break;
    }
    if (state.spawned && !state.me.alive) {
      eliminated = true;
      say('eliminated');
      break;
    }
    if (Date.now() - started > maxGameMs) {
      say(`stopping after ${args['max-minutes']} minutes`);
      break;
    }

    const reply = await engine.decide(state);
    const results = reply.actions.length
      ? await page.evaluate((a) => (window as any).__fwbot.exec(a), reply.actions).catch((e) => [String(e)])
      : [];
    log.write(JSON.stringify({ tick: state.tick, me: state.me, actions: reply.actions, results, log: reply.log }) + '\n');
    recent.push(...(reply.log ?? []));

    if (state.spawned) {
      const share = state.me.tiles / state.landTiles;
      stats.peakShare = Math.max(stats.peakShare, share);
      if (Date.now() - lastStatus > 5000) {
        lastStatus = Date.now();
        say(
          `tick ${state.tick}  land ${(100 * share).toFixed(1)}%  troops ${Math.round(state.me.troops / 10)}  gold ${state.me.gold}`,
        );
        for (const l of [...new Set(recent)].slice(-6)) say(`  ${l}`);
        recent = [];
      }
      if (share >= 0.72) {
        stats.reached72 = true;
        say('reached 72% of the map');
        break;
      }
    }
    await sleep(Math.max(0, interval - (Date.now() - t0)));
  }

  stats.survivedMs = Date.now() - started;
  log.end();
  await engine.finish({ won: stats.won || stats.reached72, eliminated });
  const score = scoreGame(stats);
  say(`game ${gameNo}: peak land ${(100 * stats.peakShare).toFixed(1)}%, score ${score.toFixed(3)}`);
  if (tuner) say(`tuner: ${tuner.report(score)}`);
}

const games = Number(args.games);
let nextGame = 1;

// Each worker owns one window and plays games back to back until the total is reached.
async function worker(page: Page): Promise<void> {
  for (;;) {
    const gameNo = nextGame++;
    if (gameNo > games) return;
    if (gameNo > parallel || page !== firstPage) await page.goto('https://frontwars.io/', { waitUntil: 'domcontentloaded' });
    try {
      await playOne(page, gameNo);
    } catch (e) {
      console.error(`game ${gameNo} failed: ${String(e).split('\n')[0]}`);
      await sleep(5000);
    }
  }
}

const pages = [firstPage];
for (let i = 1; i < Math.min(parallel, games); i++) pages.push(await newGamePage(context));
await Promise.all(pages.map(worker));
await context.close();
