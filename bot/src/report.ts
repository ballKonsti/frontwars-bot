// npm run report: compares game results per mode (rules / nets / feed) and code version.
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const file = path.join(root, 'logs', 'results.jsonl');
if (!existsSync(file)) {
  console.log('No results yet. Play some games first (npm run bot -- --eval 10).');
  process.exit(0);
}

interface Result {
  mode: string; rev: string; difficulty?: string; speed?: string; peakShare: number; share5m: number | null; ticks: number;
  goldEarned: number; won: boolean; eliminated: boolean; killedBy: string | null;
}
const rows: Result[] = readFileSync(file, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));

const groups = new Map<string, Result[]>();
for (const r of rows) {
  const key = `${r.mode} @ ${r.rev} ${r.difficulty ?? 'balanced'} x${r.speed ?? '1'}`;
  groups.set(key, [...(groups.get(key) ?? []), r]);
}

// Mean with the standard error, so a difference smaller than the +- is not a difference.
function stat(xs: number[], scale = 1, digits = 1): string {
  if (!xs.length) return '-';
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, xs.length - 1));
  return `${(m * scale).toFixed(digits)} +-${((sd / Math.sqrt(xs.length)) * scale).toFixed(digits)}`;
}

const table = [...groups].map(([key, g]) => ({
  version: key,
  games: g.length,
  'peak land %': stat(g.map((r) => r.peakShare), 100, 2),
  'land % at 5 min': stat(g.filter((r) => r.share5m != null).map((r) => r.share5m!), 100, 2),
  'survived min': stat(g.map((r) => r.ticks / 600)),
  'gold earned k': stat(g.map((r) => r.goldEarned / 1000), 1, 0),
  'won %': ((100 * g.filter((r) => r.won).length) / g.length).toFixed(0),
  'eliminated %': ((100 * g.filter((r) => r.eliminated).length) / g.length).toFixed(0),
}));
console.table(table);

const killers = new Map<string, number>();
for (const r of rows) if (r.killedBy) killers.set(r.killedBy, (killers.get(r.killedBy) ?? 0) + 1);
if (killers.size) {
  console.log('Eliminated most often by:');
  for (const [name, n] of [...killers].sort((a, b) => b[1] - a[1]).slice(0, 8)) console.log(`  ${n}x ${name}`);
}
