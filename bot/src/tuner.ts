import { existsSync, readFileSync, writeFileSync } from 'node:fs';

// Between-game auto-tuning: a (1+1) evolution strategy over a few config.json knobs.
// The incumbent is the best setting so far; each candidate is a mutation of it and replaces it only
// if its average score over EVALS games beats the incumbent's. Step size follows the 1/5 success rule.

interface Param {
  path: string; // dotted path into config.json
  min: number;
  max: number;
  log?: boolean;
  int?: boolean;
}

const PARAMS: Param[] = [
  { path: 'military.advantage', min: 1.0, max: 4.0, log: true },
  { path: 'military.botAdvantage', min: 0.2, max: 2.0, log: true },
  { path: 'military.minTroopFrac', min: 0.2, max: 0.8 },
  { path: 'military.reserveFrac', min: 0.05, max: 0.5 },
  { path: 'sustain.maxDrop', min: 0.1, max: 0.5 },
  { path: 'economy.tilesPerCity', min: 100, max: 800, log: true, int: true },
  { path: 'economy.citiesPerFactory', min: 1, max: 5, int: true },
  { path: 'economy.portsPerCity', min: 1, max: 6, int: true },
  { path: 'strategic.minTilesForSilo', min: 1000, max: 10000, log: true, int: true },
  { path: 'strategic.leaderShare', min: 0.2, max: 0.6 },
  { path: 'learning.epsilon', min: 0.02, max: 0.3, log: true },
];

const EVALS = 2;
const SIGMA_MIN = 0.02;
const SIGMA_MAX = 0.4;

type Values = Record<string, number>;

interface TunerState {
  incumbent: Values;
  incumbentScores: number[];
  candidate: Values | null;
  candidateScores: number[];
  sigma: number;
  history: Array<{ time: string; params: Values; score: number; role: 'incumbent' | 'candidate' }>;
}

export interface GameStats {
  peakShare: number;
  reached72: boolean;
  won: boolean;
  survivedMs: number;
}

export function scoreGame(g: GameStats): number {
  return g.peakShare + (g.reached72 || g.won ? 1 : 0) + 0.2 * Math.min(1, g.survivedMs / (20 * 60_000));
}

// config.json allows // comments; strip whole-line ones before parsing.
export function readConfig(path: string): any {
  return JSON.parse(readFileSync(path, 'utf8').replace(/^\s*\/\/.*$/gm, ''));
}

function getPath(obj: any, path: string): number | undefined {
  return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

function setPath(obj: any, path: string, value: number): void {
  const keys = path.split('.');
  let o = obj;
  for (const k of keys.slice(0, -1)) o = o[k] ??= {};
  o[keys[keys.length - 1]] = value;
}

// Parameters are mutated in [0,1] space so one sigma fits all of them.
const toUnit = (p: Param, v: number) =>
  p.log ? Math.log(v / p.min) / Math.log(p.max / p.min) : (v - p.min) / (p.max - p.min);
const fromUnit = (p: Param, u: number) => {
  const c = Math.min(1, Math.max(0, u));
  const v = p.log ? p.min * Math.pow(p.max / p.min, c) : p.min + c * (p.max - p.min);
  return p.int ? Math.round(v) : Number(v.toFixed(4));
};

function gaussian(): number {
  const u = 1 - Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

export class Tuner {
  private state: TunerState;

  constructor(private statePath: string, private baseConfigPath: string) {
    if (existsSync(statePath)) {
      this.state = JSON.parse(readFileSync(statePath, 'utf8'));
    } else {
      const base = readConfig(baseConfigPath);
      const incumbent: Values = {};
      for (const p of PARAMS) {
        const v = getPath(base, p.path);
        incumbent[p.path] = typeof v === 'number' ? v : fromUnit(p, 0.5);
      }
      this.state = { incumbent, incumbentScores: [], candidate: null, candidateScores: [], sigma: 0.15, history: [] };
    }
  }

  // Values for the next game: the incumbent until it has a score, then a candidate.
  private next(): { values: Values; role: 'incumbent' | 'candidate' } {
    const s = this.state;
    if (s.incumbentScores.length === 0) return { values: s.incumbent, role: 'incumbent' };
    if (!s.candidate) {
      s.candidate = {};
      for (const p of PARAMS) s.candidate[p.path] = fromUnit(p, toUnit(p, s.incumbent[p.path]) + s.sigma * gaussian());
      s.candidateScores = [];
    }
    return { values: s.candidate, role: 'candidate' };
  }

  // Writes base config + this game's parameters to outPath and returns a description.
  prepare(outPath: string): string {
    const { values, role } = this.next();
    const cfg = readConfig(this.baseConfigPath);
    for (const [k, v] of Object.entries(values)) setPath(cfg, k, v);
    writeFileSync(outPath, JSON.stringify(cfg, null, 2));
    this.save();
    return `${role} (sigma ${this.state.sigma.toFixed(3)})`;
  }

  report(score: number): string {
    const s = this.state;
    const { values, role } = this.next();
    s.history.push({ time: new Date().toISOString(), params: { ...values }, score, role });
    let msg: string;
    if (role === 'incumbent') {
      s.incumbentScores.push(score);
      msg = `incumbent scored ${score.toFixed(3)}`;
    } else {
      s.candidateScores.push(score);
      msg = `candidate scored ${score.toFixed(3)} (${s.candidateScores.length}/${EVALS})`;
      if (s.candidateScores.length >= EVALS) {
        const cand = mean(s.candidateScores);
        const inc = mean(s.incumbentScores);
        if (cand > inc) {
          s.incumbent = s.candidate!;
          s.incumbentScores = [...s.candidateScores];
          s.sigma = Math.min(SIGMA_MAX, s.sigma * 1.5);
          msg += `; accepted (${cand.toFixed(3)} > ${inc.toFixed(3)})`;
        } else {
          s.sigma = Math.max(SIGMA_MIN, s.sigma * Math.pow(1.5, -0.25));
          msg += `; rejected (${cand.toFixed(3)} <= ${inc.toFixed(3)})`;
        }
        s.candidate = null;
        s.candidateScores = [];
      }
    }
    this.save();
    return msg;
  }

  private save(): void {
    writeFileSync(this.statePath, JSON.stringify(this.state, null, 2));
  }
}
