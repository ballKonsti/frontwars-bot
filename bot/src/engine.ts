import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';

export interface EngineReply {
  actions: Array<Record<string, unknown>>;
  log?: string[];
}

export interface GameResult {
  won: boolean;
  eliminated: boolean;
}

// Line-delimited JSON over stdin/stdout to the C++ engine: one state in, one reply out.
// One engine per game, so each game can get its own config; the learner's weights live on disk.
export class Engine {
  private proc: ChildProcessWithoutNullStreams;
  private waiting: Array<(r: EngineReply) => void> = [];
  private exited: Promise<number | null>;
  private finishing = false;

  constructor(binary: string, configPath: string) {
    this.proc = spawn(binary, [configPath], { stdio: ['pipe', 'pipe', 'pipe'] });
    this.proc.stderr.on('data', (d) => process.stderr.write(`[engine] ${d}`));
    this.exited = new Promise((resolve) => {
      this.proc.on('exit', (code) => {
        if (!this.finishing) {
          console.error(`[engine] exited unexpectedly with code ${code}`);
          process.exit(1);
        }
        resolve(code);
      });
    });
    createInterface({ input: this.proc.stdout }).on('line', (line) => {
      const resolve = this.waiting.shift();
      if (!resolve) return;
      try {
        resolve(JSON.parse(line));
      } catch {
        resolve({ actions: [], log: [`bad engine output: ${line.slice(0, 200)}`] });
      }
    });
  }

  decide(state: unknown): Promise<EngineReply> {
    return new Promise((resolve) => {
      this.waiting.push(resolve);
      this.proc.stdin.write(JSON.stringify(state) + '\n');
    });
  }

  // Tells the learner how the game ended, then waits for the engine to save and exit.
  async finish(result: GameResult): Promise<void> {
    this.finishing = true;
    this.proc.stdin.end(JSON.stringify({ gameOver: true, ...result }) + '\n');
    await this.exited;
  }
}
