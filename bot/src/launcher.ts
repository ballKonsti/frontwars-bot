import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { chromium, firefox, type Page, type BrowserContext } from 'playwright';

const here = path.dirname(fileURLToPath(import.meta.url));
const AGENT_SRC = readFileSync(path.join(here, 'page-agent.js'), 'utf8');

const CHROME_CANDIDATES = ['/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/google-chrome'];

export type BrowserName = 'firefox' | 'chrome';

export interface LaunchOptions {
  browser: BrowserName;
  headless: boolean;
  profileDir: string;
}

export async function launch(opts: LaunchOptions): Promise<{ context: BrowserContext; page: Page }> {
  // Persistent profile so the player name / settings stick between runs.
  // viewport: null lets the page use the real window size instead of an emulated one.
  const context =
    opts.browser === 'firefox'
      ? await firefox.launchPersistentContext(opts.profileDir, { headless: opts.headless, viewport: null })
      : await chromium.launchPersistentContext(opts.profileDir, {
          headless: opts.headless,
          executablePath: CHROME_CANDIDATES.find((p) => existsSync(p)),
          viewport: null,
          args: ['--start-maximized'],
        });
  await context.addInitScript({ content: AGENT_SRC });
  const page = context.pages()[0] ?? (await context.newPage());
  registerPopupHandlers(page);
  await page.goto('https://frontwars.io/', { waitUntil: 'domcontentloaded' });
  return { context, page };
}

// Another window in the same profile, for playing several single-player games at once.
export async function newGamePage(context: BrowserContext): Promise<Page> {
  const page = await context.newPage();
  registerPopupHandlers(page);
  return page;
}

// Menu modals (shop and trailer promos, first-run tutorial offer) block clicks; Playwright dismisses them whenever they appear.
function registerPopupHandlers(page: Page): void {
  const dismiss: Array<[string, RegExp]> = [
    ['New here, Commander?', /^not now$/i],
    ['The Shop is open!', /^close$/i],
    ['The official trailer is out!', /^close$/i],
  ];
  for (const [title, button] of dismiss) {
    const modal = page.locator('div.fixed.inset-0', { hasText: title });
    void page.addLocatorHandler(modal, async () => {
      await modal.getByRole('button', { name: button }).first().click();
    });
  }
}

// The site may show a cookie-consent banner and a Cloudflare "Quick check" before a game starts.
// Those are for the person at the keyboard: we never click them, we just wait while they do.
const HUMAN_TIMEOUT = 5 * 60_000;

export async function startSinglePlayer(page: Page): Promise<void> {
  console.log('Opening single player. If a consent banner or "Quick check" shows up, handle it in the browser window.');
  await page.getByRole('button', { name: 'Single Player' }).click({ timeout: HUMAN_TIMEOUT });
  await page.getByRole('button', { name: /start game/i }).click({ timeout: HUMAN_TIMEOUT });
}

export async function waitForGame(page: Page, timeoutMs: number): Promise<void> {
  await page.waitForFunction(() => (window as any).__fwbot?.ready(), undefined, {
    timeout: timeoutMs,
    polling: 500,
  });
}
