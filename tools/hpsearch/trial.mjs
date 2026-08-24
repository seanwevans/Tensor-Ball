// One training run of the real app, headless, with a config overlay applied.
//
//   node trial.mjs --batches 20 --seed 1 --set learningRate=0.003 \
//                  --set exploreNoise=0.9 --out result.json
//
// --batches caps the run by batch count, --minutes caps it by wall clock;
// give both and whichever comes first ends the run.
//
// Prints a JSON result to stdout (and to --out): the config that ran, the
// backend that served it, and one row of metrics per trained batch.
import { chromium } from "/opt/node22/lib/node_modules/playwright/index.mjs";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { patchedScript } from "./patch.mjs";
import { startSite } from "./site.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");

const CHROME_ARGS = [
  "--use-gl=angle",
  "--use-angle=swiftshader",
  "--enable-unsafe-swiftshader",
  "--disable-gpu-vsync",
  "--disable-frame-rate-limit",
  "--no-sandbox",
  "--js-flags=--max-old-space-size=4096"
];

// "reward.rim=3" -> { reward: { rim: 3 } }, merged into the overlay.
export function applySet(overlay, assignment) {
  const eq = assignment.indexOf("=");
  if (eq < 0) throw new Error(`--set expects key=value, got "${assignment}"`);
  const path = assignment.slice(0, eq).split(".");
  const raw = assignment.slice(eq + 1);
  const value = raw === "true" ? true : raw === "false" ? false : Number(raw);
  if (typeof value === "number" && Number.isNaN(value))
    throw new Error(`--set value for ${path.join(".")} is not a number: "${raw}"`);
  let node = overlay;
  for (const key of path.slice(0, -1)) node = node[key] ??= {};
  node[path.at(-1)] = value;
  return overlay;
}

export async function runTrial({
  config = {},
  seed = 1,
  batches = 20,
  budgetMs = 0,
  backend = null,
  timeoutMs = 45 * 60 * 1000,
  onBatch = null
} = {}) {
  const source = await patchedScript(join(ROOT, "script.js"));
  const hooks = await readFile(join(HERE, "pagehooks.js"), "utf8");
  const site = await startSite(source);
  const browser = await chromium.launch({ headless: true, args: CHROME_ARGS });
  const started = Date.now();
  try {
    const page = await browser.newPage({ viewport: { width: 640, height: 480 } });
    const errors = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (m) => {
      if (m.type() === "error") errors.push(m.text());
    });

    await page.addInitScript({
      content: `window.__HP_PARAMS = ${JSON.stringify({ config, seed, batches, budgetMs, backend })};\n${hooks}`
    });
    await page.goto(site.origin, { waitUntil: "load", timeout: 120000 });
    // start() is async (backend selection, model build); the arena is the last
    // thing it makes before the loop starts.
    await page.waitForFunction("window.__APP && window.__APP.arena", null, { timeout: 180000 });
    await page.evaluate("window.__hpStart()");

    let seen = 0;
    for (;;) {
      const state = await page.evaluate(
        `({ done: window.__HP.done, error: window.__HP.error, backend: window.tf && window.tf.getBackend(),
            rows: window.__HP.batches.slice(${seen}) })`
      );
      for (const row of state.rows) {
        seen++;
        if (onBatch) onBatch(row);
      }
      if (state.error) throw new Error(`page: ${state.error}`);
      if (state.done) {
        const backendUsed = state.backend;
        const rows = await page.evaluate("window.__HP.batches");
        // Carried out of the page rather than restated here: the shot chart's
        // buckets and the NBA rates they are read against live in script.js,
        // and a second copy in the harness is a second thing to keep in step.
        const zoneMeta = await page.evaluate("window.__HP.zoneMeta");
        return {
          config,
          seed,
          batches,
          budgetMs,
          backend: backendUsed,
          wallMs: Date.now() - started,
          errors: errors.slice(0, 5),
          zoneMeta,
          rows
        };
      }
      if (Date.now() - started > timeoutMs)
        throw new Error(`trial timed out after ${Math.round((Date.now() - started) / 1000)}s at batch ${seen}`);
      await new Promise((r) => setTimeout(r, 250));
    }
  } finally {
    await browser.close();
    site.close();
  }
}

async function main() {
  const argv = process.argv.slice(2);
  const opts = { config: {}, seed: 1, batches: 20, budgetMs: 0, backend: null };
  let out = null;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--set") applySet(opts.config, argv[++i]);
    else if (a === "--seed") opts.seed = Number(argv[++i]);
    else if (a === "--batches") opts.batches = Number(argv[++i]);
    else if (a === "--minutes") opts.budgetMs = Number(argv[++i]) * 60000;
    else if (a === "--backend") opts.backend = argv[++i];
    else if (a === "--out") out = argv[++i];
    else if (a === "--config") Object.assign(opts.config, JSON.parse(argv[++i]));
    else throw new Error(`unknown argument ${a}`);
  }
  const result = await runTrial({
    ...opts,
    onBatch: (r) =>
      process.stderr.write(
        `batch ${String(r.batch).padStart(3)}  acc ${(r.acc * 100).toFixed(2)}%  ` +
          `reward ${r.meanReward.toFixed(2)}  d10 ${r.distP10.toFixed(2)}  ` +
          `illegal ${(r.illegal * 100).toFixed(1)}%  loss ${r.loss == null ? "--" : r.loss.toFixed(2)}  ` +
          `sim ${(r.simMs / 1000).toFixed(1)}s  step ${(r.stepMs / 1000).toFixed(1)}s\n`
      )
  });
  const json = JSON.stringify(result, null, 2);
  if (out) await writeFile(out, json);
  process.stdout.write(json + "\n");
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
