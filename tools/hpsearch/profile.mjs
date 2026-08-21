// Where does a batch's wall time actually go? Times the stereo capture, the
// critic/actor update and the physics separately, at whatever scale you give
// it, so the search can be sized against real numbers instead of guesses.
//
//   node profile.mjs --set batchSize=256 --set visionWidth=96 --backend webgl
import { chromium } from "/opt/node22/lib/node_modules/playwright/index.mjs";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { patchedScript } from "./patch.mjs";
import { startSite } from "./site.mjs";
import { applySet } from "./trial.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");

const argv = process.argv.slice(2);
const config = {};
let backend = null;
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--set") applySet(config, argv[++i]);
  else if (argv[i] === "--backend") backend = argv[++i];
}

const source = await patchedScript(join(ROOT, "script.js"));
const hooks = await readFile(join(HERE, "pagehooks.js"), "utf8");
const site = await startSite(source);
const browser = await chromium.launch({
  headless: true,
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
         "--disable-gpu-vsync", "--disable-frame-rate-limit", "--no-sandbox"]
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("PAGEERROR", String(e)));
await page.addInitScript({
  content: `window.__HP_PARAMS = ${JSON.stringify({ config, seed: 1, batches: 1e9, backend })};\n${hooks}`
});
await page.goto(site.origin, { waitUntil: "load", timeout: 120000 });
await page.waitForFunction("window.__APP && window.__APP.arena", null, { timeout: 180000 });

const out = await page.evaluate(async () => {
  const app = window.__APP;
  const a = app.arena;
  const t = () => performance.now();
  const marks = {};

  a.spawnAll();
  // Two captures: the first pays for shader compiles and shadow maps.
  let t0 = t();
  a.vision.captureBatch({ balls: a.balls, ballField: a.field.mesh, hoopPos: a.hoopPos,
    court: a.court, manualMesh: app.manual.mesh, trajectoryGroup: a.trajectory.group });
  marks.captureFirstMs = t() - t0;
  t0 = t();
  const batch = a.vision.captureBatch({ balls: a.balls, ballField: a.field.mesh, hoopPos: a.hoopPos,
    court: a.court, manualMesh: app.manual.mesh, trajectoryGroup: a.trajectory.group });
  marks.captureMs = t() - t0;

  t0 = t();
  const actions = a.agent.predictBatch(batch, 0.4);
  marks.predictMs = t() - t0;

  const rewards = actions.map(() => Math.random() * 10 - 5);
  a.agent.store(batch, actions, rewards);
  t0 = t();
  await a.agent.train();
  marks.trainFirstMs = t() - t0;
  a.agent.store(batch, actions, rewards);
  t0 = t();
  await a.agent.train();
  marks.trainMs = t() - t0;

  // 100 physics steps with the whole batch in flight.
  for (let i = 0; i < a.balls.length; i++) a.balls[i].launch(actions[i], a.hoopPos);
  t0 = t();
  for (let i = 0; i < 100; i++) { app.physics.step(); a.update(); }
  marks.physics100Ms = t() - t0;

  marks.backend = window.tf.getBackend();
  marks.renders = 2 * a.balls.length;
  marks.batchSize = window.__APP.arena.balls.length;
  marks.vision = [a.vision.W, a.vision.H, a.vision.passes, a.vision.cols, a.vision.rows];
  return marks;
});
console.log(JSON.stringify(out, null, 2));
console.log("per-view render ms:", (out.captureMs / out.renders).toFixed(2));
await browser.close();
site.close();
