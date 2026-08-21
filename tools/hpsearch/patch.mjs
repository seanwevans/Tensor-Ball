// Turns the app's script.js into a harness-drivable build, without touching the
// shipped file. Two seams are opened:
//
//   1. right after the CONFIG literal, so a trial can override any knob before
//      anything reads it (VisionSystem sizes its atlas in the constructor, the
//      agent builds its net from CONFIG, etc);
//   2. at the bootstrap line, so the harness can patch prototypes (telemetry,
//      headless speedups) before App is constructed, and keep a handle on the
//      running app.
//
// Everything else is byte-for-byte the app, which is the point: the search is
// only worth anything if the trials are the same environment the browser runs.
import { readFile } from "node:fs/promises";

const CONFIG_ANCHOR = `const UP = new THREE.Vector3(0, 1, 0);`;
const BOOT_ANCHOR = `new App().start();`;

const CONFIG_HOOK = `
// --- injected by tools/hpsearch/patch.mjs ---
if (typeof window !== "undefined" && window.__HP_CONFIG) {
  const overlay = (dst, src) => {
    for (const k of Object.keys(src)) {
      const v = src[k];
      if (v && typeof v === "object" && !Array.isArray(v) && dst[k] && typeof dst[k] === "object")
        overlay(dst[k], v);
      else dst[k] = v;
    }
  };
  overlay(CONFIG, window.__HP_CONFIG);
}
// --- end injection ---

`;

const BOOT_HOOK = `
// --- injected by tools/hpsearch/patch.mjs ---
if (typeof window !== "undefined" && window.__hpInstall) {
  window.__hpInstall({
    CONFIG, App, TrainingArena, CNNAgent, SceneManager, Dashboard,
    VisionSystem, TrajectoryTrails, Ball, tf
  });
}
window.__APP = new App();
window.__APP.start();
// --- end injection ---
`;

export async function patchedScript(path) {
  const src = await readFile(path, "utf8");
  if (!src.includes(CONFIG_ANCHOR))
    throw new Error(`patch.mjs: CONFIG anchor not found in ${path}`);
  if (!src.includes(BOOT_ANCHOR))
    throw new Error(`patch.mjs: bootstrap anchor not found in ${path}`);
  return src
    .replace(CONFIG_ANCHOR, CONFIG_HOOK + CONFIG_ANCHOR)
    .replace(BOOT_ANCHOR, BOOT_HOOK);
}
