// Wiring. Every tool in this folder wants the same six things built from the
// same script.js, and wants them refused for the same reasons.
import { loadApp, applySet, DEFAULT_SCRIPT } from "./appconfig.mjs";
import { makeStepper } from "./physics.mjs";
import { makeClearance } from "./court.mjs";
import { makeFlight } from "./flight.mjs";
import { makeCannonCourt } from "./cannoncheck.mjs";
import { mulberry32, gauss } from "./rng.mjs";

export async function makeSolver({
  scriptPath = DEFAULT_SCRIPT,
  overlay = {},
  seed = 1,
  withCannon = false
} = {}) {
  const app = await loadApp({ scriptPath, overlay });

  // Air drawn per shot has no exact inverse to find. sampleShotAir rolls the
  // drag, the lift and the breeze *after* the action is chosen, so the map from
  // action to outcome is not a function — the same action from the same spot
  // lands somewhere else next time, and "the impulse that would have made the
  // shot" stops being a thing there is one of. Everything here assumes the
  // gym's own air, which is the app's default.
  const A = app.CONFIG.air;
  if (A.wind || A.jitter.drag || A.jitter.magnus)
    throw new Error(
      "tools/inverse: CONFIG.air.wind / air.jitter make each shot's air a fresh draw, " +
        "so a shot has no exact inverse. Re-run with air.wind=0 and air.jitter.*=0 " +
        "to solve the gym this policy is being measured in."
    );

  const P = makeStepper(app);
  const clearance = makeClearance(app);
  const flight = makeFlight(app, P, clearance);
  const cannon = withCannon ? await makeCannonCourt(app) : null;
  if (withCannon && !cannon)
    throw new Error(
      "tools/inverse: cannon-es is not installed — run `npm install` in tools/inverse, " +
        "or drop the flag that asked for the cannon cross-check."
    );

  const rand = mulberry32(seed);

  // Ball.spawn's own draw: area-uniform over the annulus between
  // minSpawnDistance and the curriculum radius, rejected back into the court,
  // with Ball.spawn's fallback if no legal spot turns up.
  function sampleSpawn(maxRadius = app.CONFIG.curriculum.maxRadius) {
    const rim = app.CONFIG.rim;
    const min = app.CONFIG.minSpawnDistance;
    const outer = Math.max(min, maxRadius);
    const b = app.GEOM.spawnBounds;
    let x = 0;
    let z = 0;
    for (let attempt = 0; attempt < 24; attempt++) {
      const a = rand() * Math.PI * 2;
      const r = Math.sqrt(rand() * (outer * outer - min * min) + min * min);
      x = rim.x + Math.cos(a) * r;
      z = rim.z + Math.sin(a) * r;
      if (x >= 0 && x <= b.maxX && Math.abs(z) <= b.maxAbsZ) break;
      if (attempt === 23) {
        const d = Math.min(Math.max(min, Math.min(outer, rim.x)), rim.x);
        x = Math.max(0, rim.x - d);
        z = rim.z;
      }
    }
    return { x, y: eyeHeight(), z };
  }

  // spawnEyeHeight: a normal truncated to the ends of an NBA roster.
  function eyeHeight() {
    const S = app.CONFIG.spawnHeight;
    for (let attempt = 0; attempt < 8; attempt++) {
      const y = S.mean + gauss(rand) * S.stdDev;
      if (y >= S.min && y <= S.max) return y;
    }
    return Math.min(S.max, Math.max(S.min, S.mean));
  }

  const zoneOf = (spawn) => app.shotZone(spawn.x, spawn.z, app.CONFIG.rim);

  return { app, P, clearance, flight, cannon, rand, sampleSpawn, eyeHeight, zoneOf };
}

// Shared argument shapes, so `--set`, `--seed` and `--script` mean the same
// thing in every tool here and the same thing they mean in tools/hpsearch.
export function parseCommon(argv, opts) {
  const rest = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--set") applySet(opts.overlay, argv[++i]);
    else if (a === "--seed") opts.seed = Number(argv[++i]);
    else if (a === "--script") opts.scriptPath = argv[++i];
    else if (a === "--out") opts.out = argv[++i];
    else rest.push(a);
  }
  return rest;
}

// Wilson score interval, which behaves at 0 and 1 where the normal one does not.
export function interval(k, n, z = 1.96) {
  if (!n) return { p: null, lo: null, hi: null };
  const p = k / n;
  const d = 1 + (z * z) / n;
  const c = p + (z * z) / (2 * n);
  const s = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return { p, lo: Math.max(0, (c - s) / d), hi: Math.min(1, (c + s) / d) };
}
