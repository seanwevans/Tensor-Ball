// The solver's front door. Two modes, one for each direction of the problem.
//
//   node solve.mjs --mode reverse --n 2000 --out made.jsonl
//
//     Start balls in made positions and run the physics backwards until they
//     are in a shooter's hands. Every line out is a shot the agent could have
//     taken and made, with the action that makes it — derived, not searched.
//
//   node solve.mjs --mode target --n 200 --out oracle.jsonl
//
//     Draw spawns the way Ball.spawn draws them and solve each one: the exact
//     action, how many arcs from that spot go in at all, and how far each
//     channel can drift before the ball stops dropping.
//
//   node solve.mjs --mode target --grid 24 --out grid.jsonl
//
//     The same, on a grid over the half court rather than a sample of it.
//
// --verify re-flies every answer in cannon-es with the rings, the boards and
// the posts in the world (see cannoncheck.mjs), and the summary reports what
// that found. It is the only claim here that does not come from this folder's
// own arithmetic, so it is worth the seconds it costs.
//
// --set takes the same dotted CONFIG paths tools/hpsearch does
// (`--set launch.fwdMax=34`), so an oracle can be built for the config a search
// trial ran under rather than only for the shipped one.
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { makeSolver, parseCommon, interval } from "./harness.mjs";

function usage() {
  console.error(`usage: node solve.mjs [options]

  --mode reverse|target   which direction to solve (default reverse)
  --n N                   shots to produce (default 500)
  --seed N                PRNG seed (default 1)
  --radius FT             spawn disc to draw from (default CONFIG.curriculum.maxRadius)
  --grid N                target mode: an N x N grid over the half court instead of a sample
  --height FT             target mode: fix the shooter's eye height instead of drawing it
  --min-gap FT            demand this much air between ball and ring (default 0)
  --up-steps N            target mode: launch angles tried per spin (default 48)
  --spin-grid N           target mode: spin values tried (default 5, odd values include 0)
  --verify                re-fly every answer in cannon-es
  --out FILE              write JSONL here (summary still goes to stderr)
  --set path=value        override a CONFIG knob before solving
  --script FILE           solve a different script.js`);
}

async function main() {
  const argv = process.argv.slice(2);
  const opts = { overlay: {}, seed: 1, scriptPath: undefined, out: null };
  const rest = parseCommon(argv, opts);

  let mode = "reverse";
  let n = 500;
  let radius = null;
  let grid = 0;
  let height = null;
  let minGap = 0;
  let upSteps = 48;
  let spinGrid = 5;
  let verify = false;
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    if (a === "--mode") mode = rest[++i];
    else if (a === "--n") n = Number(rest[++i]);
    else if (a === "--radius") radius = Number(rest[++i]);
    else if (a === "--grid") grid = Number(rest[++i]);
    else if (a === "--height") height = Number(rest[++i]);
    else if (a === "--min-gap") minGap = Number(rest[++i]);
    else if (a === "--up-steps") upSteps = Number(rest[++i]);
    else if (a === "--spin-grid") spinGrid = Number(rest[++i]);
    else if (a === "--verify") verify = true;
    else if (a === "--help" || a === "-h") return usage();
    else throw new Error(`unknown argument ${a}`);
  }
  if (mode !== "reverse" && mode !== "target")
    throw new Error(`--mode must be reverse or target, got "${mode}"`);

  const S = await makeSolver({ ...opts, withCannon: verify });
  const { app, reverse, target, cannon, rand, sampleSpawn, zoneOf } = S;
  const maxRadius = radius ?? app.CONFIG.curriculum.maxRadius;
  const spinValues = Array.from({ length: Math.max(1, spinGrid) }, (_, i) =>
    spinGrid <= 1 ? 0 : -1 + (2 * i) / (spinGrid - 1)
  );

  const lines = [];
  const zones = new Map();
  const bump = (zone) => {
    let z = zones.get(zone);
    if (!z) {
      z = mode === "target"
        ? { n: 0, reachable: 0, verified: 0, clean: 0, tolFwd: 0, arcs: 0 }
        : { n: 0, verified: 0, clean: 0 };
      zones.set(zone, z);
    }
    z.n++;
    return z;
  };

  let attempts = 0;
  let produced = 0;
  let verified = 0;
  let cleanSwish = 0;
  let worstClosure = 0;

  if (mode === "reverse") {
    // Rejection sampling: entries whose shooter ends up off the court, or whose
    // launch lands outside CONFIG.launch's envelope, are not shots the app could
    // have produced. The ones that survive are exactly the ones it could.
    const cap = n * 200;
    while (produced < n && attempts < cap) {
      attempts++;
      const r = reverse.solve({ rand, maxSpawnRadius: maxRadius });
      if (!r) continue;
      if (r.verified.minGap < minGap) continue;
      produced++;
      worstClosure = Math.max(worstClosure, r.verified.closure);
      const zone = zoneOf(r.spawn);
      const z = bump(zone);
      if (verify) {
        const v = cannon.fly(r.action, r.spawn);
        if (v.scored) {
          verified++;
          z.verified++;
        }
        if (v.clean) {
          cleanSwish++;
          z.clean++;
        }
        r.cannon = v;
      }
      r.zone = zone;
      lines.push(r);
      if (produced % 250 === 0)
        process.stderr.write(`reverse: ${produced}/${n} (${attempts} entries tried)\n`);
    }
  } else {
    const spawns = [];
    if (grid > 0) {
      // A grid over the spawn disc, clipped to the court, at a fixed height.
      const rim = app.CONFIG.rim;
      const y = height ?? app.CONFIG.spawnHeight.mean;
      for (let i = 0; i < grid; i++)
        for (let j = 0; j < grid; j++) {
          const x = (rim.x * (i + 0.5)) / grid;
          const z = -app.GEOM.spawnBounds.maxAbsZ + (2 * app.GEOM.spawnBounds.maxAbsZ * (j + 0.5)) / grid;
          const d = Math.hypot(x - rim.x, z - rim.z);
          if (d < app.CONFIG.minSpawnDistance || d > maxRadius) continue;
          spawns.push({ x, y, z });
        }
    } else {
      for (let i = 0; i < n; i++) {
        const sp = sampleSpawn(maxRadius);
        if (height !== null) sp.y = height;
        spawns.push(sp);
      }
    }

    for (let i = 0; i < spawns.length; i++) {
      const r = target.solve(spawns[i], { upSteps, spinValues, minGap });
      produced++;
      const zone = zoneOf(spawns[i]);
      const z = bump(zone);
      r.zone = zone;
      if (r.reachable) {
        z.reachable++;
        z.tolFwd += Math.min(r.tolerance.fwd.plus, r.tolerance.fwd.minus);
        z.arcs += r.arcs;
        if (verify) {
          const v = cannon.fly(r.action, r.spawn);
          if (v.scored) {
            verified++;
            z.verified++;
          }
          if (v.clean) {
            cleanSwish++;
            z.clean++;
          }
          r.cannon = v;
        }
      }
      lines.push(r);
      if ((i + 1) % 25 === 0)
        process.stderr.write(`target: ${i + 1}/${spawns.length}\n`);
    }
  }

  const reachable = lines.filter((l) => l.reachable !== false).length;
  const summary = {
    mode,
    seed: opts.seed,
    script: app.scriptPath,
    shots: lines.length,
    spawnRadius: maxRadius,
    minGap,
    reachable,
    // Reverse mode's round trip: how far the forward flight's crossing landed
    // from the made position the reverse pass started at, in feet. It is the
    // whole claim of this folder in one number.
    worstClosure: mode === "reverse" ? worstClosure : null,
    entriesTried: mode === "reverse" ? attempts : null,
    verified: verify ? { scored: verified, cleanSwish: cleanSwish, of: reachable } : null,
    zones: Object.fromEntries(
      [...zones].map(([k, z]) => [
        k,
        mode === "target"
          ? {
              ...z,
              reachRate: z.n ? z.reachable / z.n : null,
              meanTolFwd: z.reachable ? z.tolFwd / z.reachable : null
            }
          : z
      ])
    )
  };

  if (opts.out) {
    await writeFile(
      opts.out,
      lines.map((l) => JSON.stringify(l)).join("\n") + "\n"
    );
    process.stderr.write(`wrote ${lines.length} shots to ${opts.out}\n`);
  }

  // The table, to stderr, so --out's JSONL can be piped without it.
  const pct = (v) => (v == null ? "—" : (100 * v).toFixed(1) + "%");
  process.stderr.write(`\n=== ${mode} — ${lines.length} shots, spawn radius ${maxRadius.toFixed(1)}ft ===\n\n`);
  if (mode === "reverse") {
    process.stderr.write(
      `${attempts} made positions tried, ${produced} survived the app's own gates ` +
        `(${pct(produced / attempts)})\n` +
        `round-trip closure, worst of ${produced}: ${worstClosure.toExponential(2)} ft\n`
    );
    if (verify)
      process.stderr.write(
        `cannon-es: ${verified}/${produced} scored, ${cleanSwish}/${produced} clean swishes\n`
      );
    process.stderr.write("\nzone              shots\n");
    for (const zone of app.SHOT_ZONES) {
      const z = zones.get(zone.key);
      if (!z) continue;
      process.stderr.write(zone.label.padEnd(18) + String(z.n).padStart(6) + "\n");
    }
  } else {
    process.stderr.write("zone              spawns  reachable          tol(fwd)   arcs\n");
    for (const zone of app.SHOT_ZONES) {
      const z = zones.get(zone.key);
      if (!z) continue;
      const ci = interval(z.reachable, z.n);
      process.stderr.write(
        zone.label.padEnd(18) +
          String(z.n).padStart(6) +
          pct(ci.p).padStart(11) +
          ` [${pct(ci.lo)}, ${pct(ci.hi)}]`.padEnd(20) +
          (z.reachable ? "±" + (z.tolFwd / z.reachable).toFixed(4) : "—").padStart(9) +
          (z.reachable ? (z.arcs / z.reachable).toFixed(0) : "—").padStart(7) +
          "\n"
      );
    }
    if (verify)
      process.stderr.write(
        `\ncannon-es: ${verified}/${reachable} scored, ${cleanSwish}/${reachable} clean swishes\n`
      );
  }

  process.stdout.write(JSON.stringify(summary, null, 2) + "\n");
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
