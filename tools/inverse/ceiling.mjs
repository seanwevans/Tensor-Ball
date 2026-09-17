// What the best possible policy would shoot, zone by zone.
//
//   node ceiling.mjs --sigma 0.02,0.05,0.15,0.25 --n 300
//
// The dashboard reports what the agent hits. This reports what there is to hit,
// which is the other half of the sentence and the only thing that makes the
// first half readable: an agent at 40% from mid-range is doing well or badly
// depending entirely on whether 45% or 95% was available, and nothing in a
// training run answers that — the run only ever sees the actions its own policy
// proposed.
//
// The answer here is built from the exact solutions, so it owes nothing to any
// training run:
//
//   1. Draw spawns the way Ball.spawn draws them.
//   2. Solve each one exactly (target.mjs): the whole family of arcs that go in,
//      and the action that flies each.
//   3. Hand the best few to a policy that has learned that exact answer and
//      explores around it the way CNNAgent.predictBatch does — action =
//      clamp(mean + sigma * gauss()) per channel — and count what drops.
//
// Step 3 is why sigma is the argument. The policy's spread is something it
// learns rather than something a schedule imposes (CONFIG.policy), so the
// interesting question is not "what is the ceiling" but "what is the ceiling at
// the spread the policy has actually settled on". Read the Policy Sigma off the
// dashboard, pass it here, and the number that comes back is what the run could
// be shooting if its mean action were perfect. exp(CONFIG.policy.logStdMin) and
// exp(logStdInit) are the ends of that range: 0.02 and 0.25.
//
// Two engines. `free` flies the arc and counts the shots that drop through
// clean, which is fast and is a lower bound on the app's own make rate because
// it gives up nothing to a friendly bounce. `cannon` re-flies every sample in
// cannon-es with the rings and boards present, so a shot that rattles in counts
// the way the app counts it — about thirty times slower and the number the
// dashboard is comparable to.
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { makeSolver, parseCommon } from "./harness.mjs";
import { gauss } from "./rng.mjs";
import { state } from "./physics.mjs";
import { actionToLaunch, launchToAction } from "./court.mjs";

function usage() {
  console.error(`usage: node ceiling.mjs [options]

  --sigma A,B,C     exploration spreads to report, in action units (default 0,0.02,0.05,0.15,0.25)
  --n N             spawns to draw (default 300)
  --samples N       shots per spawn per sigma (default 256)
  --candidates N    arcs from each spawn's family to try (default 6)
  --engine free|cannon   free flight (clean swishes) or the real world (default free)
  --radius FT       spawn disc to draw from (default CONFIG.curriculum.maxRadius)
  --up-steps N      launch angles tried per spin when solving (default 32)
  --spin-grid N     spin values tried when solving (default 3)
  --seed N          PRNG seed (default 1)
  --out FILE        write the full result as JSON
  --set path=value  override a CONFIG knob before solving`);
}

async function main() {
  const argv = process.argv.slice(2);
  const opts = { overlay: {}, seed: 1, scriptPath: undefined, out: null };
  const rest = parseCommon(argv, opts);

  let sigmas = [0, 0.02, 0.05, 0.15, 0.25];
  let n = 300;
  let samples = 256;
  let candidates = 6;
  let engine = "free";
  let radius = null;
  let upSteps = 32;
  let spinGrid = 3;
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    if (a === "--sigma") sigmas = rest[++i].split(",").map(Number);
    else if (a === "--n") n = Number(rest[++i]);
    else if (a === "--samples") samples = Number(rest[++i]);
    else if (a === "--candidates") candidates = Number(rest[++i]);
    else if (a === "--engine") engine = rest[++i];
    else if (a === "--radius") radius = Number(rest[++i]);
    else if (a === "--up-steps") upSteps = Number(rest[++i]);
    else if (a === "--spin-grid") spinGrid = Number(rest[++i]);
    else if (a === "--help" || a === "-h") return usage();
    else throw new Error(`unknown argument ${a}`);
  }
  if (engine !== "free" && engine !== "cannon")
    throw new Error(`--engine must be free or cannon, got "${engine}"`);

  const S = await makeSolver({ ...opts, withCannon: engine === "cannon" });
  const { app, target, flight, cannon, rand, sampleSpawn, zoneOf } = S;
  const L = app.CONFIG.launch;
  const maxRadius = radius ?? app.CONFIG.curriculum.maxRadius;
  const spinValues = Array.from({ length: Math.max(1, spinGrid) }, (_, i) =>
    spinGrid <= 1 ? 0 : -1 + (2 * i) / (spinGrid - 1)
  );

  const s = state(0, 0, 0, 0, 0, 0);
  const clamp = (v) => Math.max(-1, Math.min(1, v));

  // One shot from a policy whose mean is `action` and whose spread is `sigma`,
  // sampled the way CNNAgent.predictBatch samples: additive Gaussian noise per
  // channel, clamped to the envelope.
  const drawn = new Array(4);
  function attempt(spawn, action, sigma, frame) {
    for (let k = 0; k < 4; k++) drawn[k] = clamp(action[k] + sigma * gauss(rand));
    if (engine === "cannon") return cannon.fly(drawn, spawn).scored;
    const l = actionToLaunch(drawn, spawn.x, spawn.z, app.CONFIG.rim, L, frame);
    s.px = spawn.x;
    s.py = spawn.y;
    s.pz = spawn.z;
    s.vx = l.vx;
    s.vy = l.vy;
    s.vz = l.vz;
    s.wx = l.wx;
    s.wy = l.wy;
    s.wz = l.wz;
    const r = flight.simulate(s);
    return r.scored && !r.contact;
  }

  function rateOf(spawn, action, sigma, frame, m) {
    if (sigma === 0) return attempt(spawn, action, 0, frame) ? 1 : 0;
    let k = 0;
    for (let i = 0; i < m; i++) if (attempt(spawn, action, sigma, frame)) k++;
    return k / m;
  }

  const rows = [];
  const started = Date.now();
  for (let i = 0; i < n; i++) {
    const spawn = sampleSpawn(maxRadius);
    const { frame, arcs } = target.family(spawn, { upSteps, spinValues });
    const zone = zoneOf(spawn);
    if (!arcs.length) {
      rows.push({ spawn, zone, distance: frame.distance, reachable: false, rates: sigmas.map(() => 0) });
      continue;
    }
    const ranked = arcs.slice().sort((a, b) => b.fwdHalf - a.fwdHalf);
    const pool = ranked.slice(0, Math.max(1, candidates)).map((arc) =>
      launchToAction(
        {
          vx: frame.fx * arc.vFwd,
          vy: arc.vUp,
          vz: frame.fz * arc.vFwd,
          wx: -frame.sxx * arc.spin,
          wy: 0,
          wz: -frame.szz * arc.spin
        },
        spawn.x,
        spawn.z,
        app.CONFIG.rim,
        L,
        frame
      ).action
    );

    const rates = [];
    const chosen = [];
    for (const sigma of sigmas) {
      // Pick the arc on one sample and score it on another. Taking the best of
      // several noisy estimates and then reporting that estimate would report
      // the luck as well as the arc — at 256 samples the winner's curse is
      // worth about a point, which is the same size as some of the differences
      // this is being used to read.
      let bestAction = pool[0];
      let bestPilot = -1;
      for (const action of pool) {
        const r = rateOf(spawn, action, sigma, frame, Math.max(32, samples >> 2));
        if (r > bestPilot) {
          bestPilot = r;
          bestAction = action;
        }
      }
      rates.push(rateOf(spawn, bestAction, sigma, frame, samples));
      chosen.push(bestAction);
    }
    rows.push({
      spawn,
      zone,
      distance: frame.distance,
      reachable: true,
      arcs: arcs.length,
      action: chosen[0],
      rates
    });
    if ((i + 1) % 25 === 0)
      process.stderr.write(
        `${i + 1}/${n} spawns  (${((Date.now() - started) / 1000).toFixed(0)}s)\n`
      );
  }

  // Pooled per zone. The interval is across spawns rather than across shots:
  // the shots from one spawn are not independent of each other — they share an
  // arc and a distance — so treating all n*samples of them as separate draws
  // would quote an interval several times tighter than the estimate deserves.
  function pool(filter) {
    const sel = rows.filter(filter);
    return sigmas.map((_, si) => {
      if (!sel.length) return { mean: null, se: null, n: 0 };
      let sum = 0;
      for (const r of sel) sum += r.rates[si];
      const mean = sum / sel.length;
      let v = 0;
      for (const r of sel) v += (r.rates[si] - mean) ** 2;
      return {
        mean,
        se: sel.length > 1 ? Math.sqrt(v / (sel.length * (sel.length - 1))) : null,
        n: sel.length
      };
    });
  }

  const result = {
    seed: opts.seed,
    script: app.scriptPath,
    engine,
    spawns: n,
    samples,
    candidates,
    spawnRadius: maxRadius,
    sigmas,
    overall: pool(() => true),
    zones: Object.fromEntries(
      app.SHOT_ZONES.map((z) => [z.key, { label: z.label, league: z.league, elite: z.elite, ceiling: pool((r) => r.zone === z.key) }])
    )
  };
  if (opts.out) {
    await writeFile(opts.out, JSON.stringify({ ...result, rows }, null, 2));
    process.stderr.write(`wrote ${opts.out}\n`);
  }

  const cell = (c) => (c.n ? (100 * c.mean).toFixed(1).padStart(6) + "%" : "     —");
  process.stderr.write(
    `\n=== achievable make rate, ${engine} engine — ${n} spawns x ${samples} shots, radius ${maxRadius.toFixed(1)}ft ===\n\n`
  );
  process.stderr.write(
    "zone              spawns" + sigmas.map((g) => `  sigma ${g}`.padStart(13)).join("") + "     NBA   elite\n"
  );
  for (const z of app.SHOT_ZONES) {
    const c = result.zones[z.key].ceiling;
    if (!c[0].n) continue;
    process.stderr.write(
      z.label.padEnd(18) +
        String(c[0].n).padStart(6) +
        c.map((x) => cell(x).padStart(13)).join("") +
        (z.league == null ? "—" : (100 * z.league).toFixed(0) + "%").padStart(8) +
        (z.elite == null ? "—" : (100 * z.elite).toFixed(0) + "%").padStart(8) +
        "\n"
    );
  }
  process.stderr.write(
    "\n" +
      "all zones".padEnd(18) +
      String(n).padStart(6) +
      result.overall.map((x) => cell(x).padStart(13)).join("") +
      "\n"
  );
  process.stdout.write(JSON.stringify(result, null, 2) + "\n");
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
