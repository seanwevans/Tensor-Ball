// The solver's front door.
//
//   node solve.mjs --n 2000 --out made.jsonl
//
// Start balls in made positions and run the physics backwards until they are in
// a shooter's hands. Every line out is a shot the agent could have taken and
// made, with the action that makes it — derived, not searched.
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
import { makeSolver, parseCommon } from "./harness.mjs";

function usage() {
  console.error(`usage: node solve.mjs [options]

  --n N                   shots to produce (default 500)
  --seed N                PRNG seed (default 1)
  --radius FT             spawn disc to draw from (default CONFIG.curriculum.maxRadius)
  --min-gap FT            demand this much air between ball and ring (default 0)
  --verify                re-fly every answer in cannon-es
  --out FILE              write JSONL here (summary still goes to stderr)
  --set path=value        override a CONFIG knob before solving
  --script FILE           solve a different script.js`);
}

async function main() {
  const argv = process.argv.slice(2);
  const opts = { overlay: {}, seed: 1, scriptPath: undefined, out: null };
  const rest = parseCommon(argv, opts);

  const mode = "reverse";
  let n = 500;
  let radius = null;
  let minGap = 0;
  let verify = false;
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    if (a === "--n") n = Number(rest[++i]);
    else if (a === "--radius") radius = Number(rest[++i]);
    else if (a === "--min-gap") minGap = Number(rest[++i]);
    else if (a === "--verify") verify = true;
    else if (a === "--help" || a === "-h") return usage();
    else throw new Error(`unknown argument ${a}`);
  }
  const S = await makeSolver({ ...opts, withCannon: verify });
  const { app, reverse, cannon, rand, zoneOf } = S;
  const maxRadius = radius ?? app.CONFIG.curriculum.maxRadius;

  const lines = [];
  const zones = new Map();
  const bump = (zone) => {
    let z = zones.get(zone);
    if (!z) {
      z = { n: 0, verified: 0, clean: 0 };
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
    worstClosure,
    entriesTried: attempts,
    verified: verify ? { scored: verified, cleanSwish: cleanSwish, of: reachable } : null,
    zones: Object.fromEntries(zones)
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
  process.stderr.write(`\n=== reverse — ${lines.length} shots, spawn radius ${maxRadius.toFixed(1)}ft ===\n\n`);
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

  process.stdout.write(JSON.stringify(summary, null, 2) + "\n");
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
