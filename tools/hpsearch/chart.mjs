// The shot chart behind a trial file: what the policy shoots from each of the
// league's zones, next to what the league shoots from it.
//
//   node chart.mjs results/nba/baseline.s1.json [more.json ...]
//   node chart.mjs --tail 0.25 run.json      # last quarter of the run only
//
// A run's headline accuracy is pooled over a spawn disc that reaches half
// court, so it is not a number any box score has a counterpart for and it
// moves whenever the curriculum opens the floor. Per zone it is comparable.
//
// Zones and benchmarks come out of the trial file (trial.mjs carries them
// across from the app's SHOT_ZONES), so this file holds no basketball of its
// own to fall out of step.
import { readFile } from "node:fs/promises";

const files = [];
let tail = 0.25;
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--tail") tail = Number(argv[++i]);
  else files.push(argv[i]);
}
if (!files.length) {
  console.error("usage: node chart.mjs [--tail 0.25] trial.json ...");
  process.exit(1);
}

const pct = (made, n) => (n ? ((100 * made) / n).toFixed(1) + "%" : "--");

for (const file of files) {
  const rec = JSON.parse(await readFile(file, "utf8"));
  const rows = rec.rows || [];
  const from = Math.max(0, Math.floor(rows.length * (1 - tail)));
  const window = rows.slice(from);
  const pooled = new Map();
  for (const row of window)
    for (const [key, z] of Object.entries(row.zones || {})) {
      const acc = pooled.get(key) || { n: 0, made: 0, evalN: 0, evalMade: 0 };
      acc.n += z.n;
      acc.made += z.made;
      acc.evalN += z.evalN;
      acc.evalMade += z.evalMade;
      pooled.set(key, acc);
    }

  const last = rows.at(-1);
  console.log(
    `\n=== ${file} — batches ${from + 1}-${rows.length}` +
      `${last ? `, spawn radius ${last.spawnRadius.toFixed(1)}ft` : ""} ===\n`
  );
  console.log(
    "zone            greedy      n   behaviour      n     NBA   elite"
  );
  let evalN = 0;
  let evalMade = 0;
  for (const zone of rec.zoneMeta || []) {
    const z = pooled.get(zone.key);
    if (!z) continue;
    evalN += z.evalN;
    evalMade += z.evalMade;
    console.log(
      zone.label.padEnd(15) +
        pct(z.evalMade, z.evalN).padStart(6) +
        String(z.evalN).padStart(7) +
        pct(z.made, z.n).padStart(12) +
        String(z.n).padStart(7) +
        (zone.league == null ? "—" : (zone.league * 100).toFixed(0) + "%").padStart(8) +
        (zone.elite == null ? "—" : (zone.elite * 100).toFixed(0) + "%").padStart(8)
    );
  }
  console.log(
    "\nall zones".padEnd(15) + pct(evalMade, evalN).padStart(6) + String(evalN).padStart(7)
  );
}
