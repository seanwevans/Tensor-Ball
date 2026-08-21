// Reads a stage's trial files and prints its ranking plus the accuracy curve
// behind each row, so a result can be read without opening the JSON.
//
//   node report.mjs results/screen [results/combo ...]
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { rank, table, summarize } from "./score.mjs";

async function load(dir) {
  const files = (await readdir(dir)).filter((f) => f.endsWith(".json") && !f.startsWith("_"));
  return Promise.all(files.map(async (f) => JSON.parse(await readFile(join(dir, f), "utf8"))));
}

for (const dir of process.argv.slice(2)) {
  const records = await load(dir);
  const ranked = rank(records);
  console.log(`\n=== ${dir} — ${records.length} trials ===\n`);
  console.log(table(ranked));
  console.log("\naccuracy by batch (%)\n");
  const byName = new Map();
  for (const r of records) {
    if (!byName.has(r.name)) byName.set(r.name, []);
    byName.get(r.name).push(r);
  }
  for (const row of ranked) {
    for (const rec of byName.get(row.name) || []) {
      const curve = (rec.rows || []).map((b) => (b.acc * 100).toFixed(1).padStart(4)).join(" ");
      console.log(`${(rec.name + " s" + rec.seed).padEnd(26)} ${curve}`);
    }
  }
  console.log("\nconfig overlays\n");
  for (const row of ranked) {
    const rec = (byName.get(row.name) || [])[0];
    console.log(`${row.name.padEnd(26)} ${JSON.stringify(rec ? rec.config : {})}`);
  }
}
