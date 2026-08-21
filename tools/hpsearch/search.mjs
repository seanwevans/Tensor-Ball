// Runs a list of candidate configs as headless trials and ranks them.
//
//   node search.mjs --stage screen --batches 12 --seeds 1 --concurrency 3 \
//                   --out results/screen
//
// Trials are independent processes-worth of work inside one node process (each
// gets its own browser), run --concurrency at a time. Every finished trial is
// written out immediately, so a stage that dies partway through still leaves
// usable results behind.
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { runTrial } from "./trial.mjs";
import { stages } from "./space.mjs";
import { summarize, rank, table } from "./score.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));

async function pool(items, concurrency, worker) {
  const queue = items.slice();
  const running = new Set();
  const done = [];
  while (queue.length || running.size) {
    while (running.size < concurrency && queue.length) {
      const item = queue.shift();
      const p = worker(item).then(
        (v) => done.push(v),
        (e) => done.push({ ...item, failed: String(e && e.stack ? e.stack : e) })
      );
      running.add(p);
      p.finally(() => running.delete(p));
    }
    if (running.size) await Promise.race(running);
  }
  return done;
}

async function main() {
  const argv = process.argv.slice(2);
  const opts = { stage: null, batches: 12, minutes: 0, seeds: [1], concurrency: 3, out: null, only: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--stage") opts.stage = argv[++i];
    else if (a === "--batches") opts.batches = Number(argv[++i]);
    else if (a === "--minutes") opts.minutes = Number(argv[++i]);
    else if (a === "--seeds") opts.seeds = argv[++i].split(",").map(Number);
    else if (a === "--concurrency") opts.concurrency = Number(argv[++i]);
    else if (a === "--out") opts.out = argv[++i];
    else if (a === "--only") opts.only = argv[++i].split(",");
    else throw new Error(`unknown argument ${a}`);
  }
  let candidates = stages[opts.stage];
  if (!candidates) throw new Error(`unknown stage "${opts.stage}"; have ${Object.keys(stages).join(", ")}`);
  if (opts.only) candidates = candidates.filter((c) => opts.only.includes(c.name));

  const outDir = join(HERE, opts.out || `results/${opts.stage}`);
  await mkdir(outDir, { recursive: true });

  const jobs = [];
  for (const c of candidates)
    for (const seed of opts.seeds) jobs.push({ name: c.name, config: c.set, seed });

  const started = Date.now();
  console.log(
    `stage ${opts.stage}: ${candidates.length} configs x ${opts.seeds.length} seed(s) ` +
      `x ${opts.batches} batches, ${opts.concurrency} at a time -> ${outDir}`
  );

  const results = await pool(jobs, opts.concurrency, async (job) => {
    const file = join(outDir, `${job.name}.s${job.seed}.json`);
    if (existsSync(file)) {
      console.log(`  skip ${job.name} seed ${job.seed} (already done)`);
      return JSON.parse(await readFile(file, "utf8"));
    }
    const t = Date.now();
    const result = await runTrial({
      config: job.config,
      seed: job.seed,
      batches: opts.batches,
      budgetMs: opts.minutes * 60000
    });
    const record = { name: job.name, ...result };
    await writeFile(file, JSON.stringify(record, null, 2));
    const s = summarize(record);
    console.log(
      `  done ${job.name.padEnd(22)} seed ${job.seed}  ` +
        `final ${(s.finalAcc * 100).toFixed(2)}%  auc ${(s.auc * 100).toFixed(2)}%  ` +
        `score ${(s.score * 100).toFixed(2)}  (${((Date.now() - t) / 60000).toFixed(1)} min)`
    );
    return record;
  });

  const ranked = rank(results);
  await writeFile(join(outDir, "_ranked.json"), JSON.stringify(ranked, null, 2));
  console.log(`\n${table(ranked)}`);
  console.log(`\nstage wall time ${((Date.now() - started) / 60000).toFixed(1)} min`);
}

await main();
