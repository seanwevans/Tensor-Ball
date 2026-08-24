// "Most accurate, quickest" is two things, so the score is two things.
//
//   finalAcc — mean batch accuracy over the last quarter of the run: where the
//              policy actually got to.
//   auc      — mean accuracy across every batch of the run: a config that is at
//              3% by batch five scores above one that arrives at the same 3%
//              only at batch twenty, which is the whole point of the exercise.
//
// score is the mean of the two, so a config has to be both good and fast to win
// and neither half can be traded away entirely.
//
// The speed half is counted in batches rather than seconds deliberately. A
// batch costs the same wall time whatever the config — the critic/actor update
// dominates it and depends only on batchSize and the vision resolution, which
// the search holds fixed — so batches and seconds are the same axis here, and
// batches are the one that doesn't move when trials share a machine.
//
// One batch is 1024 shots, so a batch's accuracy carries a standard error near
// 0.35% at the rates seen early in training. finalAcc averages the last quarter
// of the run to cut that down, and distP10 (how close the batch's best tenth of
// shots came to the rim) is reported alongside as the dense, far less noisy
// version of the same question.
export function summarize(record) {
  const rows = record.rows || [];
  if (!rows.length) return { name: record.name, seed: record.seed, failed: true, score: -1 };
  const tail = rows.slice(Math.max(0, Math.floor(rows.length * 0.75)));
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

  const finalAcc = mean(tail.map((r) => r.acc));
  const auc = mean(rows.map((r) => r.acc));
  // The same two numbers over the greedy balls alone — the policy's own shooting
  // rather than the policy's plus whatever spread it is currently exploring at.
  // Ranked on `acc` still, because that is what every stage on disk was ranked
  // on, but a config that only looks better because it explores less shows up
  // here as one whose greedy accuracy did not move.
  const greedy = rows.map((r) => r.accGreedy).filter((v) => v != null);
  const greedyTail = tail.map((r) => r.accGreedy).filter((v) => v != null);
  return {
    name: record.name,
    seed: record.seed,
    batches: rows.length,
    finalAcc,
    auc,
    finalGreedy: greedyTail.length ? mean(greedyTail) : null,
    greedyAuc: greedy.length ? mean(greedy) : null,
    score: 0.5 * finalAcc + 0.5 * auc,
    startAcc: mean(rows.slice(0, Math.max(1, Math.floor(rows.length * 0.25))).map((r) => r.acc)),
    finalReward: mean(tail.map((r) => r.meanReward)),
    finalDistP10: mean(tail.map((r) => r.distP10)),
    finalIllegal: mean(tail.map((r) => r.illegal)),
    minutes: (record.wallMs || 0) / 60000,
    batchSeconds: mean(rows.map((r) => (r.simMs + r.stepMs) / 1000))
  };
}

// Group a stage's per-seed records by config name and average their summaries.
export function rank(records) {
  const byName = new Map();
  for (const rec of records) {
    if (rec.failed) {
      byName.set(rec.name, [{ name: rec.name, failed: rec.failed, score: -1 }]);
      continue;
    }
    if (!byName.has(rec.name)) byName.set(rec.name, []);
    byName.get(rec.name).push(summarize(rec));
  }
  const out = [];
  for (const [name, seeds] of byName) {
    const num = (k) => seeds.map((s) => s[k]).filter((v) => typeof v === "number");
    const avg = (k) => (num(k).length ? num(k).reduce((a, b) => a + b, 0) / num(k).length : null);
    const spread = (k) => (num(k).length > 1 ? Math.max(...num(k)) - Math.min(...num(k)) : 0);
    out.push({
      name,
      seeds: seeds.map((s) => s.seed),
      failed: seeds[0].failed || null,
      score: avg("score"),
      finalAcc: avg("finalAcc"),
      auc: avg("auc"),
      finalGreedy: avg("finalGreedy"),
      greedyAuc: avg("greedyAuc"),
      startAcc: avg("startAcc"),
      finalAccSpread: spread("finalAcc"),
      finalReward: avg("finalReward"),
      finalDistP10: avg("finalDistP10"),
      finalIllegal: avg("finalIllegal"),
      minutes: avg("minutes"),
      batchSeconds: avg("batchSeconds")
    });
  }
  return out.sort((a, b) => (b.score ?? -1) - (a.score ?? -1));
}

export function table(ranked) {
  const pct = (v) => (v == null ? "  --  " : (v * 100).toFixed(2).padStart(6));
  const head =
    "config                  score   final    auc  greedy  g-auc   start  spread  reward   d10  illegal   min   s/batch";
  const lines = ranked.map((r) =>
    r.failed
      ? `${r.name.padEnd(22)}  FAILED: ${String(r.failed).slice(0, 90)}`
      : [
          r.name.padEnd(22),
          pct(r.score),
          pct(r.finalAcc),
          pct(r.auc),
          pct(r.finalGreedy),
          pct(r.greedyAuc),
          pct(r.startAcc),
          pct(r.finalAccSpread),
          (r.finalReward ?? 0).toFixed(2).padStart(6),
          (r.finalDistP10 ?? 0).toFixed(2).padStart(5),
          pct(r.finalIllegal),
          (r.minutes ?? 0).toFixed(1).padStart(5),
          (r.batchSeconds ?? 0).toFixed(1).padStart(6)
        ].join(" ")
  );
  return [head, ...lines].join("\n");
}
