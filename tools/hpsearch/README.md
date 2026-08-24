# Hyperparameter search harness

Runs Tensor Ball's training loop headlessly so configs can be compared without
a human watching a tab, and ranks them on how accurate a policy they reach and
how fast they get there.

## What it runs

The trials run **the app**, not a reimplementation of it. `patch.mjs` opens two
seams in `script.js` — one right after the `CONFIG` literal so a trial can
overlay any knob before anything reads it, one at the bootstrap line so the
harness can attach measurement — and `site.mjs` serves the repo's own
`index.html` with TensorFlow.js, three and cannon-es vendored out of
`node_modules` instead of the CDNs. Same physics, same reward, same stereo
capture, same network.

`pagehooks.js` holds everything that differs from a browser session, and all of
it is either measurement or headless plumbing:

- `Math.random` is a seeded PRNG, so two configs see identical spawn positions,
  exploration draws and weight init, and differences between them come from the
  config rather than from luck;
- `requestAnimationFrame` is unhooked from the frame clock, because a 600-frame
  shot costs ten wall seconds at 60Hz and the physics step is a fixed 1/60
  either way;
- the main canvas render, the trajectory trails and the dashboard canvases are
  stubbed. None of them feed the agent. **The stereo capture is untouched** —
  it still renders the real scene at the real resolution.

## Use

```sh
npm install                     # vendored deps, once

# one run
node trial.mjs --batches 20 --seed 1 --set exploreNoise=0.9 --out run.json

# where the time goes
node profile.mjs --set batchSize=256

# a whole stage from space.mjs, ranked
node search.mjs --stage screen --batches 14 --seeds 1 --concurrency 3
```

`--set` takes dotted CONFIG paths (`--set reward.missDistanceScale=5`).
`--batches` caps a run by batch count and `--minutes` by wall clock; pass both
and whichever comes first ends the run. Finished trials are written out one at a
time, and re-running a stage skips the trials already on disk, so an interrupted
stage picks up where it stopped.

## The shot chart

Every trial row carries per-zone attempt and make counts (`rows[i].zones`), cut
into the buckets the NBA's own shot charts use, and the trial file carries the
zone definitions and the league rates alongside them. `chart.mjs` pools them
over the tail of a run and prints the comparison:

```sh
node chart.mjs --tail 0.25 results/nba/baseline.s1.json
```

A run's `acc` is pooled over a spawn disc that reaches half court, so it is not
comparable to any shooting number a basketball reference would print, and it
moves when the curriculum opens the floor. Per zone it is.

## Scoring

`score.mjs` ranks on `0.5 * finalAcc + 0.5 * auc`, where `finalAcc` is mean
accuracy over the last quarter of the run and `auc` is mean accuracy across
every batch. Both are the behaviour policy's; the table carries the same two
numbers over the greedy balls alone (`greedy`, `g-auc`) beside them, so a
config that only looks better because it explores less shows up as one whose
greedy accuracy did not move — a config has to reach a good policy *and* get there early to win.
See the header comment there for why the speed half is counted in batches, when
to use `--minutes` instead, and what the noise floor on a batch's accuracy is.

## Cost

On a 4-core sandbox with SwiftShader (no GPU), a full-size batch — 1024 balls,
96px per eye — costs about 21s: ~2s stereo capture, ~5s action inference, ~14s
critic + actor update. One trial saturates roughly three cores, so concurrency
past 2-3 buys little.
