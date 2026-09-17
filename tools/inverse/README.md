# Inverse solver

Exact answers for a problem the app can otherwise only search: **what launch
would have made this shot?**

Everything else in this repository is forward. The agent proposes an action, the
physics play out, the reward comes back, and a thousand shots a batch of that is
what the policy is built from. Nothing in that loop ever says what the *right*
action was — only how the ones it happened to try worked out. So there is no
external answer to check the policy against, and no way to tell an agent that
has plateaued from an agent that has run out of room.

This folder supplies the missing half. A shot between the launch and the ring
touches nothing, so it is three lines of arithmetic, and those three lines are
invertible. Start a ball in a made position, run them backwards, and the shooter
appears: the spot on the floor, the height of the eyes, the launch velocity, and
the action that produces it. No optimiser, no residual, no training run. Run the
same arithmetic the other way and any spot on the floor gives up its exact
action, how much of it you can get wrong, and — once that is known everywhere —
what the best possible policy would shoot.

```sh
npm install          # cannon-es, for the cross-check; once
node selftest.mjs    # every claim below, checked
```

## Why it is exact

Not because flight is time-symmetric. It is not — drag is dissipative, and
running the clock backwards through it is a different flow, not the same one
reversed. What is true is narrower and more useful: the *discrete step the app
takes* is a bijection, and it can be inverted in closed form.

Written in the order `PhysicsWorld.step` and cannon-es's `World.internalStep`
apply them, one step of free flight is

```
a       = airAccel(v_n, w_n)          applyAir runs before world.step, so the
v_{n+1} = Ld * v_n + (g + a) * dt     air reads the velocity from before damping
w_{n+1} = Ad * w_n                    Ld = (1-linearDamping)^dt, Ad likewise
p_{n+1} = p_n + v_{n+1} * dt          semi-implicit: the NEW velocity
```

and inverting it is

```
p_n = p_{n+1} - v_{n+1} * dt          explicit
w_n = w_{n+1} / Ad                    a scale, so a division
v_n : solve Ld*v + (g + airAccel(v, w_n))*dt = v_{n+1}
```

The last line is implicit — the air force depends on the velocity being solved
for — so it is a 3x3 root find, done in `physics.mjs` by Newton against the
analytic Jacobian of the drag and Magnus terms. It converges in two or three
iterations, because over one 1/60s step the air contributes about a tenth of a
foot per second to a velocity of thirty, and it converges to machine precision.

That is the whole claim, and it is measurable: step a state forward 160 times
and back 160 times and it returns to within **4e-14** of where it started. Over
a thousand generated shots, the largest gap between the made position a reverse
pass started from and the point the forward flight actually crossed the ring's
plane is **2.9e-14 ft**. That is nine femtometres — about the width of an
atomic nucleus — against a hole the ball clears by three and a half inches.

The forward stepper is not a model of the app's physics either. Against
cannon-es, with the same body and the same `applyAir`, it agrees **bit for bit**
over two hundred steps of free flight — not to a tolerance, to zero. So an
answer derived here is an answer about the app, not about a reimplementation of
it.

### Where it stops

Two boundaries, both hard-checked rather than assumed.

**Contacts.** The arithmetic above is free flight. The moment a ball would touch
the ring, the board, the post or the floor, it stops describing anything, so
`flight.mjs` sweeps every trajectory against those bodies — continuously, by
conservative advancement in `court.mjs`, which is strictly stricter than the
app's own test since cannon only samples the frame positions. A trajectory that
clears continuously clears cannon's discrete test whatever the ball's speed, so
"this shot touched nothing" is a proof rather than a sampling.

**The reverse horizon.** Drag takes energy out going forwards, so it puts energy
in going backwards, and a reverse pass run past the launch keeps accelerating:
the speed roughly doubles every twenty steps once it is out past 100ft/s. Worse
than inaccurate, it eventually answers a different question, because
`v -> v - k|v|v*dt` folds back on itself at `1/(2*k*dt)` — 4545ft/s here — and
past that fold a target velocity has two preimages or none. A real shot is sixty
to a hundred and sixty steps and spends all of them three orders of magnitude
inside that, so it never arises; `solvePrevVelocity` throws by name if it does,
rather than returning a plausible number off the wrong branch.

## Reverse: a made basket in, a shot out

`reverse.mjs`. Sample a ball crossing the ring — an offset inside the scoring
radius, a speed, an angle, a heading — and trace it back until it is at a
shooter's eye height on the way up. The launch that comes out is then run
through the app's own gates: the spot has to be one `Ball.spawn` would pick, the
height one it would draw from `CONFIG.spawnHeight`, the action inside
`CONFIG.launch`'s envelope. About one made position in four survives all three,
and the ones that do are exactly the shots the agent could have taken and made.

Two small fixed points make the launch land on the numbers asked for rather than
whatever falls out. Spin decays forwards, so it grows backwards, and the entry
spin has to be pre-divided by the decay over a flight whose length takes a pass
to measure. And `Ball.launch` puts the spin axis across the line from the shot
to the ring, which is not known until the shot has been traced back to a
shooter. Both settle in two or three passes.

Which frame of the reverse pass is the launch is a choice, and the one made here
is to let the arc pick the shooter: the pass is stopped at whole steps, several
of which land inside `CONFIG.spawnHeight`'s band, and one is drawn from them
against the app's own truncated-normal density. That keeps the launch exactly a
whole number of steps from the crossing, which is why the round trip closes to
femtometres rather than to the size of one frame of flight.

```sh
node solve.mjs --mode reverse --n 2000 --verify --out made.jsonl
```

Every line is one shot, and carries its own evidence:

```json
{
  "spawn":  { "x": 30.145, "y": 6.118, "z": 2.379 },
  "action": [-0.60703, 0.21516, -0.00990, -0.43779],
  "launch": { "vx": 8.142, "vy": 28.797, "vz": -1.639, ... },
  "margin": 0.393,
  "shotDistance": 11.846,
  "steps": 91,
  "entry":  { "offsetX": 0.190, "offsetZ": 0.003, "speed": 22.549, "angleDeg": 71.2 },
  "verified": { "scored": true, "minGap": 0.023, "closure": 7.1e-15, ... },
  "cannon":   { "scored": true, "clean": true, "hitRim": false, ... },
  "zone": "paint"
}
```

`margin` is how much room the tightest action channel has left inside the
envelope; `minGap` is how much air was between the ball and the ring at the
closest point of the whole flight; `closure` is the round trip — how far the
forward flight's crossing landed from the made position the reverse pass started
at.

This is the cheap direction — one pass of arithmetic per shot, no search — and
its one real limitation is that it does not get to choose where the shooter
stands. The spawn distribution that comes out is whatever the entry sampling
induces, which leans nearer the basket than `Ball.spawn` does, so it is a set of
correct shots rather than a sample of the app's own states. When the spawns have
to match the app's, use the other direction.

## Target: a spot on the floor in, the exact action out

`target.mjs`. This is the boundary-value problem, and it collapses to a
one-dimensional root find once two things are noticed.

`Ball.launch` aims the shot down the line to the ring, so with the side channel
at zero the flight lies in the vertical plane through the ring's axis: drag is
along the velocity, Magnus is `w x v` about an axis `Ball.launch` puts square
across that plane, and neither can push the ball out of it. The crossing is
therefore laterally dead centre — **exactly**, measured at 3e-14 ft — so
`side = 0` is not a good guess, it is the answer.

What is left is one equation, that the arc come down through the ring's plane at
the distance the ring actually is, on two free numbers. So the makes from a spot
are a *curve*, not a point — the same family a shooter picks an arc from — and
the solver sweeps the up channel and the spin channel, solving the forward
channel exactly on each by bracketed false position. From mid-range there are
about 165 of them.

Which one to call *the* answer is a real choice. The one made here is the shot
with the most room for error: the largest box in action space that still drops
the ball through the hole. That is what a policy with any spread at all wants to
be aiming at, and the width of it is the number worth reporting on its own.

```sh
node solve.mjs --mode target --n 200 --verify --out oracle.jsonl
node solve.mjs --mode target --grid 24 --out grid.jsonl
```

## Tolerance, and why it is the interesting output

The exact action is less useful than how much of it you can get wrong. Over 200
spawns drawn the way `Ball.spawn` draws them, at the shipped config:

```
zone              spawns  reachable          tol(fwd)   arcs
Restricted             2     100.0%            ±0.0251     50
Paint                 23     100.0%            ±0.0183    183
Mid-range             73     100.0%            ±0.0148    165
Above break 3         32     100.0%            ±0.0135    141
Beyond 30ft           70     100.0%            ±0.0128    104
```

Two things to read off it.

**Every spawn on the court has a make.** Not nearly all — all of them, verified
in cannon-es with the rings and boards present. Half court included. So no part
of the agent's accuracy is explained by having been given an impossible shot,
and the curriculum's job is entirely about what is *learnable* from a distance
rather than about what is reachable.

**The window is narrower than the policy's own floor.** `tol(fwd)` is in action
units — the same units the actor emits and `CONFIG.policy` measures sigma in.
Mid-range tolerates ±0.0148 on the forward channel. The tightest spread the
policy is allowed to learn is `exp(CONFIG.policy.logStdMin)` = 0.02, and it
starts at 0.25. So even a policy that has learned the exact answer is sampling
wider than the hole for most of a run, and the gap between what it knows and
what it hits is not a learning problem at all.

## The accuracy ceiling

`ceiling.mjs` turns that observation into the number the dashboard should be
read against: hand the exact answer to a policy that explores the way
`CNNAgent.predictBatch` explores — `clamp(mean + sigma * gauss())` per channel —
and count what drops.

```sh
node ceiling.mjs --n 300 --samples 256
node ceiling.mjs --n 200 --samples 128 --engine cannon   # rattle-ins included
```

At the shipped config, over the full 48ft spawn disc:

```
                          the exact policy, exploring at sigma
zone              spawns     0     0.02    0.05    0.15    0.25     NBA   elite
Restricted             4  100.0%   72.5%   31.6%    5.8%    2.1%     65%     70%
Paint                 33  100.0%   57.9%   21.6%    3.1%    1.0%     42%     50%
Mid-range            104  100.0%   47.3%   15.2%    2.0%    0.8%     41%     50%
Above break 3         46  100.0%   43.2%   12.5%    1.6%    0.5%     36%     42%
Beyond 30ft          113  100.0%   40.8%   11.3%    1.5%    0.5%       —       —

all zones            300  100.0%   45.7%   14.3%    1.9%    0.7%
```

That is the `free` engine, which counts only shots that drop through without
touching anything, so it is a floor rather than the figure: it gives up every
shot that would have rattled home. `--engine cannon` re-flies each sample in the
real world instead and counts those too — pooled over its own 200 spawns it
reads 59.4% / 23.9% / 4.9% at sigma 0.02 / 0.05 / 0.15, against 45.7% / 14.3% /
1.9% here. Thirty times slower, and the one comparable to the dashboard.

Read the Policy Sigma off the dashboard and the matching column is what that run
could be shooting if its mean action were perfect. A run at sigma 0.05 hitting
12% is not a policy that has failed to learn the shot; it is a policy shooting
near its ceiling and held there by its own spread.

The 0 column is the reachability check again, from the other side: with no
exploration at all, the exact answer goes in every time, everywhere.

> The ceilings quoted in `CONFIG`'s own comments ("0.15 caps accuracy near 56%
> and 0.04 near 70%") are a different quantity and should not be lined up
> against this table. They were modelled under the uniform jitter that
> `CONFIG.policy` replaced, where the knob was a full width rather than a
> standard deviation: a floor of 0.15 meant uniform ±0.075, which is sigma
> 0.043, and 0.04 meant sigma 0.0115. On this table's convention and the current
> 0.30ft scoring rule those two land at 30.8% and 84.5% (cannon engine).

## Cross-validating a policy

The point of an answer nothing was trained on. Three things it supports, none of
which a training run can produce for itself:

**Is the policy's action right, or only its reward?** Take the greedy action the
actor emits for a state and compare it to the exact one for that spawn, channel
by channel, against `tolerance` as the yardstick. A miss with the forward
channel 0.01 off is a policy that has essentially solved the shot and is being
beaten by its own spread. A miss with it 0.4 off has not learned the shot.

**Has the policy plateaued, or has it arrived?** `ceiling.mjs` at the run's
current Policy Sigma. A flat accuracy trace sitting at its ceiling and a flat one
sitting well under it are the same picture and completely different problems.

**Is a config change worth anything?** The ceiling is a property of the physics,
the envelope and the scoring rule, not of any policy, so it can be swept with
`--set` in seconds and answers "could this help" before a single trial is spent
finding out. `--set launch.fwdMax=34` moves the envelope; `--set air.magnus=0`
takes the lift out; `--set hoopEntry.scoreRadius=0.2` tightens the hole.

## Options

`--set` takes the same dotted `CONFIG` paths `tools/hpsearch` does
(`--set launch.fwdMax=34`, `--set hoopEntry.scoreRadius=0.2`), so an oracle can
be built for the config a search trial ran under rather than only for the
shipped one. `--script` points at a different `script.js`. `--seed` makes a run
reproducible; the generator is the same mulberry32
`tools/hpsearch/pagehooks.js` seeds `Math.random` with. `--verify` re-flies every
answer in cannon-es. `node solve.mjs --help` and `node ceiling.mjs --help` have
the rest.

## What is checked, and by what

Nothing here is trusted because the arithmetic looked right.

`cannoncheck.mjs` builds the app's world for real — `PhysicsWorld`'s gravity,
broadphase, solver and contact material, `Court`'s floor, both hoops' rings,
boards, posts and scoring sensors — launches a ball through `Ball.launch`'s own
impulse, steps it with `applyAir` in front of each step, and grades it with
`trackHoopPass`. It shares no code with the stepper, so if the replica and the
clearance sweep were both wrong in the same direction, this is what would
notice.

Of 1000 shots from a reverse pass and 200 from a targeted solve: **1200 scored,
1200 clean swishes, none touching anything.**

`selftest.mjs` asserts the rest: that `script.js` is still shaped the way the
config reader reads it, that both directions of the stepper round-trip, that the
action mapping round-trips, that reverse-pass and targeted solutions alike all
drop through cannon-es's ring untouched, that the crossing really is laterally
exact, that just inside the reported tolerance a shot drops and just outside it
does not, and that per-shot weather is refused rather than quietly solved as
though it were not there.

There are no physics constants in this folder. `appconfig.mjs` reads `CONFIG`,
`COURT`, `SHOT_ZONES` and `shotZone` out of `script.js` as one contiguous slice —
the same span `tools/hpsearch/patch.mjs` anchors on — and pulls gravity, the
timestep, the ball's mass and damping, and the rim, board, post and sensor
geometry out of the class bodies by anchored regex, each of which throws by name
if its anchor moves. An oracle is the last thing that should be confidently
wrong because a constant was copied and left behind.

The one dependency is cannon-es, pinned to the app's own version, and it is
optional: without `npm install` the cross-checks skip by name and everything
else still runs.

## Limits

- **One air for the whole gym.** `CONFIG.air.wind` and `air.jitter` draw the
  weather after the action is chosen, which makes the map from action to outcome
  stop being a function — the same action from the same spot lands somewhere
  else next time, and "the impulse that would have made the shot" stops being a
  thing there is one of. `makeSolver` refuses that config by name rather than
  solving it as if the air were still.
- **Clean swishes only, on the `free` engine.** The solver integrates free
  flight, so a shot that banks in off the board or rattles home is outside what
  it will construct. Continuous clearance implies cannon finds no contact
  either, so its rates are a true lower bound on the app's and not an
  approximation of them — a loose one, worth 14 points at sigma 0.02 and a
  factor of two and a half at 0.15. `--engine cannon` closes the gap by
  re-flying every sample in the real world.
- **Reverse mode gives correct shots, not a sample of the app's.** The spawns
  come out of the entry sampling, not out of `Ball.spawn`; target mode is the
  one whose spawns match.
- **The canonical answer is a choice, not a theorem.** Every arc in the family
  is an exact make; "the" exact action is the one with the widest box around it,
  which is a defensible criterion and not the only one. `target.family()` hands
  back the whole curve for callers who want a different one.
- **The ceiling is a ceiling over the exact answers**, taken as the best of the
  few widest arcs from each spawn rather than over every possible mean action.
  It is what a policy that has learned the exact answer would shoot, which is
  close to but not provably the maximum over all policies at that spread.

## Files

| | |
| --- | --- |
| `appconfig.mjs` | `script.js`'s own numbers, read out of it rather than restated |
| `physics.mjs` | one step of free flight, and its exact inverse |
| `court.mjs` | the action mapping, and clearance against the bodies |
| `flight.mjs` | a shot flown forward, graded by `TrainingArena.update`'s rules |
| `reverse.mjs` | made position in, shot out |
| `target.mjs` | spot on the floor in, exact action and tolerance out |
| `cannoncheck.mjs` | the same shot re-flown in cannon-es, sharing no code |
| `harness.mjs` | wiring, spawn sampling, argument shapes |
| `rng.mjs` | the seeded generator `tools/hpsearch` uses, so runs reproduce |
| `solve.mjs` | the CLI: build a dataset either way |
| `ceiling.mjs` | the CLI: achievable make rate by zone, at a given spread |
| `selftest.mjs` | every claim above |
