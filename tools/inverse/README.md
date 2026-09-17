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
the action that produces it. No optimiser, no residual, no training run.

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
four hundred generated shots, the largest gap between the made position a
reverse pass started from and the point the forward flight actually crossed the
ring's plane is **2.8e-14 ft**. That is nine femtometres — about the width of an
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

## A made basket in, a shot out

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
node solve.mjs --n 2000 --verify --out made.jsonl
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

`--set` takes the same dotted `CONFIG` paths `tools/hpsearch` does
(`--set launch.fwdMax=34`, `--set hoopEntry.scoreRadius=0.2`), so a set of shots
can be built for the config a search trial ran under rather than only for the
shipped one. `--script` points at a different `script.js`. `--seed` makes a run
reproducible; the generator is the same mulberry32
`tools/hpsearch/pagehooks.js` seeds `Math.random` with. `node solve.mjs --help`
has the rest.

This is the cheap direction — one pass of arithmetic per shot, no search — and
its one real limitation is that it does not get to choose where the shooter
stands. The spawn distribution that comes out is whatever the entry sampling
induces, which leans nearer the basket than `Ball.spawn` does, so it is a set of
correct shots rather than a sample of the app's own states.

## What is checked, and by what

Nothing here is trusted because the arithmetic looked right.

`cannoncheck.mjs` builds the app's world for real — `PhysicsWorld`'s gravity,
broadphase, solver and contact material, `Court`'s floor, both hoops' rings,
boards, posts and scoring sensors — launches a ball through `Ball.launch`'s own
impulse, steps it with `applyAir` in front of each step, and grades it with
`trackHoopPass`. It shares no code with the stepper, so if the replica and the
clearance sweep were both wrong in the same direction, this is what would
notice.

Of 400 shots from a reverse pass at the shipped config: **400 scored, 400 clean
swishes, none touching anything.**

`selftest.mjs` asserts the rest: that `script.js` is still shaped the way the
config reader reads it, that both directions of the stepper round-trip, that the
action mapping round-trips, that a hundred and twenty reverse-pass solutions all
drop through cannon-es's ring untouched, and that per-shot weather is refused
rather than quietly solved as though it were not there.

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
- **Free flight only.** A shot that banks in off the board or rattles home is
  outside what the reverse pass will construct; every shot it emits is a clean
  swish. `cannoncheck.mjs` is how a shot that rattles gets flown.
- **Correct shots, not a sample of the app's.** See the end of the reverse-pass
  section: the spawns come out of the entry sampling, not out of `Ball.spawn`.

## Files

| | |
| --- | --- |
| `appconfig.mjs` | `script.js`'s own numbers, read out of it rather than restated |
| `physics.mjs` | one step of free flight, and its exact inverse |
| `court.mjs` | the action mapping, and clearance against the bodies |
| `flight.mjs` | a shot flown forward, graded by `TrainingArena.update`'s rules |
| `reverse.mjs` | made position in, shot out |
| `cannoncheck.mjs` | the same shot re-flown in cannon-es, sharing no code |
| `harness.mjs` | wiring, spawn sampling, argument shapes |
| `rng.mjs` | the seeded generator `tools/hpsearch` uses, so runs reproduce |
| `solve.mjs` | the CLI: build a set of exactly-made shots |
| `selftest.mjs` | every claim above |
