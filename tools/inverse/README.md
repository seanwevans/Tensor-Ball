# Inverse solver

The app's flight physics, exactly invertible.

Everything else in this repository is forward. The agent proposes an action, the
physics play out, the reward comes back, and a thousand shots a batch of that is
what the policy is built from. Nothing in that loop ever says what the *right*
action was — only how the ones it happened to try worked out. So there is no
external answer to check the policy against, and no way to tell an agent that
has plateaued from an agent that has run out of room.

A shot between the launch and the ring touches nothing, so it is three lines of
arithmetic, and those three lines are invertible. This is the part that makes
them so: run them backwards from a ball dropping through the ring and the ball
retraces its flight to the shooter's hands, exactly. What gets built on top of
that — deriving the launch that made a basket, solving the exact action from any
spot on the floor, and the accuracy ceiling both imply — follows.

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
and back 160 times and it returns to within **4e-14** of where it started.

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

## What is checked, and by what

Nothing here is trusted because the arithmetic looked right.

`cannoncheck.mjs` builds the app's world for real — `PhysicsWorld`'s gravity,
broadphase, solver and contact material, `Court`'s floor, both hoops' rings,
boards, posts and scoring sensors — launches a ball through `Ball.launch`'s own
impulse, steps it with `applyAir` in front of each step, and grades it with
`trackHoopPass`. It shares no code with the stepper, so if the replica and the
clearance sweep were both wrong in the same direction, this is what would
notice.

`selftest.mjs` asserts the rest: that `script.js` is still shaped the way the
config reader reads it, that both directions of the stepper round-trip, that the
action mapping round-trips, and that per-shot weather is refused rather than
quietly solved as though it were not there.

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
- **Free flight only.** Anything after a contact is cannon's business, not this
  folder's. `cannoncheck.mjs` is how a shot that rattles gets flown.

## Files

| | |
| --- | --- |
| `appconfig.mjs` | `script.js`'s own numbers, read out of it rather than restated |
| `physics.mjs` | one step of free flight, and its exact inverse |
| `court.mjs` | the action mapping, and clearance against the bodies |
| `flight.mjs` | a shot flown forward, graded by `TrainingArena.update`'s rules |
| `cannoncheck.mjs` | the same shot re-flown in cannon-es, sharing no code |
| `harness.mjs` | wiring, spawn sampling, argument shapes |
| `rng.mjs` | the seeded generator `tools/hpsearch` uses, so runs reproduce |
| `selftest.mjs` | every claim above |
