# Tensor Ball

A browser-based basketball reinforcement-learning playground. A thousand agents
learn to shoot a hoop from raw stereo vision, trained live in the tab with
TensorFlow.js — no server, no build step.

The whole batch is on the court at once. Every ball of every batch is drawn,
shoots, and stays where it landed until the next batch spawns, so a single
glance shows the entire policy: early on the shots scatter everywhere, and as
the critic learns the cloud pulls in toward the rim.

<img src="https://github.com/user-attachments/assets/d7765d49-cd3f-412c-bb49-44dbc7f32d08" />

## What it is

Each ball carries a pair of stereo cameras aimed down the line to the rim —
parallel, not verged onto it, so the rim's own disparity across the pair is the
distance to it rather than zero. Every training
batch, the agent renders each ball's view, feeds the two grayscale images through
a small convolutional network, and outputs a 4-vector launch action (vertical
power, forward power, aim adjustment, spin) — `CONFIG.launch` is the envelope
in ft/s and rad/s that vector is stretched onto. The ball is launched, the
physics play out, and the shot's outcome becomes the reward. The agent learns from vision
alone — it is never told where the hoop is.

### How the learning works

It's a contextual-bandit actor-critic: each shot is a single action with a single
terminal reward, so there's no temporal credit assignment.

- **Critic** — a CNN that predicts the expected reward (value) of a state.
- **Actor** — shares the critic's convolutional backbone (copied over after each
  batch, and frozen in the actor) and learns a launch action. It imitates the
  best slice of each batch by advantage — `CONFIG.imitateFraction`, the top
  fifth — a form of self-imitation / advantage-weighted regression. The bar is
  set by the batch rather than by the critic's zero, because a miss is worth
  about -0.3 and a make about +25: once the policy makes anything the value
  head sits above every miss, a sign gate admits exactly the makes, and the
  shot that rattled out is discarded on the same grounds as the one that sailed
  into the third row.
- **Reward** — a made basket scores far and away the highest (bonused by shot
  distance), where "made" means the ball's centre crossed the rim's plane
  descending and inside the hole — the hole minus the ball, 0.30ft
  (`CONFIG.hoopEntry.scoreRadius`), not merely somewhere near the rim; near misses are shaped by how close the ball got to the rim, with
  small bonuses for hitting the rim or backboard. Direction through the hoop is
  what separates the top of the scale from the bottom: a ball that comes *up*
  through the rim from underneath takes a large flat penalty and is barred from
  scoring on the way back down, because its proximity shaping is otherwise the
  best in the batch and shooting straight up through the net would be the
  easiest policy to learn.
- **Exploration** — the policy is Gaussian and learns its own spread. The actor
  emits a mean and a log standard deviation per action channel, shots are
  sampled from that distribution, and the advantage-weighted objective is the
  likelihood of the action taken. So exploration narrows where the agent's good
  shots agree and stays wide where they do not — per state, rather than on a
  global schedule (`CONFIG.policy` sets where it starts and how far it may go).
  The current spread is on the dashboard as Policy Sigma. A slice of each batch
  (`CONFIG.evalBalls`) takes the distribution's mean instead of a draw from it,
  so the dashboard reports the policy's own accuracy next to the behaviour
  policy's — the two move independently, and only the first is what the agent
  has learned. Those greedy balls fit the critic and feed the replay buffer but
  are held out of the actor's own update: an action drawn at the distribution's
  mean cannot move that mean, and under a likelihood objective the only thing
  it says about the spread is "narrower", whatever the policy has learned.
- **Replay** — the highest-advantage shots are kept in a fixed-capacity buffer
  (`CONFIG.replay`) and replayed into each actor update, so a made basket is
  worth more than the single gradient step it used to get before being
  discarded. Entries are re-priced against the current critic as they are
  drawn: an advantage only means something next to the value estimate it was
  measured against, and a buffer that keeps comparing fresh advantages to stale
  ones lets the bar to get in ratchet up until almost nothing clears it.
- **Spin & air** — the fourth action channel is backspin or topspin, applied as
  the ball's launch angular velocity. It is worth having because the ball flies
  through air: `CONFIG.air` adds quadratic drag and a Magnus force, both written
  as accelerations so the ball's (unphysically heavy) mass drops out and the
  constants can be calibrated from a real basketball — terminal velocity lands
  at 70ft/s against a real ball's ~66. Sweeping the action space against the
  physics, having spin to choose is worth about 4 points of achievable accuracy
  over locking it at zero, and backspin wins at most spawns. The air is the same
  for every shot by default; `air.wind` and `air.jitter` roll it per shot
  instead, which is more like a real gym but adds reward noise the agent has no
  way to see coming.
- **Curriculum** — balls spawn inside a radius of the rim that starts small and
  grows, one-way, whenever the rolling accuracy clears a threshold
  (`CONFIG.curriculum`). The far court is where shots are both hardest to hit
  and hardest to range — stereo disparity is sub-pixel out there — so opening
  it up is something the policy earns rather than something it starts with.

The court carries the full NBA marking set at regulation dimensions — the
three-point arc at 23.75ft from the middle of the ring with 22ft corners, a
16ft lane, free-throw circles solid away from the basket and dashed inside the
lane, restricted-area arcs, and the centre circles. All of it merges into one
mesh, because vision capture renders the scene twice per ball and a draw call
added to the court is one paid two thousand times a batch.

The live dashboards show the agent's stereo input, the learned first-layer
filters, dense-layer activations, the critic's batch loss, and running accuracy.

### Reading the accuracy: the shot chart

The headline accuracy is one make rate pooled over the whole spawn disc, and
once the curriculum has opened that disc it reaches half court. Over a third of
every batch is then launched from beyond 30ft — territory the NBA takes about
one shot in two hundred from, most of them buzzer heaves. A number pooled over
2ft layups and 45ft heaves has no counterpart in any box score, and it drops
whenever the curriculum opens the floor rather than when the policy gets worse.

So the dashboard also carries a **shot chart**: the same shots split into the
buckets the league's own charts use — restricted area, paint, mid-range, corner
three, above the break — with what the best shooters in the NBA manage from
each alongside. Every zone is scored on the greedy balls only
(`CONFIG.evalBalls`), because parity with a shooter is a claim about the policy,
not about the policy plus however wide it is currently exploring.

| Zone | League | Elite |
| --- | --- | --- |
| Restricted area (inside 4ft) | ~65% | ~70% |
| Paint, outside the restricted arc | ~42% | ~50% |
| Mid-range | ~41% | ~50% |
| Corner three | ~39% | ~45% |
| Above the break three | ~36% | ~42% |
| Beyond 30ft | — | — |

Those are game rates, shot against a defence, off the dribble, on a shot clock.
An agent alone in a gym with nobody's hand in its face should be expected to
beat them — and until it reaches them it is not close. Beyond 30ft has no
benchmark on purpose: there is no NBA rate worth being at parity with out
there.

Zones are cut on the lines already painted on the floor — the same `COURT`
constants `Court._buildMarkings` draws from, so a spawn is judged against the
line it is standing next to. Each zone averages over its own last
`CONFIG.zoneWindow` shots rather than a shared window: the spawn disc is
area-uniform, so the restricted area collects about one shot for every thirty
from beyond the arc, and a window measured in batches would leave it reading
nothing for most of a run.

`tools/hpsearch/chart.mjs` prints the same table for a finished headless run.

### Batch size vs. retina resolution

The two are budgeted against each other in `CONFIG`, because every buffer in
the pipeline — the capture atlas, the readback, the state tensor — scales with
`batchSize * visionWidth * visionHeight * 2`. The defaults spend that budget on
balls rather than pixels: 1024 balls at 96x96 per eye. Raising the resolution
without lowering the batch (or vice versa) is what the numbers are there for;
`maxAtlasDim` keeps the GPU-side capture bounded either way, by filling a
capped atlas in as many passes as the batch needs.

## Controls

The app opens in **manual mode** with a single ball you can play with:

| Action | Control |
| --- | --- |
| Aim & shoot | Left-drag the ball |
| Reposition the ball | Right-drag the ball |
| Orbit the camera | Drag the background |
| Reset the ball | Space |
| Start / stop learning | **START TRAINING** button |
| Save the trained policy | **EXPORT POLICY** button (downloads the actor) |
| Load a saved policy | **IMPORT POLICY** button (select the exported .zip) |

The exported actor has an eight-wide head: four action means (apply `tanh`) and
four log standard deviations. Take the means for a deterministic policy.

**EXPORT POLICY** downloads one file, `basketball-agent-actor.zip`, and
**IMPORT POLICY** takes it back. Inside the archive is exactly what
TensorFlow.js writes for a model — `model.json` (topology) and
`model.weights.bin` — so unzipping it gives an ordinary tfjs model that
`tf.loadLayersModel` will read anywhere else; the archive is a container, not a
second format. Rezipping those two files with any tool produces something the
importer still accepts, compressed or not, and in a folder or not.

The weights are copied into the running actor, so training continues from the
imported policy rather than from scratch; the shared conv filters are pushed
into the critic at the same time, so the first batch after an import does not
overwrite them. An archive that has been truncated or corrupted fails its
checksum, and one exported under a different `visionWidth`/`visionHeight` or
from a build with a different action space fails a shape check — either way the
running policy is left alone. Import is only available while training is
stopped: the batch in flight was shot by the outgoing policy, and the update it
triggers on landing would pull the freshly imported actor back toward it.

While training, the manual ball is disabled and the batch takes over the court.
Stopping training clears the batch off the floor and hands the court back.

## Running locally

The app is static, but it uses native ES modules, so it must be served over HTTP
(opening `index.html` via `file://` will fail on the module imports). Any static
server works, for example:

```sh
python3 -m http.server 8000
# then open http://localhost:8000
```

A **WebGPU**-capable browser gives the best performance; the app falls back to
**WebGL**, then **CPU**, automatically (the current backend is shown in the
dashboard).

## Tech stack

- [Three.js](https://threejs.org/) — rendering and the agents' stereo vision.
- [cannon-es](https://github.com/pmndrs/cannon-es) — rigid-body physics.
- [TensorFlow.js](https://www.tensorflow.org/js) — the actor-critic network,
  running on WebGPU / WebGL / CPU.

The code is one dependency-free `script.js`, organized into small classes:
`SceneManager`, `PhysicsWorld`, `Court`, `Ball`, `BallField` (the batch, drawn
as one instanced mesh), `VisionSystem` (stereo capture), `CNNAgent` (the
network), `TrainingArena` (the RL loop), and `Dashboard` (the visualizations).

### A note on dependencies & integrity

The TensorFlow.js scripts are loaded from jsDelivr with
[Subresource Integrity](https://developer.mozilla.org/en-US/docs/Web/Security/Subresource_Integrity)
hashes pinned in `index.html`, so a tampered CDN response is rejected by the
browser. Three.js and cannon-es are imported as ES modules from esm.sh; native
`import` has no SRI mechanism, so they are pinned by exact version only. For a
fully locked-down deployment, vendor all four locally.

## License

See [LICENSE](LICENSE).
