# Tensor Ball

A browser-based basketball reinforcement-learning playground. Hundreds of agents
learn to shoot a hoop from raw stereo vision, trained live in the tab with
TensorFlow.js — no server, no build step.

<img src="https://github.com/user-attachments/assets/d7765d49-cd3f-412c-bb49-44dbc7f32d08" />

## What it is

Each ball carries a pair of stereo cameras pointed at the rim. Every training
batch, the agent renders each ball's view, feeds the two grayscale images through
a small convolutional network, and outputs a 3-vector launch action (vertical
power, forward power, aim adjustment). The ball is launched, the physics play
out, and the shot's outcome becomes the reward. The agent learns from vision
alone — it is never told where the hoop is.

### How the learning works

It's a contextual-bandit actor-critic: each shot is a single action with a single
terminal reward, so there's no temporal credit assignment.

- **Critic** — a CNN that predicts the expected reward (value) of a state.
- **Actor** — shares the critic's convolutional backbone (copied over after each
  batch, and frozen in the actor) and learns a launch action. It imitates only
  the actions that beat the critic's value estimate (positive advantage), a form
  of self-imitation / advantage-weighted regression.
- **Reward** — a made basket scores highest (bonused by shot distance); near
  misses are shaped by how close the ball got to the rim, with small bonuses for
  hitting the rim or backboard.
- **Exploration** — Gaussian-ish noise added to actions, annealed over training
  so the agent explores widely early and exploits as it improves.

The live dashboards show the agent's stereo input, the learned first-layer
filters, dense-layer activations, the critic's batch loss, and running accuracy.

### Training runs off the render clock

Training is decoupled from what you see. The batch is simulated **headless** —
the training balls are rigid bodies with no meshes and are never drawn — and
each rendered frame fast-forwards as much physics as fits in a fixed wall-clock
budget (`CONFIG.simBudgetMs`) instead of advancing one step per frame. A batch
now takes as long as it takes to *compute*, rather than as long as a thousand
balls take to fall.

What the court shows instead is a **showcase**: the ten best shots the agent has
taken are promoted once a second and replayed at normal speed, laying down
trails as they go. It's a read-only view of training, so rendering can stutter
or be throttled without costing a single shot. The `Shots/s` and `Render`
readouts in the training panel report the two rates separately.

Batch state is held as 8-bit luma (what the framebuffer actually produces) and
converted to float a chunk at a time on its way to the GPU, so peak GPU
residency is set by `CONFIG.gpuChunk` rather than by the batch size. Vision
capture pages through a bounded atlas for the same reason. Between them, batch
size and retina resolution can be raised without the memory footprint growing
to match.

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

While training, the manual ball is hidden and the court switches to replaying
the agent's best recent shots.

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
`SceneManager`, `PhysicsWorld`, `Court`, `Ball`, `VisionSystem` (stereo capture),
`CNNAgent` (the network), `TrainingArena` (the RL loop), `Showcase` (replay of
the best shots), and `Dashboard` (the panels).

Everything tunable lives in the `CONFIG` object at the top of `script.js`. The
throughput-relevant knobs are `simBudgetMs` (physics fast-forwarded per rendered
frame), `gpuChunk` (samples per GPU upload), `visionAtlasMax` (vision atlas page
size), and `maxFlightSteps` (when to abandon a shot). `batchSize` and
`visionWidth`/`visionHeight` still dominate the cost of a batch — they set how
much pixel data every stage has to move.

### A note on dependencies & integrity

The TensorFlow.js scripts are loaded from jsDelivr with
[Subresource Integrity](https://developer.mozilla.org/en-US/docs/Web/Security/Subresource_Integrity)
hashes pinned in `index.html`, so a tampered CDN response is rejected by the
browser. Three.js and cannon-es are imported as ES modules from esm.sh; native
`import` has no SRI mechanism, so they are pinned by exact version only. For a
fully locked-down deployment, vendor all four locally.

## License

See [LICENSE](LICENSE).
