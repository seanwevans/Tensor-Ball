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
- **Reward** — a made basket scores far and away the highest (bonused by shot
  distance); near misses are shaped by how close the ball got to the rim, with
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
  The current spread is on the dashboard as Policy Sigma.

The live dashboards show the agent's stereo input, the learned first-layer
filters, dense-layer activations, the critic's batch loss, and running accuracy.

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

The exported actor has a six-wide head: three action means (apply `tanh`) and
three log standard deviations. Take the means for a deterministic policy.

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
