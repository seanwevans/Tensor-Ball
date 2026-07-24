<img src="https://github.com/user-attachments/assets/d7765d49-cd3f-412c-bb49-44dbc7f32d08" />

# Tensor Ball

A browser-based basketball reinforcement-learning playground. An actor–critic
agent learns to sink shots from stereo "vision" — it never sees the world
state directly, only a pair of downsampled camera images rendered toward the
hoop — while you watch the network's filters, activations, and loss update
live. You can also grab a ball yourself and shoot in manual mode.

Everything runs client-side: rendering with [three.js](https://threejs.org/),
rigid-body physics with [cannon-es](https://pmndrs.github.io/cannon-es/), and
training with [TensorFlow.js](https://www.tensorflow.org/js) on the GPU
(WebGPU, falling back to WebGL, then CPU).

## Running it

The app uses ES modules and loads dependencies from CDNs, so it needs to be
served over HTTP — opening `index.html` from the filesystem (`file://`) will
not work. Any static server does the job:

```bash
# Python
python3 -m http.server 8000

# or Node
npx serve .
```

Then open <http://localhost:8000>. A browser with WebGPU or WebGL support is
recommended; the backend in use is shown in the bottom-left panel.

## Controls

| Input | Action |
| --- | --- |
| **Left-drag ball** | Aim & shoot (drag back like a slingshot) |
| **Right-drag ball** | Reposition the ball |
| **Drag background** | Orbit the camera |
| **Space** | Reset the manual ball |
| **START TRAINING** | Toggle the RL training loop (fast-forward) |
| **EXPORT POLICY** | Download the trained actor network |

## How it works

- **Vision.** For each shot the agent renders a stereo (left/right) pair of
  64×64 grayscale views aimed at the hoop, giving it a crude sense of depth.
- **Agent.** A small CNN shared backbone (two conv layers → dense) feeds two
  heads: a *critic* that predicts the shot's return and an *actor* that
  outputs a 3-vector action — forward power, vertical power, and lateral aim.
- **Training.** Each batch spawns many balls at random spots, launches them
  with the actor's (noisy) actions, and scores the outcomes. The critic is
  fit on the returns; the actor is nudged toward the actions that beat the
  critic's value estimate (advantage-weighted regression).
- **Reward.** Made baskets score highest (scaled by shot distance); grazing
  the rim or backboard earns partial credit; otherwise the reward falls off
  with how close the ball got to the hoop.

Key hyperparameters (batch size, learning rate, exploration noise, vision
resolution, court geometry) live in the `CONFIG` object at the top of
`script.js`.

## Project layout

| File | Purpose |
| --- | --- |
| `index.html` | Page structure, dashboard panels, CDN script tags |
| `script.js` | Everything: scene, physics, vision, agent, training loop, UI |
| `style.css` | Dashboard and overlay styling |

## License

See [LICENSE](LICENSE).
