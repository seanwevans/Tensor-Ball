import * as THREE from "https://esm.sh/three@0.132.2";
import { OrbitControls } from "https://esm.sh/three@0.132.2/examples/jsm/controls/OrbitControls.js";
import * as CANNON from "https://esm.sh/cannon-es@0.19.0";

// Sizing note. The bottleneck this configuration is tuned against is not the
// GPU — a discrete card sits near idle while a batch runs, because every stage
// that feeds it was on the CPU and serialized behind a synchronous readback.
// The budget is therefore spent where it converts into learning: a much larger
// network, larger optimizer minibatches, and a whole batch resident in VRAM,
// paid for by not pushing a quarter-megapixel per eye through a JS pixel loop.
const CONFIG = {
  // Shots per policy update. The reward is a single terminal scalar per shot,
  // so the advantage estimate is noisy and the batch is the only thing
  // averaging it; doubling it cuts the standard error of the batch statistics
  // without costing proportional wall clock, because the whole batch flies in
  // one shared physics timeline rather than one after another.
  batchSize: 2048,
  // 128px retina rather than 256. Nothing in this task needs a quarter-megapixel
  // per eye: from the far end of the court the backboard subtends ~14% of the
  // 60-degree FOV, so it lands on ~18 pixels at 128 and the rim on ~5 — enough
  // to localize and range the hoop, which is all the policy has to do. What 256
  // bought instead was 4x the framebuffer readback, 4x the luma conversion, and
  // 4x the CPU->GPU upload, every bit of it on the critical path and none of it
  // on the GPU. Halving the retina is what pays for CONFIG.model below.
  visionWidth: 128,
  visionHeight: 128,
  ballRadius: 0.4,
  learningRate: 0.001,
  l2: 0.001,
  maxHistory: 100,
  accuracyWindow: 2048,
  exploreNoise: 0.4,
  exploreNoiseMin: 0.15,
  exploreNoiseDecay: 0.999,
  advantageTemp: 1.0,
  advantageClip: 20.0,
  ipd: 0.2067,
  visionFov: 60,

  // --- Network capacity -----------------------------------------------------
  // The shared conv backbone and the dense head, in one place. The old stack
  // was conv(8) -> conv(16) -> dense(64): about 15k convolutional weights, thin
  // enough that the first layer could afford maybe a couple of oriented edge
  // detectors per eye and the policy had to read hoop distance out of 64
  // numbers. It also left a 16GB card running at single-digit utilization,
  // because a network that small cannot saturate anything.
  //
  // This is the knob to turn to spend more GPU: widening the convs or the dense
  // head raises both VRAM residency and per-step compute, and nothing else in
  // the pipeline has to change — the layer count, the freeze list, the
  // critic->actor weight sync, and the kernel/activation dashboards are all
  // derived from this list rather than hardcoded.
  model: {
    convs: [
      { filters: 32, kernelSize: 8, strides: 4 },
      { filters: 64, kernelSize: 4, strides: 2 },
      { filters: 64, kernelSize: 3, strides: 1 }
    ],
    dense: 512
  },

  // --- Optimizer ------------------------------------------------------------
  // Samples per optimizer step. The old value was fit()'s default of 32, which
  // on a batch of 1024 meant 32 steps over 32-sample minibatches — small enough
  // that each kernel launch cost more than the arithmetic it dispatched. 128
  // keeps roughly the same number of steps per batch (2048/128 = 16 per epoch,
  // two epochs) while giving each one four times the samples, so the gradients
  // are less noisy and the GPU actually has work to do per launch.
  minibatch: 128,
  // Passes over each batch. The batch is already resident in VRAM by the time
  // training starts, so a second pass costs GPU time and nothing else — no
  // re-upload, no re-conversion. Every shot is expensive to collect (it has to
  // be simulated); using each one twice is the cheapest sample-efficiency win
  // available.
  trainEpochs: 2,
  // Samples per forward-only pass (action inference, value estimation). Bounds
  // activation memory for the passes that do not need gradients.
  gpuChunk: 256,

  // --- Throughput -----------------------------------------------------------
  // Training no longer runs at display rate. The old loop stepped physics once
  // per requestAnimationFrame, so a batch could not finish faster than the
  // slowest ball's real-time flight — roughly ten seconds of wall clock spent
  // watching balls fall. These bound how much simulation is fast-forwarded per
  // rendered frame: keep stepping until either the step cap or the wall-clock
  // budget is hit, then yield so the browser can paint.
  //
  // simBudgetMs is the knob that trades framerate for learning rate. At 20ms
  // against a scene render of comparable cost, something close to half of every
  // second went into drawing a reel of ten balls and the simulation got the
  // rest. Raising the slice to 50ms and skipping repaints (renderIntervalMs)
  // shifts that split toward the batch from both directions at once.
  simBudgetMs: 50,
  // High enough that simBudgetMs is the binding constraint rather than this.
  maxStepsPerFrame: 2000,
  // Minimum wall clock between rendered frames while training. The court is a
  // read-only replay of shots already taken, so a repaint costs simulation time
  // and returns nothing to the agent. Manual mode ignores this and renders every
  // frame — there, the picture *is* the application.
  //
  // Note how this interacts with simBudgetMs: a loop tick already takes at
  // least simBudgetMs, so an interval below that never skips anything. 66
  // against a 50ms budget drops roughly one repaint in three (~15fps playback,
  // ~75% of wall clock to the batch). Push it to 100+ to trade more of the
  // reel's smoothness for shots/sec.
  renderIntervalMs: 66,
  // A shot that hasn't resolved in this many steps is abandoned. Launch
  // impulses top out around 45 ft/s upward against 32.2 ft/s^2, so a full
  // flight is ~170 steps; 300 is slack, and the old 600 just let the batch's
  // stragglers hold everyone up for twice as long as any real shot needs.
  maxFlightSteps: 300,
  // Flight paths are recorded for playback only, so they are subsampled and
  // stored in a preallocated Float32Array per ball. The old path recorded a
  // fresh THREE.Vector3 every step for every ball — ~600k allocations per
  // batch, all of it garbage the moment the batch ended.
  pathStride: 3,
  maxPathPoints: 110,
  // Longest edge of one vision atlas page, in pixels. batchSize tiles no
  // longer have to fit in a single render target: capture walks the batch in
  // pages of (visionAtlasMax / visionWidth)^2 tiles. At 128px tiles a 4096px
  // page holds 1024 balls, so a 2048-ball batch is captured in two pages —
  // two synchronous readback stalls per batch instead of the sixteen that
  // 64-ball pages forced, for the same total bytes transferred.
  visionAtlasMax: 4096,

  // --- Visualization --------------------------------------------------------
  // Rendering is decoupled from training: the batch is simulated headless (no
  // meshes at all) and the court instead replays the best shots the agent has
  // taken recently. Nothing here feeds back into learning.
  showcaseCount: 10,
  showcaseRefreshMs: 1000,

  groups: { court: 1, ball: 2 },
  rim: { x: 41.75, y: 10, z: 0 }
};

const UP = new THREE.Vector3(0, 1, 0);

class Assets {
  constructor() {
    this.woodTexture = Assets.woodTexture();
    this.particleTexture = Assets.softParticleTexture();
    this.ballColorMap = Assets.ballTexture();

    this.floor = new THREE.MeshPhysicalMaterial({
      map: this.woodTexture,
      color: 0x555555,
      roughness: 0.8,
      metalness: 0.1,
      clearcoat: 0.2,
      clearcoatRoughness: 0.4,
      reflectivity: 0.1
    });
    this.line = new THREE.MeshBasicMaterial({ color: 0xaaaaaa });
    this.post = new THREE.MeshStandardMaterial({
      color: 0x111111,
      roughness: 0.5
    });
    this.glass = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      transmission: 0.1,
      transparent: true,
      opacity: 0.9,
      roughness: 0.0,
      metalness: 0.1,
      reflectivity: 1.0,
      clearcoat: 1.0
    });
    this.rim = new THREE.MeshStandardMaterial({
      color: 0xff4400,
      metalness: 0.6,
      roughness: 0.2,
      emissive: 0xff2200,
      emissiveIntensity: 1.0
    });
    this.highContrast = new THREE.MeshBasicMaterial({ color: 0xffffff });
    this.manualBall = new THREE.MeshStandardMaterial({
      map: this.ballColorMap,
      color: 0xffffff,
      roughness: 0.4,
      metalness: 0.1
    });
    this.batchBall = new THREE.MeshStandardMaterial({
      color: 0xe66700,
      roughness: 0.4,
      metalness: 0.1
    });

    const trail = (color, opacity) =>
      new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
    this.trailMaterials = {
      score: trail(0x00ff00, 0.3),
      rim: trail(0xffff00, 0.3),
      miss: trail(0xff0000, 0.15)
    };

    const endpoint = (color, opacity) =>
      new THREE.SpriteMaterial({
        map: this.particleTexture,
        color,
        transparent: true,
        opacity,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
    this.endpointMaterials = {
      score: endpoint(0x00ff00, 0.8),
      rim: endpoint(0xffff00, 0.8),
      miss: endpoint(0xff0000, 0.4)
    };
  }

  static woodTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#8a6e4e";
    ctx.fillRect(0, 0, 512, 512);
    ctx.globalAlpha = 0.15;
    ctx.fillStyle = "#5c4033";
    for (let i = 0; i < 100; i++) {
      ctx.beginPath();
      ctx.ellipse(
        Math.random() * 512,
        Math.random() * 512,
        50 + Math.random() * 200,
        2 + Math.random() * 5,
        0,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
    return new THREE.CanvasTexture(canvas);
  }

  static softParticleTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext("2d");
    const grad = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
    grad.addColorStop(0, "rgba(255,255,255,1)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 32, 32);
    return new THREE.CanvasTexture(canvas);
  }

  static ballTexture() {
    const cvs = document.createElement("canvas");
    cvs.width = 512;
    cvs.height = 512;
    const ctx = cvs.getContext("2d");
    ctx.fillStyle = "#C85A17";
    ctx.fillRect(0, 0, 512, 512);
    ctx.fillStyle = "rgba(0,0,0,0.1)";
    for (let i = 0; i < 5000; i++) {
      ctx.beginPath();
      ctx.arc(
        Math.random() * 512,
        Math.random() * 512,
        1 + Math.random() * 2,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
    return new THREE.CanvasTexture(cvs);
  }
}

class CNNAgent {
  constructor({ learningRate, batchSize, l2 }) {
    this.learningRate = learningRate;
    this.batchSize = batchSize;
    this.l2Reg = tf.regularizers.l2({ l2 });
    // Input geometry, derived from CONFIG so the network and all tensor
    // reshapes track the vision resolution. Channels is 2 (stereo L/R).
    this.visionW = CONFIG.visionWidth;
    this.visionH = CONFIG.visionHeight;
    this.channels = 2;
    this.frameSize = this.visionW * this.visionH * this.channels;
    this.convCount = CONFIG.model.convs.length;
    this.actor = this._buildActor();
    this.critic = this._buildCritic();

    // Scratch for assembling a shuffled minibatch's regression targets without
    // allocating inside the training loop.
    this._mbActions = new Float32Array(CONFIG.minibatch * 3);
    this._mbWeights = new Float32Array(CONFIG.minibatch);
    this._order = new Int32Array(CONFIG.batchSize);
  }

  // Uploads a whole batch of states and leaves it resident in VRAM.
  //
  // The batch used to be walked in gpuChunk-sized pieces, and each of the four
  // passes over it (action inference, critic fit, value estimation, actor fit)
  // re-uploaded and re-widened every sample from scratch. At 1024 x 256x256x2
  // that was four traversals of 67 million elements through tf.tensor's
  // Uint8Array -> Float32Array conversion, all of it on the CPU, all of it on
  // the critical path — and the reason a 16GB card could sit at 1% while a
  // batch trained.
  //
  // A 2048-sample batch at 128px is 268MB as float32, which the card this is
  // aimed at has in abundance. So it goes up once, right after capture, and
  // stays up until the batch is trained on and retired: one conversion and one
  // upload per batch instead of four, and every subsequent pass is a GPU-side
  // slice or gather off a tensor that is already there.
  //
  // `state` is normalized to [0, 1] at capture time (VisionSystem._readPage),
  // so there is no scaling pass here either. The caller owns the buffer and
  // must not overwrite it until the returned tensor is disposed.
  uploadBatch(state, count) {
    return tf.tensor(
      state.subarray(0, count * this.frameSize),
      [count, this.visionH, this.visionW, this.channels],
      "float32"
    );
  }

  // Shared conv stack, built from CONFIG.model: convs... -> flatten -> dense.
  // Layer indices [0, convCount) are the conv layers in BOTH nets — the weight
  // sync in trainOnBatch() depends on that alignment, as does the freeze list
  // and getActivations()' hand-rolled forward pass, so all three read
  // this.convCount rather than assuming a fixed depth.
  //
  // denseReg is null for the actor: the dense layers are the only trainable
  // parts of the policy, and L2 on them pulls the policy toward constant zero
  // output — i.e. the same shot from every position — which competes directly
  // with an already-weak improvement signal. The critic keeps L2 everywhere,
  // and since the critic owns the conv backbone the filters stay regularized
  // regardless.
  _convBase(model, denseReg = this.l2Reg) {
    CONFIG.model.convs.forEach((c, i) => {
      model.add(
        tf.layers.conv2d({
          // Only the first layer declares the input shape; the rest infer it.
          ...(i === 0
            ? { inputShape: [this.visionH, this.visionW, this.channels] }
            : {}),
          filters: c.filters,
          kernelSize: c.kernelSize,
          strides: c.strides,
          activation: "relu",
          kernelRegularizer: this.l2Reg
        })
      );
    });
    model.add(tf.layers.flatten());
    model.add(
      tf.layers.dense({
        units: CONFIG.model.dense,
        activation: "relu",
        kernelRegularizer: denseReg
      })
    );
  }

  _buildActor() {
    const m = tf.sequential();
    this._convBase(m, null);
    // The critic owns the shared conv backbone: trainOnBatch() copies the
    // critic's conv weights into the actor after every batch. Freeze the
    // actor's conv layers so the actor update doesn't waste work computing
    // gradients that get overwritten, and so the actor's dense head trains
    // against a stable feature extractor.
    for (let i = 0; i < this.convCount; i++) m.layers[i].trainable = false;
    m.add(tf.layers.dense({ units: 3, activation: "tanh" }));
    // Compiled so the model stays serializable for EXPORT POLICY, but the actor
    // is not trained through fit() — see _fitActorWeighted, which needs a
    // per-sample weighted loss that fit() cannot express in this tfjs version.
    m.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: "meanSquaredError"
    });
    this.actorOptimizer = tf.train.adam(this.learningRate);
    return m;
  }

  _buildCritic() {
    const m = tf.sequential();
    this._convBase(m);
    m.add(tf.layers.dense({ units: 1, kernelRegularizer: this.l2Reg }));
    m.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: "meanSquaredError"
    });
    return m;
  }

  // x: the resident batch tensor from uploadBatch(), [count, H, W, 2].
  // Writes count * 3 action components into `out` (a Float32Array). Still
  // chunked, but the chunks are GPU-side slices of a tensor that is already
  // uploaded rather than fresh uploads, so this pass no longer touches the CPU
  // state buffer at all.
  //
  // Readback is await .data() rather than .dataSync(): dataSync() blocks the
  // main thread until the GPU drains, which with gpuChunk-sized chunks meant a
  // full pipeline stall per chunk and left the card idle between them. The
  // async form lets the next chunk's work be enqueued while the previous
  // chunk's result is still in flight.
  async predictBatch(x, count, out, noiseScale = 0) {
    // Clamp to [-1, 1] after adding exploration noise: the actor's output
    // is tanh-bounded, and these actions are later stored as regression
    // targets, so out-of-range values would be unreachable training goals.
    const clamp = (v) => Math.max(-1, Math.min(1, v));
    const chunk = CONFIG.gpuChunk;
    for (let start = 0; start < count; start += chunk) {
      const n = Math.min(chunk, count - start);
      const pred = tf.tidy(() =>
        this.actor.predict(x.slice([start, 0, 0, 0], [n, -1, -1, -1]))
      );
      const data = await pred.data(); // n * 3
      pred.dispose();
      for (let i = 0; i < n * 3; i++)
        out[start * 3 + i] = clamp(
          data[i] + (Math.random() - 0.5) * noiseScale
        );
    }
    return out;
  }

  // Activations for a single sample, addressed by its index in the batch.
  // Walks the actor by hand: convs, flatten, dense, then the output head. The
  // indices come from convCount so the walk survives a change to the conv
  // stack — layers are [0, convCount) convs, [convCount] flatten,
  // [convCount + 1] dense, [convCount + 2] output.
  getActivations(x, index) {
    return tf.tidy(() => {
      let t = x.slice([index, 0, 0, 0], [1, -1, -1, -1]);
      const n = this.convCount;
      for (let i = 0; i <= n + 1; i++) t = this.actor.layers[i].apply(t);
      const dense = t;
      const output = this.actor.layers[n + 2].apply(dense);
      return { dense: dense.dataSync(), output: output.dataSync() };
    });
  }

  // Trains critic on returns, then the actor on advantage-weighted actions.
  // Returns the critic loss (or null if nothing to train on).
  //
  // x is the resident batch tensor; actions/rewards describe the same batch.
  // Nothing is retained afterwards — the caller owns x and disposes it.
  //
  // All three passes now run against that one resident tensor. The previous
  // version walked the batch in gpuChunk pieces three separate times, uploading
  // and float-widening every sample once per pass, because holding the batch on
  // the GPU was assumed to be unaffordable. On a card with headroom it is the
  // cheap option: one upload, three GPU-side passes, and the per-sample
  // bookkeeping (values, advantages, weights) still lives in plain typed arrays
  // on the CPU where it costs 4 bytes a sample.
  async trainOnBatch(x, actions, rewards, count) {
    if (count === 0) return null;
    const mb = CONFIG.minibatch;
    const epochs = CONFIG.trainEpochs;

    // --- Pass 1: critic regression onto realized reward ---------------------
    // fit() rather than a hand-rolled minimize() so the L2 regularizer losses
    // and the Adam slot state keep behaving as they did. It now sees the whole
    // batch in one call, which lets it shuffle across the batch instead of
    // marching through gpuChunk-sized windows in capture order, and lets a
    // second epoch reuse the resident tensor for free.
    let loss = 0;
    const yb = tf.tensor2d(rewards.subarray(0, count), [count, 1]);
    try {
      const history = await this.critic.fit(x, yb, {
        epochs,
        batchSize: mb,
        shuffle: true,
        verbose: 0
      });
      const ls = history.history.loss;
      // Report the last epoch's loss: it is the one that describes the critic
      // the actor update below is about to be scored against.
      if (ls && ls.length) loss = ls[ls.length - 1];
    } finally {
      yb.dispose();
    }

    // --- Pass 2: value estimates for the whole batch ------------------------
    // Read back as a CPU Float32Array so the advantage statistics below cost
    // nothing on the GPU. predict()'s own batchSize bounds the activation
    // memory of the forward pass without re-slicing the input by hand.
    const vT = this.critic.predict(x, { batchSize: CONFIG.gpuChunk });
    const values = await vT.data();
    vT.dispose();

    // Actor: regress toward the actions taken, weighted by how far each beat
    // the critic's value estimate.
    //
    // This replaces a hard `advantage > 0` filter that trained on the better
    // half of the batch with every surviving sample weighted equally. That
    // filter throws away the magnitude of the advantage entirely, so a shot
    // that beat the baseline by a hair pulled exactly as hard as one that swept
    // the net — the regression target collapsed toward the unweighted mean of
    // roughly half the actions, which is close to the mean of all of them. The
    // result was a policy that barely varied with the input image. Exponential
    // weighting keeps every sample but lets the genuinely good ones dominate.
    let advMean = 0;
    for (let i = 0; i < count; i++) advMean += rewards[i] - values[i];
    advMean /= count;
    let advVar = 0;
    for (let i = 0; i < count; i++)
      advVar += (rewards[i] - values[i] - advMean) ** 2;
    const advStd = Math.sqrt(advVar / count) || 1;

    const weightData = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const z = (rewards[i] - values[i] - advMean) / advStd;
      weightData[i] = Math.min(
        CONFIG.advantageClip,
        Math.exp(z / CONFIG.advantageTemp)
      );
    }

    // Sync the shared conv layers (critic -> actor) BEFORE fitting the actor.
    // The actor's conv layers are frozen, so the actor update only adjusts its
    // dense head — and that head should be trained against the same features
    // the actor will use at inference. Syncing after the fit would instead
    // train the head on the previous batch's stale features, then swap the
    // backbone out from under it.
    tf.tidy(() => {
      for (let i = 0; i < this.convCount; i++)
        this.actor.layers[i].setWeights(this.critic.layers[i].getWeights());
    });

    // --- Pass 3: advantage-weighted actor update ----------------------------
    this._fitActorWeighted(x, actions, weightData, count);

    return loss;
  }

  // Minibatch Adam over a per-sample weighted MSE, across the whole batch.
  // model.fit() cannot do this: passing sampleWeight throws "Support
  // sampleWeight is not implemented yet" in tfjs 4.x, so the update is driven
  // explicitly. Only trainableWeights are passed to minimize(), which already
  // excludes the frozen conv layers.
  //
  // Minibatches are gathered from the resident batch tensor on the GPU, so
  // shuffling costs a gather rather than a re-upload. The old version could
  // only shuffle *within* a gpuChunk window — in practice it did not shuffle at
  // all — which meant every epoch presented the samples in the same order and
  // consecutive optimizer steps saw correlated slices of the batch.
  _fitActorWeighted(x, actions, weights, count) {
    const vars = this.actor.trainableWeights.map((w) => w.val);
    const mb = CONFIG.minibatch;
    const order = this._order;
    for (let i = 0; i < count; i++) order[i] = i;

    for (let e = 0; e < CONFIG.trainEpochs; e++) {
      // Fisher-Yates over the index array, reused between epochs.
      for (let i = count - 1; i > 0; i--) {
        const j = (Math.random() * (i + 1)) | 0;
        const t = order[i];
        order[i] = order[j];
        order[j] = t;
      }

      for (let off = 0; off < count; off += mb) {
        const m = Math.min(mb, count - off);
        // Gather this minibatch's targets and weights into contiguous scratch
        // so the tensors below are built from a flat copy rather than a
        // scattered read per element.
        for (let i = 0; i < m; i++) {
          const src = order[off + i];
          this._mbActions[i * 3] = actions[src * 3];
          this._mbActions[i * 3 + 1] = actions[src * 3 + 1];
          this._mbActions[i * 3 + 2] = actions[src * 3 + 2];
          this._mbWeights[i] = weights[src];
        }
        tf.tidy(() => {
          const idx = tf.tensor1d(order.subarray(off, off + m), "int32");
          const xb = tf.gather(x, idx);
          const yb = tf.tensor2d(this._mbActions.subarray(0, m * 3), [m, 3]);
          const wb = tf.tensor1d(this._mbWeights.subarray(0, m));
          this.actorOptimizer.minimize(
            () => {
              const pred = this.actor.apply(xb, { training: true });
              const perSample = pred.sub(yb).square().mean(1);
              return perSample.mul(wb).mean();
            },
            false,
            vars
          );
        });
      }
    }
  }

  async saveActor() {
    await this.actor.save("downloads://basketball-agent-actor");
  }
}

class SceneManager {
  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x222222);
    this.scene.fog = new THREE.Fog(0x222222, 50, 150);

    this.camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.01,
      10000
    );
    this.camera.position.set(0, 10, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    document.body.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.05;
    this.controls.target.set(CONFIG.rim.x, CONFIG.rim.y, CONFIG.rim.z);
    this.controls.update();

    this._addLights();

    window.addEventListener("resize", () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  _addLights() {
    const ambient = new THREE.AmbientLight(0xffffff, 0.2);

    const dir = new THREE.DirectionalLight(0xffffff, 0.5);
    dir.position.set(20, 50, 20);
    dir.castShadow = true;
    dir.shadow.mapSize.width = 2048;
    dir.shadow.mapSize.height = 2048;

    const hoop = new THREE.SpotLight(0xffffff, 2.0);
    hoop.position.set(25, 25, 0);
    hoop.target.position.set(CONFIG.rim.x, CONFIG.rim.y, CONFIG.rim.z);
    hoop.angle = Math.PI / 6;
    hoop.penumbra = 0.2;
    hoop.castShadow = true;

    this.scene.add(ambient, dir, hoop, hoop.target);
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

class PhysicsWorld {
  constructor() {
    this.world = new CANNON.World();
    this.world.gravity.set(0, -32.2, 0);
    // SAPBroadphase (sweep-and-prune) instead of NaiveBroadphase: with
    // CONFIG.batchSize balls in the world, naive O(n^2) pair enumeration
    // dominates each step. Balls never collide with each other (their mask
    // is court-only), so sweep-and-prune's sorted-axis pruning is a pure win.
    this.world.broadphase = new CANNON.SAPBroadphase(this.world);
    this.world.solver.iterations = 10;
    // Finished balls are put to sleep instead of being left to bounce and roll.
    // Without this the world kept integrating and colliding every ball that had
    // already been scored and banked for the rest of the batch — dead weight
    // that grew as the batch drained, which is precisely when the remaining
    // live shots most need the step budget.
    this.world.allowSleep = true;

    this.concrete = new CANNON.Material("concrete");
    this.plastic = new CANNON.Material("plastic");
    this.world.addContactMaterial(
      new CANNON.ContactMaterial(this.concrete, this.plastic, {
        friction: 0.5,
        restitution: 0.8
      })
    );
  }

  add(body) {
    this.world.addBody(body);
  }

  step() {
    this.world.step(1 / 60);
  }

  onBeginContact(cb) {
    this.world.addEventListener("beginContact", cb);
  }
}

class Court {
  constructor(scene, physics, assets) {
    this.scene = scene;
    this.physics = physics;
    this.assets = assets;

    this.rimPosition = new THREE.Vector3(
      CONFIG.rim.x,
      CONFIG.rim.y,
      CONFIG.rim.z
    );
    this.rimPositionCannon = new CANNON.Vec3(
      CONFIG.rim.x,
      CONFIG.rim.y,
      CONFIG.rim.z
    );

    // Right-hoop bodies used for scoring / reward classification.
    this.rimMesh = null;
    this.rimBody = null;
    this.backboardBody = null;
    this.scoringSensor = null;

    this._buildFloor();

    this.group = new THREE.Group();
    scene.add(this.group);
    this.group.add(this._line(94.17, 0.17, 0, -25));
    this.group.add(this._line(94.17, 0.17, 0, 25));
    this.group.add(this._line(0.17, 50, -47, 0));
    this.group.add(this._line(0.17, 50, 47, 0));
    this.group.add(this._line(0.17, 50, 0, 0));

    scene.add(this._buildHoop(true));
    scene.add(this._buildHoop(false));
  }

  _buildFloor() {
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(94, 50),
      this.assets.floor
    );
    mesh.rotation.x = -Math.PI / 2;
    mesh.receiveShadow = true;
    this.scene.add(mesh);

    const body = new CANNON.Body({ mass: 0, material: this.physics.concrete });
    body.addShape(new CANNON.Plane());
    body.quaternion.setFromAxisAngle(new CANNON.Vec3(1, 0, 0), -Math.PI / 2);
    body.collisionFilterGroup = CONFIG.groups.court;
    body.collisionFilterMask = CONFIG.groups.ball | CONFIG.groups.court;
    this.physics.add(body);
  }

  _line(w, h, x, z) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), this.assets.line);
    m.rotation.x = -Math.PI / 2;
    m.position.set(x, 0.02, z);
    return m;
  }

  _staticBody(x, y, z, shape) {
    const body = new CANNON.Body({ mass: 0, material: this.physics.concrete });
    if (shape) body.addShape(shape);
    body.position.set(x, y, z);
    body.collisionFilterGroup = CONFIG.groups.court;
    body.collisionFilterMask = CONFIG.groups.ball | CONFIG.groups.court;
    return body;
  }

  _buildHoop(isLeft) {
    const group = new THREE.Group();
    const sign = isLeft ? -1 : 1;
    const baseX = sign * 47;
    const boardX = baseX - sign * 4;
    const rimX = boardX - sign * 1.25;

    const pole = new THREE.Mesh(
      new THREE.BoxGeometry(1, 12, 1),
      this.assets.post
    );
    pole.position.set(baseX + sign * 6, 6, 0);
    group.add(pole);

    const board = new THREE.Mesh(
      new THREE.BoxGeometry(0.1, 3.5, 6),
      this.assets.glass
    );
    board.position.set(boardX, 10.75, 0);
    group.add(board);

    const rim = new THREE.Mesh(
      new THREE.TorusGeometry(0.75, 0.05, 16, 100),
      this.assets.rim
    );
    rim.rotation.x = Math.PI / 2;
    rim.position.set(rimX, 10, 0);
    group.add(rim);
    if (!isLeft) this.rimMesh = rim;

    const poleBody = this._staticBody(
      baseX + sign * 6,
      6,
      0,
      new CANNON.Box(new CANNON.Vec3(0.5, 6, 0.5))
    );
    this.physics.add(poleBody);

    const boardBody = this._staticBody(
      boardX,
      10.75,
      0,
      new CANNON.Box(new CANNON.Vec3(0.05, 1.75, 3))
    );
    this.physics.add(boardBody);
    if (!isLeft) this.backboardBody = boardBody;

    const rimBody = this._staticBody(rimX, 10, 0, null);
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2;
      rimBody.addShape(
        new CANNON.Sphere(0.05),
        new CANNON.Vec3(Math.cos(a) * 0.75, 0, Math.sin(a) * 0.75)
      );
    }
    this.physics.add(rimBody);
    if (!isLeft) this.rimBody = rimBody;

    const sensorBody = new CANNON.Body({
      mass: 0,
      isTrigger: true,
      shape: new CANNON.Cylinder(0.5, 0.5, 0.5, 8)
    });
    sensorBody.position.set(rimX, 9.5, 0);
    sensorBody.collisionFilterGroup = CONFIG.groups.court;
    sensorBody.collisionFilterMask = CONFIG.groups.ball | CONFIG.groups.court;
    this.physics.add(sensorBody);
    if (!isLeft) this.scoringSensor = sensorBody;

    return group;
  }

  setHighContrast(on) {
    if (this.rimMesh)
      this.rimMesh.material = on ? this.assets.highContrast : this.assets.rim;
  }
}

// A training ball is pure physics — no mesh, no material, nothing in the
// render graph. The batch is simulated headless and never drawn; what the user
// sees is the Showcase replaying recorded flight paths. Carrying 1024 shadow-
// casting spheres through every frame cost more render time than the entire
// learning step, and none of it was legible anyway at that density.
class Ball {
  constructor(id, physics, radius) {
    this.id = id;

    // Flight path, subsampled and preallocated: [x,y,z] triples, pathLen of
    // maxPathPoints used. Only ever read back for playback of the best shots.
    this.path = new Float32Array(CONFIG.maxPathPoints * 3);
    this.pathLen = 0;

    this.body = new CANNON.Body({
      mass: 2,
      shape: new CANNON.Sphere(radius),
      material: physics.plastic,
      linearDamping: 0.1,
      angularDamping: 0.1
    });
    this.body.collisionFilterGroup = CONFIG.groups.ball;
    this.body.collisionFilterMask = CONFIG.groups.court;
    physics.add(this.body);

    this._resetState();
  }

  _resetState() {
    this.active = false;
    this.pathLen = 0;
    this.steps = 0;
    this.minDist = 100;
    this.scored = false;
    this.hitBackboard = false;
    this.hitRim = false;
  }

  // Position at a random launch spot and mark active.
  spawn() {
    const x = Math.random() * 42;
    const z = (Math.random() - 0.5) * 46;
    const y = 4.5 + Math.random() * 1.5;
    this.body.position.set(x, y, z);
    this.body.velocity.set(0, 0, 0);
    this.body.angularVelocity.set(0, 0, 0);
    this.body.sleep();

    this._resetState();
    this.active = true;
    this.record();
  }

  // Append the current position to the flight path. Once the path is full the
  // last slot is overwritten instead of dropping the sample, so a long shot's
  // recorded endpoint is still where the ball actually ended up.
  record() {
    const p = this.body.position;
    const i = this.pathLen < CONFIG.maxPathPoints ? this.pathLen++ : this.pathLen - 1;
    const o = i * 3;
    this.path[o] = p.x;
    this.path[o + 1] = p.y;
    this.path[o + 2] = p.z;
  }

  // Apply the launch impulse derived from the agent's 3-vector action, read
  // from `actions` at `offset` (a flat batch-wide Float32Array).
  launch(actions, offset, hoopPos) {
    const p = this.body.position;
    const dirToHoop = new THREE.Vector3(
      hoopPos.x - p.x,
      0,
      hoopPos.z - p.z
    ).normalize();
    const dirSide = new THREE.Vector3().crossVectors(UP, dirToHoop).normalize();

    const magFwd = (actions[offset] + 1) * 50;
    const magUp = (actions[offset + 1] + 1) * 35 + 20;
    const magSide = actions[offset + 2] * 20;

    this.body.wakeUp();
    this.body.applyImpulse(
      new CANNON.Vec3(
        dirToHoop.x * magFwd + dirSide.x * magSide,
        magUp,
        dirToHoop.z * magFwd + dirSide.z * magSide
      ),
      new CANNON.Vec3(0, 0, 0)
    );
  }

  // Stop simulating: zero the motion and sleep the body so the world stops
  // spending broadphase and solver time on a shot that is already scored.
  retire() {
    this.active = false;
    this.body.velocity.set(0, 0, 0);
    this.body.angularVelocity.set(0, 0, 0);
    this.body.sleep();
  }

  onContact(other, court) {
    if (other === court.scoringSensor && this.body.velocity.y < 0)
      this.scored = true;
    else if (other === court.backboardBody) this.hitBackboard = true;
    else if (other === court.rimBody) this.hitRim = true;
  }
}

class VisionSystem {
  constructor(renderer, scene, batchSize) {
    this.renderer = renderer;
    this.scene = scene;
    const W = CONFIG.visionWidth;
    const H = CONFIG.visionHeight;
    this.W = W;
    this.H = H;
    this.frameSize = W * H * 2;

    // Atlas layout. Each ball's per-eye WxH view is a tile in a cols x rows
    // grid packed into one render target per eye, so a whole page of balls is
    // read back with a single readRenderTargetPixels() per eye rather than the
    // two GPU->CPU pipeline flushes per ball the original path needed.
    //
    // The atlas is now PAGED rather than sized to hold the entire batch. Fitting
    // 1024 tiles of 256px meant an 8192x8192 target — at the texture-size limit
    // of a lot of hardware, and 268MB of readback buffer per eye. Capacity is
    // capped at visionAtlasMax on a side and captureBatch walks the batch one
    // page at a time, which keeps the buffers at a fixed ~16MB per eye no
    // matter how large the batch grows.
    const maxTiles = Math.max(
      1,
      Math.floor(CONFIG.visionAtlasMax / W) * Math.floor(CONFIG.visionAtlasMax / H)
    );
    this.pageSize = Math.min(batchSize, maxTiles);
    this.cols = Math.min(this.pageSize, Math.max(1, Math.floor(CONFIG.visionAtlasMax / W)));
    this.rows = Math.ceil(this.pageSize / this.cols);
    const atlasW = this.cols * W;
    const atlasH = this.rows * H;
    this.atlasW = atlasW;
    this.atlasH = atlasH;

    this.rtLeft = new THREE.WebGLRenderTarget(atlasW, atlasH);
    this.rtRight = new THREE.WebGLRenderTarget(atlasW, atlasH);
    // Tiling lives on the render targets, not on the renderer. setViewport()/
    // setScissor()/setScissorTest() write renderer globals that setRenderTarget()
    // immediately overwrites from the bound target — and WebGLShadowMap.render()
    // calls setRenderTarget() itself to rebind after drawing shadow maps, which
    // happens inside every render(). Via the renderer API the tile rect was
    // therefore dropped before the scene drew: each ball cleared and repainted
    // the whole atlas, leaving every tile a crop of the *last* ball's view.
    this.rtLeft.scissorTest = true;
    this.rtRight.scissorTest = true;
    // Aspect is W/H rather than a hardcoded 1 so a non-square retina doesn't
    // silently stretch the view. Near plane sits just outside the ball so the
    // agent never sees the inside of its own skin.
    const aspect = W / H;
    this.camLeft = new THREE.PerspectiveCamera(CONFIG.visionFov, aspect, 0.1, 200);
    this.camRight = new THREE.PerspectiveCamera(CONFIG.visionFov, aspect, 0.1, 200);
    this.bufLeft = new Uint8Array(atlasW * atlasH * 4);
    this.bufRight = new Uint8Array(atlasW * atlasH * 4);
    this._eye = new THREE.Vector3();
    this._right = new THREE.Vector3();
  }

  // Renders each ball's stereo view toward the hoop and packs grayscale L/R
  // samples into `out`, a caller-owned Float32Array of balls.length*frameSize.
  //
  // Output is normalized luma in [0, 1] rather than 8-bit. Bytes were the right
  // call when every training pass re-uploaded the batch and the divide happened
  // on the GPU; now the batch is uploaded once and lives in VRAM, so writing
  // floats straight out of the unpack loop removes an entire CPU-side traversal
  // of the batch (tf.tensor's Uint8Array -> Float32Array widening) at the cost
  // of one extra multiply in a loop that was already touching every pixel.
  //
  // `hidden` lists objects that must not appear in the agents' view.
  captureBatch(balls, hoopPos, court, hidden, out) {
    const shadows = this.renderer.shadowMap;

    // Shadow maps are the single biggest capture cost and contribute nothing:
    // the scene is switched to high contrast precisely so the agent sees hoop
    // geometry rather than lighting. Every renderer.render() otherwise re-runs
    // both shadow-casting lights, so a 1024-ball batch paid 2048 shadow passes
    // on a completely static scene. Suspending autoUpdate skips the re-render
    // while leaving the shader variants (and therefore the compiled programs)
    // untouched — toggling shadowMap.enabled instead would force a full
    // material recompile twice per batch.
    const shadowAuto = shadows.autoUpdate;
    shadows.autoUpdate = false;

    const wasVisible = hidden.map((o) => o.visible);
    for (const o of hidden) o.visible = false;
    court.setHighContrast(true);

    // Nothing in the scene moves between the renders below — only the two
    // cameras do, and cameras update their own matrices. WebGLRenderer.render()
    // otherwise walks the entire scene graph recomputing world matrices on
    // every call, which across two renders per ball is thousands of full-graph
    // traversals per batch spent confirming that a static court has not moved.
    // Update once up front, then suspend it for the duration of the capture.
    const sceneAuto = this.scene.autoUpdate;
    this.scene.updateMatrixWorld();
    this.scene.autoUpdate = false;
    // Depth-sorting a fixed opaque scene into the same order 4096 times running
    // is pure overhead; the draw order does not affect what the agent sees.
    const sortObjects = this.renderer.sortObjects;
    this.renderer.sortObjects = false;

    for (let start = 0; start < balls.length; start += this.pageSize) {
      const n = Math.min(this.pageSize, balls.length - start);
      this._renderPage(balls, start, n, hoopPos);
      this._readPage(out, start, n);
    }

    // Restore.
    this.renderer.sortObjects = sortObjects;
    this.scene.autoUpdate = sceneAuto;
    court.setHighContrast(false);
    for (let i = 0; i < hidden.length; i++) hidden[i].visible = wasVisible[i];
    // Unbinding restores the canvas viewport/scissor from the renderer globals,
    // which this path never touches, so there is nothing else to put back.
    this.renderer.setRenderTarget(null);
    shadows.autoUpdate = shadowAuto;
    shadows.needsUpdate = shadowAuto;
  }

  // Draws balls [start, start+n) into one atlas page, one tile each per eye.
  _renderPage(balls, start, n, hoopPos) {
    const { W, H, cols } = this;
    const half = CONFIG.ipd / 2;

    for (let i = 0; i < n; i++) {
      const pos = balls[start + i].body.position;
      this._eye.set(
        hoopPos.x - pos.x,
        hoopPos.y - pos.y,
        hoopPos.z - pos.z
      ).normalize();
      this._right.crossVectors(this._eye, UP).normalize().multiplyScalar(half);

      this.camLeft.position.set(
        pos.x - this._right.x,
        pos.y - this._right.y,
        pos.z - this._right.z
      );
      this.camRight.position.set(
        pos.x + this._right.x,
        pos.y + this._right.y,
        pos.z + this._right.z
      );
      this.camLeft.lookAt(hoopPos);
      this.camRight.lookAt(hoopPos);

      const tileX = (i % cols) * W;
      const tileY = Math.floor(i / cols) * H;

      // Confine each ball's render to its own tile. The scissor test keeps both
      // the background clear and the draw inside the tile, so neighbouring tiles
      // survive across the loop and the atlas fills up one ball at a time. The
      // rect has to be set on the target *before* binding it — setRenderTarget()
      // is what copies it into the active GL state.
      this.rtLeft.viewport.set(tileX, tileY, W, H);
      this.rtLeft.scissor.set(tileX, tileY, W, H);
      this.renderer.setRenderTarget(this.rtLeft);
      this.renderer.render(this.scene, this.camLeft);

      this.rtRight.viewport.set(tileX, tileY, W, H);
      this.rtRight.scissor.set(tileX, tileY, W, H);
      this.renderer.setRenderTarget(this.rtRight);
      this.renderer.render(this.scene, this.camRight);
    }
  }

  // One readback per eye covers the whole page; unpack the tiles into `out`.
  _readPage(out, start, n) {
    const { W, H, frameSize, cols, atlasW } = this;
    const bl = this.bufLeft;
    const br = this.bufRight;

    // Only the rows this page actually filled. A trailing partial page (batch
    // size not a multiple of page size) otherwise pays a full-atlas readback
    // for a handful of tiles. Reading full width keeps the row stride equal to
    // atlasW, which is what the unpack indexing below assumes.
    const usedH = Math.ceil(n / cols) * H;

    this.renderer.readRenderTargetPixels(this.rtLeft, 0, 0, atlasW, usedH, bl);
    this.renderer.readRenderTargetPixels(this.rtRight, 0, 0, atlasW, usedH, br);

    // Integer luma: 77/150/29 are the Rec.601 weights scaled by 256. The single
    // trailing multiply by 1/(256*255) undoes that scale and normalizes to
    // [0, 1] in one step, so the loop still does three integer multiplies and
    // one float multiply per pixel per eye — and the batch arrives at the GPU
    // already in the range the network wants, with no separate conversion or
    // divide pass over it. At 128x128x2048 this inner loop runs 33M times a
    // batch, down from 67M at the old resolution.
    const S = 1 / (256 * 255);
    for (let i = 0; i < n; i++) {
      const tileX = (i % cols) * W;
      const tileY = Math.floor(i / cols) * H;
      let o = (start + i) * frameSize;
      for (let y = 0; y < H; y++) {
        let a = ((tileY + y) * atlasW + tileX) * 4;
        for (let x = 0; x < W; x++) {
          out[o] = (77 * bl[a] + 150 * bl[a + 1] + 29 * bl[a + 2]) * S;
          out[o + 1] = (77 * br[a] + 150 * br[a + 1] + 29 * br[a + 2]) * S;
          o += 2;
          a += 4;
        }
      }
    }
  }
}

class TrajectoryTrails {
  constructor(scene, assets, maxHistory) {
    this.assets = assets;
    this.max = maxHistory;
    this.history = [];
    this.group = new THREE.Group();
    scene.add(this.group);
  }

  // path: flat Float32Array of xyz triples, `len` points used. Taking the raw
  // buffer instead of an array of THREE.Vector3 lets a recorded flight go
  // straight into a BufferAttribute with no intermediate objects.
  add(path, len, type, reward) {
    if (len < 2) return;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(path.subarray(0, len * 3), 3)
    );
    const material = this.assets.trailMaterials[type].clone();
    let opacity = 0.05;
    if (type === "score") {
      const t = Math.max(0, Math.min(1, (reward - 10) / 25));
      opacity = 0.3 + t * 0.7;
    } else if (type === "rim") {
      opacity = 0.2;
    }
    material.opacity = opacity;
    const line = new THREE.Line(geometry, material);
    this.group.add(line);

    const spriteMat = this.assets.endpointMaterials[type].clone();
    spriteMat.opacity = Math.min(1.0, opacity + 0.2);
    const sprite = new THREE.Sprite(spriteMat);
    const last = (len - 1) * 3;
    sprite.position.set(path[last], path[last + 1], path[last + 2]);
    sprite.scale.set(1.5, 1.5, 1.5);
    this.group.add(sprite);

    this.history.push({ line, sprite });
    if (this.history.length > this.max) {
      const old = this.history.shift();
      this.group.remove(old.line);
      this.group.remove(old.sprite);
      old.line.geometry.dispose();
      old.line.material.dispose();
      old.sprite.material.dispose();
    }
  }
}

// The visible court, decoupled from training.
//
// Training runs headless and as fast as the machine allows — batches can now
// resolve several times a second, which is far past the rate at which watching
// 1024 simultaneous balls tells anyone anything. So the arena doesn't draw
// itself. It offers every finished shot here, the Showcase keeps only the best
// handful, and once a second it promotes them to a reel: their trails are laid
// down and a small pool of real meshes replays the flights at normal speed.
//
// Nothing in this class feeds back into learning. Rendering can stall, skip, or
// be throttled to zero and the agent's throughput is unaffected.
class Showcase {
  constructor(scene, assets, trails, { count, refreshMs }) {
    this.count = count;
    this.refreshMs = refreshMs;
    this.trails = trails;

    this.group = new THREE.Group();
    scene.add(this.group);

    const geometry = new THREE.SphereGeometry(CONFIG.ballRadius, 16, 16);
    this.meshes = [];
    for (let i = 0; i < count; i++) {
      const m = new THREE.Mesh(geometry, assets.batchBall);
      m.castShadow = true;
      m.visible = false;
      this.group.add(m);
      this.meshes.push(m);
    }

    // Shots seen since the last promotion, best-first; capped at `count`.
    this.candidates = [];
    this.reel = [];
    this.cursor = 0;
    this.reelPoints = 0;
    this.lastRefresh = 0;
    this.bestReward = -Infinity;
  }

  // Offer a finished shot. Copies the path only if the shot is good enough to
  // make the current top `count`, so a 1024-ball batch performs ~10 copies
  // rather than 1024.
  offer(ball, reward, type) {
    if (ball.pathLen < 2) return;
    if (
      this.candidates.length >= this.count &&
      reward <= this.candidates[this.candidates.length - 1].reward
    )
      return;

    const shot = {
      path: ball.path.slice(0, ball.pathLen * 3),
      len: ball.pathLen,
      reward,
      type
    };
    let i = this.candidates.length;
    while (i > 0 && this.candidates[i - 1].reward < reward) i--;
    this.candidates.splice(i, 0, shot);
    if (this.candidates.length > this.count) this.candidates.length = this.count;
  }

  // Promote the accumulated best shots to the visible reel, at most once per
  // refreshMs. Returns true if the reel changed.
  refresh(now) {
    if (now - this.lastRefresh < this.refreshMs) return false;
    this.lastRefresh = now;
    if (this.candidates.length === 0) return false;

    this.reel = this.candidates;
    this.candidates = [];
    this.cursor = 0;
    this.reelPoints = 0;
    this.bestReward = this.reel[0].reward;
    for (const s of this.reel)
      if (s.len > this.reelPoints) this.reelPoints = s.len;

    // The trail history is bounded at CONFIG.maxHistory, so it now holds the
    // last maxHistory *showcased* shots — roughly ten seconds of the agent's
    // best work — instead of being flooded and evicted 1024 entries at a time
    // by a single batch.
    for (const s of this.reel) this.trails.add(s.path, s.len, s.type, s.reward);
    return true;
  }

  // Advance playback. Paths were sampled every CONFIG.pathStride physics steps
  // at 60Hz, so replaying that many points per second per stride puts the
  // flights back at real speed regardless of how fast training is running.
  update(dt) {
    if (this.reel.length === 0) return;
    this.cursor += (dt * 60) / CONFIG.pathStride;
    if (this.cursor >= this.reelPoints) this.cursor = 0; // loop until refresh

    const c = this.cursor;
    const i0 = Math.floor(c);
    const frac = c - i0;
    for (let i = 0; i < this.meshes.length; i++) {
      const mesh = this.meshes[i];
      const shot = this.reel[i];
      if (!shot || i0 >= shot.len) {
        mesh.visible = false;
        continue;
      }
      const a = i0 * 3;
      const b = i0 + 1 < shot.len ? a + 3 : a;
      mesh.position.set(
        shot.path[a] + (shot.path[b] - shot.path[a]) * frac,
        shot.path[a + 1] + (shot.path[b + 1] - shot.path[a + 1]) * frac,
        shot.path[a + 2] + (shot.path[b + 2] - shot.path[a + 2]) * frac
      );
      mesh.visible = true;
    }
  }

  hideAll() {
    for (const m of this.meshes) m.visible = false;
  }
}

class Dashboard {
  constructor() {
    // Size the stereo-view canvas from CONFIG (L|R side by side) so it tracks
    // the vision resolution instead of assuming a fixed 128x64.
    this.agentViewCanvas = document.getElementById("agent-view-canvas");
    this.agentViewCanvas.width = CONFIG.visionWidth * 2;
    this.agentViewCanvas.height = CONFIG.visionHeight;
    this.agentViewCtx = this.agentViewCanvas.getContext("2d");
    this.kernelCanvas = document.getElementById("kernel-canvas");
    this.kernelCtx = this.kernelCanvas.getContext("2d");
    // Nearest-neighbour, not bilinear. An 8x8 kernel blown up to tile size with
    // smoothing on turns every filter into a soft blob, which reads as "the
    // filters never learned anything" no matter what the weights actually are.
    this.kernelCtx.imageSmoothingEnabled = false;
    this.actCanvas = document.getElementById("act-canvas");
    this.actCtx = this.actCanvas.getContext("2d");
    this.lossCanvas = document.getElementById("loss-canvas");
    this.lossCtx = this.lossCanvas.getContext("2d");

    this.tfStatus = document.getElementById("tf-backend-status");
    this.uiEp = document.getElementById("ep-count");
    this.uiBaskets = document.getElementById("baskets");
    this.uiReward = document.getElementById("last-reward");
    this.uiAcc = document.getElementById("accuracy");
    this.uiStatus = document.getElementById("status");
    this.uiBatch = document.getElementById("batch-progress");
    this.uiRate = document.getElementById("shot-rate");
    this.uiFps = document.getElementById("fps");
    this.uiAgVert = document.getElementById("ag-fy");
    this.uiAgFwd = document.getElementById("ag-fx");
    this.uiAgAim = document.getElementById("ag-fz");
    this.uiAgResult = document.getElementById("ag-result");
    this.uiAgReward = document.getElementById("ag-reward");

    this.lossHistory = [];

    // Throughput counters, flushed to the UI once a second.
    this._shots = 0;
    this._frames = 0;
    this._rateSince = performance.now();
  }

  // Called once per frame loop tick with the shots retired since the last call
  // and whether that tick actually drew. Reporting the two rates separately is
  // the point: they are now independent, and seeing shots/sec hold steady while
  // fps moves (or vice versa) is how you confirm the visualization really is
  // off the training path. `rendered` is counted rather than assumed because
  // rendering is throttled below the rAF rate while training.
  tickRates(shots, rendered, now) {
    this._shots += shots;
    this._frames += rendered;
    const elapsed = now - this._rateSince;
    if (elapsed < 1000) return;
    const perSec = 1000 / elapsed;
    if (this.uiRate)
      this.uiRate.innerText = Math.round(this._shots * perSec) + "/s";
    if (this.uiFps) this.uiFps.innerText = Math.round(this._frames * perSec);
    this._shots = 0;
    this._frames = 0;
    this._rateSince = now;
  }

  setBackend(name, ok) {
    this.tfStatus.innerText = `Backend: ${name}`;
    if (ok) this.tfStatus.style.color = "#00ff00";
  }

  setStatus(text, color) {
    this.uiStatus.innerText = text;
    this.uiStatus.style.color = color;
  }

  setBatchProgress(text) {
    if (this.uiBatch) this.uiBatch.innerText = text;
  }

  // Fills the "Last Agent Decision" panel from the flat action buffer. The
  // component order is the one Ball.launch() reads: 0 forward, 1 vertical,
  // 2 aim.
  setAgentDecision(actions, offset, reward, scored) {
    const fmt = (v) => (v >= 0 ? "+" : "") + v.toFixed(3);
    this.uiAgFwd.innerText = fmt(actions[offset]);
    this.uiAgVert.innerText = fmt(actions[offset + 1]);
    this.uiAgAim.innerText = fmt(actions[offset + 2]);
    this.uiAgResult.innerText = scored ? "BASKET" : "MISS";
    this.uiAgResult.className = scored ? "outcome-score" : "outcome-miss";
    this.uiAgReward.innerText = reward.toFixed(2);
  }

  setStats({ accuracy, baskets, episodes, meanReward }) {
    this.uiAcc.innerText = Math.round(accuracy * 100) + "%";
    this.uiBaskets.innerText = baskets;
    this.uiEp.innerText = episodes;
    // Mean over the batch rather than a single shot's reward: with a thousand
    // shots resolving at once there is no meaningful "last" one, and the batch
    // mean is the number that actually tracks whether the policy is improving.
    if (this.uiReward) this.uiReward.innerText = meanReward.toFixed(2);
  }

  pushLoss(loss) {
    this.lossHistory.push({ train: loss });
    this.drawLoss();
  }

  drawLoss() {
    const w = this.lossCanvas.width;
    const h = this.lossCanvas.height;
    this.lossCtx.fillStyle = "#111";
    this.lossCtx.fillRect(0, 0, w, h);
    if (this.lossHistory.length < 2) return;

    let maxLoss = 0;
    this.lossHistory.forEach((d) => {
      if (d.train > maxLoss) maxLoss = d.train;
    });
    maxLoss = Math.max(maxLoss, 10.0);
    const denom = Math.max(1, this.lossHistory.length - 1);
    const mapX = (i) => (i / denom) * w;
    const mapY = (val) => h - (val / maxLoss) * h;

    this.lossCtx.beginPath();
    this.lossCtx.strokeStyle = "#44aaff";
    this.lossCtx.lineWidth = 2;
    for (let i = 0; i < this.lossHistory.length; i++) {
      const x = mapX(i);
      const y = mapY(this.lossHistory[i].train);
      if (i === 0) this.lossCtx.moveTo(x, y);
      else this.lossCtx.lineTo(x, y);
    }
    this.lossCtx.stroke();
  }

  // Draws sample `index` out of the state batch as a side-by-side L|R pair.
  // The batch is normalized luma in [0, 1] (VisionSystem writes it that way so
  // it can go to the GPU untouched), so it is scaled back to 0-255 here.
  drawAgentView(state, index) {
    const W = CONFIG.visionWidth;
    const H = CONFIG.visionHeight;
    const dW = W * 2; // combined L|R width
    const base = index * W * H * 2;
    const img = this.agentViewCtx.createImageData(dW, H);
    for (let i = 0; i < W * H; i++) {
      const row = Math.floor(i / W);
      const col = i % W;
      const invRow = H - 1 - row;

      const idxL = (invRow * dW + col) * 4;
      const valL = state[base + i * 2] * 255;
      img.data[idxL] = valL;
      img.data[idxL + 1] = valL;
      img.data[idxL + 2] = valL;
      img.data[idxL + 3] = 255;

      const idxR = (invRow * dW + (col + W)) * 4;
      const valR = state[base + i * 2 + 1] * 255;
      img.data[idxR] = valR;
      img.data[idxR + 1] = valR;
      img.data[idxR + 2] = valR;
      img.data[idxR + 3] = 255;
    }
    this.agentViewCtx.putImageData(img, 0, 0);
  }

  // Draws the L1 filter bank: one column per filter, one row per input eye, so
  // the left/right kernels of a filter sit above each other and the L/R
  // asymmetry that encodes disparity is actually visible. Geometry comes from
  // the weight tensor's own shape [kH, kW, inChannels, filters] rather than
  // hardcoded 8/8/2 — otherwise the strided index silently reads garbage the
  // moment the conv config changes.
  visualizeKernels(agent) {
    const kernel = agent.actor.layers[0].getWeights()[0];
    const [kH, kW, inChannels, numFilters] = kernel.shape;
    const wData = kernel.dataSync();

    // Wrap the filter bank into a grid instead of one long strip. A strip was
    // fine for 8 filters; at 32 it stretches to a 16:1 ribbon that the panel
    // scales down to slivers a couple of pixels tall. Each filter still stacks
    // its per-eye kernels vertically, so the L/R asymmetry that encodes
    // disparity stays readable within a filter's cell.
    const PER_ROW = Math.min(numFilters, 8);
    const TILE = 32;
    const PAD = 2;
    const cv = this.kernelCanvas;
    const filterRows = Math.ceil(numFilters / PER_ROW);
    const needW = PER_ROW * TILE;
    const needH = filterRows * inChannels * TILE;
    if (cv.width !== needW || cv.height !== needH) {
      cv.width = needW;
      cv.height = needH;
      this.kernelCtx.imageSmoothingEnabled = false; // reset by a resize
    }
    this.kernelCtx.clearRect(0, 0, cv.width, cv.height);

    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < wData.length; i++) {
      if (wData[i] < min) min = wData[i];
      if (wData[i] > max) max = wData[i];
    }
    const range = max - min || 1;

    const tile = document.createElement("canvas");
    tile.width = kW;
    tile.height = kH;
    const tileCtx = tile.getContext("2d");

    for (let c = 0; c < inChannels; c++) {
      for (let f = 0; f < numFilters; f++) {
        const imgData = tileCtx.createImageData(kW, kH);
        for (let y = 0; y < kH; y++) {
          for (let x = 0; x < kW; x++) {
            const idx =
              y * (kW * inChannels * numFilters) +
              x * (inChannels * numFilters) +
              c * numFilters +
              f;
            const v = Math.floor(((wData[idx] - min) / range) * 255);
            const p = (y * kW + x) * 4;
            imgData.data[p] = v;
            imgData.data[p + 1] = v;
            imgData.data[p + 2] = v;
            imgData.data[p + 3] = 255;
          }
        }
        tileCtx.putImageData(imgData, 0, 0);
        this.kernelCtx.drawImage(
          tile,
          (f % PER_ROW) * TILE,
          (Math.floor(f / PER_ROW) * inChannels + c) * TILE,
          TILE - PAD,
          TILE - PAD
        );
      }
    }
  }

  // Draws the dense layer as a heat grid rather than a node diagram. At 64
  // units, discrete circles with edges drawn to the output head fit; at
  // CONFIG.model.dense = 512 they do not, and the wiring degenerates into a
  // solid wash that shows nothing. The grid is laid out from dense.length, so
  // it tracks whatever width the layer actually has.
  //
  // Activations are normalized against the frame's own maximum: relu output is
  // unbounded, so a fixed 0-1 mapping either clips everything to white once the
  // network warms up or shows black while it is still small.
  visualizeActivations({ dense, output }) {
    const ctx = this.actCtx;
    const w = this.actCanvas.width;
    const h = this.actCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const nodeSize = 6;
    const outX = w - 30;
    const outYStart = h / 2 - 20;
    const outGap = 20;
    const gridW = outX - 30;

    const n = dense.length;
    // Cell aspect closest to square for n cells in a gridW x h box.
    const cols = Math.max(1, Math.round(Math.sqrt((n * gridW) / h)));
    const rows = Math.ceil(n / cols);
    const cw = gridW / cols;
    const ch = h / rows;

    let max = 0;
    for (let i = 0; i < n; i++) if (dense[i] > max) max = dense[i];
    const inv = max > 0 ? 1 / max : 0;

    for (let i = 0; i < n; i++) {
      const v = Math.floor(dense[i] * inv * 255);
      ctx.fillStyle = `rgb(${v}, ${v}, ${Math.min(255, v + 50)})`;
      ctx.fillRect(
        (i % cols) * cw,
        Math.floor(i / cols) * ch,
        Math.max(1, cw - 1),
        Math.max(1, ch - 1)
      );
    }

    const outPositions = [
      { x: outX, y: outYStart },
      { x: outX, y: outYStart + outGap },
      { x: outX, y: outYStart + outGap * 2 }
    ];
    // Matches Ball.launch()'s component order: 0 forward, 1 vertical, 2 aim.
    // These were labelled V/F/A, transposing the first two against the action
    // the ball is actually launched with.
    const labels = ["F", "V", "A"];
    for (let i = 0; i < 3; i++) {
      const val = (output[i] + 1) / 2;
      const intensity = Math.min(255, Math.floor(val * 255));
      ctx.fillStyle = `rgb(${intensity}, 50, ${255 - intensity})`;
      ctx.beginPath();
      ctx.arc(outPositions[i].x, outPositions[i].y, nodeSize, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "10px monospace";
      ctx.fillText(labels[i], outPositions[i].x + 8, outPositions[i].y + 3);
    }
  }
}

class TrainingArena {
  constructor({ agent, physics, court, vision, dashboard, showcase }) {
    this.agent = agent;
    this.court = court;
    this.vision = vision;
    this.dashboard = dashboard;
    this.showcase = showcase;

    this.hoopPos = court.rimPosition;
    this.hoopPosCannon = court.rimPositionCannon;

    this.balls = [];
    this.bodyToBall = new Map();
    for (let i = 0; i < CONFIG.batchSize; i++) {
      const b = new Ball(i, physics, CONFIG.ballRadius);
      this.balls.push(b);
      this.bodyToBall.set(b.body, b);
    }

    // Batch-wide buffers, allocated once and reused for every batch. The old
    // path allocated a fresh Float32Array per batch plus a per-ball slice of
    // it — at 1024 x 256x256x2 that is 537MB of allocation and 537MB of
    // garbage per batch, which on its own was enough to stall the tab.
    //
    // stateFloats holds normalized luma rather than bytes: it is uploaded to
    // the GPU verbatim, once per batch, and the tensor that results
    // (stateTensor) stays resident for the whole flight so every training pass
    // reads it without another conversion. 2048 x 128x128x2 is 268MB on each
    // side, which is the deliberate trade — VRAM and system RAM are what this
    // machine has spare; CPU passes over 33M floats are what it does not.
    const fs = this.vision.frameSize;
    this.stateFloats = new Float32Array(CONFIG.batchSize * fs);
    this.stateTensor = null;
    this.actions = new Float32Array(CONFIG.batchSize * 3);
    this.rewards = new Float32Array(CONFIG.batchSize);

    this.episodeStats = { count: 0, baskets: 0, shots: 0 };
    // Rolling hit/miss window as a ring buffer. An Array with a shift() per
    // retired shot is O(window) per shot — at a 2048-shot window and a
    // 2048-shot batch that is four million element moves per batch, spent
    // entirely on keeping an array the right length.
    this.accWindow = new Uint8Array(CONFIG.accuracyWindow);
    this.accCursor = 0;
    this.accFilled = 0;
    this.accSum = 0;
    this.isTrainingStep = false;
    this.activeCount = 0;
    this.exploreNoise = CONFIG.exploreNoise;
    // Objects that must be hidden from the agents' cameras during capture.
    this.hiddenDuringCapture = [];
  }

  // Push one shot outcome into the rolling accuracy window.
  _recordOutcome(scored) {
    const v = scored ? 1 : 0;
    if (this.accFilled === this.accWindow.length)
      this.accSum -= this.accWindow[this.accCursor];
    else this.accFilled++;
    this.accWindow[this.accCursor] = v;
    this.accSum += v;
    this.accCursor = (this.accCursor + 1) % this.accWindow.length;
  }

  // Start a fresh batch from outside the cycle (the START TRAINING button).
  // No-op if the pipeline is already busy: the in-flight cycle will launch its
  // own batch when it finishes, and starting a second one on top of it is what
  // used to produce "another fit() call is ongoing" and reads of an already
  // disposed state tensor.
  async startBatch() {
    if (this.isTrainingStep) return;
    this.isTrainingStep = true;
    try {
      await this._spawnAndLaunch();
    } catch (e) {
      console.error("Batch Error", e);
    } finally {
      this.isTrainingStep = false;
    }
  }

  // Spawn, capture vision, predict, and launch a fresh batch.
  //
  // Async because action inference reads its results back with await .data()
  // instead of blocking the main thread on .dataSync(). It does not touch
  // isTrainingStep: the caller owns that latch for the whole cycle, so that
  // the frame loop cannot step physics on balls that have been spawned but not
  // yet launched, or mistake the not-yet-set activeCount of a half-built batch
  // for a finished one and try to train on it.
  async _spawnAndLaunch() {
    // Free last batch's VRAM before capture overwrites the buffer that backs
    // it. The tensor is a view onto stateFloats until tfjs has uploaded it,
    // so disposing here — rather than after the new upload — is what makes
    // reusing a single state buffer safe.
    this._disposeState();

    for (const b of this.balls) b.spawn();
    this.dashboard.setBatchProgress("Capturing...");

    this.vision.captureBatch(
      this.balls,
      this.hoopPos,
      this.court,
      this.hiddenDuringCapture,
      this.stateFloats
    );
    this.stateTensor = this.agent.uploadBatch(
      this.stateFloats,
      this.balls.length
    );
    await this.agent.predictBatch(
      this.stateTensor,
      this.balls.length,
      this.actions,
      this.exploreNoise
    );
    for (let i = 0; i < this.balls.length; i++)
      this.balls[i].launch(this.actions, i * 3, this.hoopPos);

    // Last, so the loop only starts stepping a batch that is fully launched.
    this.activeCount = this.balls.length;
    this.dashboard.setBatchProgress("Simulating...");
  }

  _disposeState() {
    if (this.stateTensor) {
      this.stateTensor.dispose();
      this.stateTensor = null;
    }
  }

  // Advance one simulated step; returns the number of balls still in flight.
  update() {
    let active = 0;
    const stride = CONFIG.pathStride;
    for (const b of this.balls) {
      if (!b.active) continue;

      b.steps++;
      if (b.steps % stride === 0) b.record();

      const dist = b.body.position.distanceTo(this.hoopPosCannon);
      if (dist < b.minDist) b.minDist = dist;

      const isStopped = b.body.velocity.length() < 0.2;
      const isOOB =
        Math.abs(b.body.position.x) > 50 || Math.abs(b.body.position.z) > 30;
      // Terminate the moment the ball reaches floor level. It rests at
      // y == ballRadius on the (infinite) floor plane and every shot launches
      // from y >= 4.5, so this only fires once a shot has come back down —
      // culling missed shots on first floor contact instead of letting them
      // bounce and roll to a stop. Made shots still end via b.scored at the
      // hoop, well above the floor. (Replaces the old y < -2 check, which the
      // infinite floor plane made unreachable.)
      const isOnFloor = b.body.position.y < CONFIG.ballRadius + 0.15;
      const isTimeout = b.steps > CONFIG.maxFlightSteps;

      if (
        b.scored ||
        isOnFloor ||
        isOOB ||
        (isStopped && b.steps > 10) ||
        isTimeout
      ) {
        b.record(); // pin the path's endpoint to where the shot actually ended
        b.retire();
      } else {
        active++;
      }
    }
    this.activeCount = active;
    return active;
  }

  _reward(b) {
    let shotDistance = 0;
    if (b.pathLen > 0) {
      shotDistance = Math.sqrt(
        Math.pow(b.path[0] - this.hoopPos.x, 2) +
          Math.pow(b.path[2] - this.hoopPos.z, 2)
      );
    }
    if (b.scored) {
      const reward = 10.0 * (1.0 + shotDistance / 20.0);
      return { reward, type: "score" };
    }
    let reward = -b.minDist / 10;
    if (b.hitRim) return { reward: reward + 2.0, type: "rim" };
    if (b.hitBackboard) return { reward: reward + 0.5, type: "rim" };
    return { reward, type: "miss" };
  }

  // Compute rewards, train, then relaunch. Re-entrant-guarded.
  async finishBatch() {
    if (this.isTrainingStep) return;
    this.isTrainingStep = true;

    const n = this.balls.length;
    // Everything is inside the try: the latch is released in exactly one place,
    // so a throw anywhere — scoring, a dashboard draw, the training step —
    // costs one batch rather than wedging the simulation with the latch stuck
    // on and no visible error.
    try {
      let best = 0;
      let rewardSum = 0;
      for (let i = 0; i < n; i++) {
        const b = this.balls[i];
        const { reward, type } = this._reward(b);
        this.rewards[i] = reward;
        rewardSum += reward;
        if (reward > this.rewards[best]) best = i;
        if (b.scored) this.episodeStats.baskets++;
        // The showcase decides on its own what is worth drawing and copies only
        // those paths; every other flight in the batch is simply discarded.
        this.showcase.offer(b, reward, type);
        this.episodeStats.shots++;
        this.episodeStats.count++;
        this._recordOutcome(b.scored);
      }

      const rollingAcc = this.accFilled > 0 ? this.accSum / this.accFilled : 0;
      this.dashboard.setStats({
        accuracy: rollingAcc,
        baskets: this.episodeStats.baskets,
        episodes: this.episodeStats.count,
        meanReward: rewardSum / n
      });
      this.dashboard.setBatchProgress("Training GPU...");

      // The dashboards show the batch's best shot rather than ball 0 — with the
      // court no longer showing the batch, an arbitrary sample was the one view
      // of what the agent is doing, and a random miss is the least informative
      // choice available. Drawn before training so it matches the weights that
      // actually produced the shot.
      //
      // A batch whose capture failed has no state tensor; skip straight to the
      // relaunch in the finally rather than handing null to the agent.
      if (this.stateTensor) {
        this.dashboard.drawAgentView(this.stateFloats, best);
        this.dashboard.visualizeActivations(
          this.agent.getActivations(this.stateTensor, best)
        );
        this.dashboard.visualizeKernels(this.agent);
        this.dashboard.setAgentDecision(
          this.actions,
          best * 3,
          this.rewards[best],
          this.balls[best].scored
        );

        const loss = await this.agent.trainOnBatch(
          this.stateTensor,
          this.actions,
          this.rewards,
          n
        );
        if (loss != null) this.dashboard.pushLoss(loss);

        // Anneal exploration once per trained batch so the policy tightens
        // toward exploitation as it improves, with a floor that keeps a little
        // jitter.
        this.exploreNoise = Math.max(
          CONFIG.exploreNoiseMin,
          this.exploreNoise * CONFIG.exploreNoiseDecay
        );
      }
    } catch (e) {
      console.error("Train Error", e);
    } finally {
      // Relaunching in finally so a failed training step costs one batch rather
      // than wedging the simulation: the loop only advances again once there
      // are live balls to advance. _spawnAndLaunch() disposes the batch tensor
      // itself, before it overwrites the buffer backing it.
      try {
        await this._spawnAndLaunch();
      } catch (e) {
        console.error("Batch Error", e);
      }
      this.isTrainingStep = false;
    }
    return n;
  }

  // Drop the batch's GPU residency when training stops. Only safe while idle —
  // the caller checks isTrainingStep, because an in-flight cycle is still
  // reading the tensor this disposes.
  release() {
    this._disposeState();
  }
}

class ManualBall {
  constructor(scene, physics, assets, camera, radius) {
    this.scene = scene;
    this.camera = camera;
    this.radius = radius;

    this.mesh = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 32, 32),
      assets.manualBall
    );
    this.mesh.castShadow = true;
    scene.add(this.mesh);

    this.body = new CANNON.Body({
      mass: 2,
      shape: new CANNON.Sphere(radius),
      material: physics.plastic,
      linearDamping: 0.1,
      angularDamping: 0.1
    });
    this.body.collisionFilterGroup = CONFIG.groups.ball;
    this.body.collisionFilterMask = CONFIG.groups.court;
    physics.add(this.body);

    this.enabled = true; // disabled while training
    this.inProgress = false;
    this.scored = false;

    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.dragPlane = new THREE.Plane();
    this.planeNormal = new THREE.Vector3();
    this.dragStart = new THREE.Vector3();
    this.dragCurrent = new THREE.Vector3();
    this.holdOffset = new THREE.Vector3();
    this.isDragging = false;
    this.isHolding = false;

    this.arrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 0, 0),
      1,
      0xffff00
    );
    this.arrow.visible = false;
    scene.add(this.arrow);

    this._bindPointer();
    this.reset();
  }

  // Disabling also hides and sleeps the ball. It used to stay visible and
  // dynamic during training, which under fast-forward meant it dropped to the
  // floor at whatever multiple of real time the sim was running at and then sat
  // there — a stray ball on a court that is otherwise showing the agent's best
  // work.
  setEnabled(v) {
    this.enabled = v;
    this.mesh.visible = v;
    this.arrow.visible = false;
    if (v) this.body.wakeUp();
    else this.body.sleep();
  }

  reset() {
    this.inProgress = false;
    this.scored = false;
    const x = Math.random() * 42;
    const z = (Math.random() - 0.5) * 46;
    const y = 4.5 + Math.random() * 1.5;
    this.body.position.set(x, y, z);
    this.body.velocity.set(0, 0, 0);
    this.body.angularVelocity.set(0, 0, 0);
    this.body.sleep();
    this.mesh.position.copy(this.body.position);
    this.mesh.quaternion.copy(this.body.quaternion);
  }

  onContact(other, court) {
    if (other === court.scoringSensor && this.body.velocity.y < 0)
      this.scored = true;
  }

  update() {
    this.mesh.position.copy(this.body.position);
    this.mesh.quaternion.copy(this.body.quaternion);
    if (this.inProgress) {
      const isStopped = this.body.velocity.length() < 0.5;
      const isBelow = this.body.position.y < 1;
      const isOOB =
        Math.abs(this.body.position.x) > 50 ||
        Math.abs(this.body.position.z) > 30;
      if (this.scored || isBelow || isOOB || isStopped) this.reset();
    }
  }

  _updateMouse(e) {
    this.mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    this.mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  }

  _bindPointer() {
    window.addEventListener("contextmenu", (e) => e.preventDefault());

    window.addEventListener("keydown", (e) => {
      if (e.code === "Space") this.reset();
    });

    window.addEventListener("pointermove", (e) => {
      if (!this.enabled) return;
      this._updateMouse(e);

      if (this.isHolding) {
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const target = new THREE.Vector3();
        this.raycaster.ray.intersectPlane(this.dragPlane, target);
        if (target) {
          target.add(this.holdOffset);
          if (target.y < this.radius) target.y = this.radius;
          this.body.position.copy(target);
          this.body.velocity.set(0, 0, 0);
          this.mesh.position.copy(target);
        }
      } else if (this.isDragging) {
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const target = new THREE.Vector3();
        this.raycaster.ray.intersectPlane(this.dragPlane, target);
        if (target) {
          this.dragCurrent.copy(target);
          const f = new THREE.Vector3()
            .copy(this.dragStart)
            .sub(this.dragCurrent);
          const len = f.length();
          if (len > 0) {
            this.arrow.position.copy(this.mesh.position);
            this.arrow.setDirection(f.normalize());
            this.arrow.setLength(len, len * 0.2, len * 0.1);
          }
        }
      }
    });

    window.addEventListener(
      "pointerdown",
      (e) => {
        if (!this.enabled) return;
        this._updateMouse(e);
        if (e.button === 1) return;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObject(this.mesh);

        if (e.button === 2 && intersects.length > 0) {
          this.isHolding = true;
          e.stopImmediatePropagation();
          this.body.type = CANNON.Body.KINEMATIC;
          this.body.velocity.set(0, 0, 0);
          this.planeNormal
            .copy(this.camera.position)
            .sub(this.mesh.position)
            .normalize();
          this.dragPlane.setFromNormalAndCoplanarPoint(
            this.planeNormal,
            this.mesh.position
          );
          const target = new THREE.Vector3();
          this.raycaster.ray.intersectPlane(this.dragPlane, target);
          if (target) this.holdOffset.copy(this.mesh.position).sub(target);
        }

        if (e.button === 0 && intersects.length > 0) {
          this.isDragging = true;
          e.stopImmediatePropagation();
          this.planeNormal
            .copy(this.camera.position)
            .sub(this.mesh.position)
            .normalize();
          this.dragPlane.setFromNormalAndCoplanarPoint(
            this.planeNormal,
            this.mesh.position
          );
          this.raycaster.ray.intersectPlane(this.dragPlane, this.dragStart);
          this.dragCurrent.copy(this.dragStart);
          this.arrow.visible = true;
        }
      },
      { capture: true }
    );

    window.addEventListener("pointerup", () => {
      if (this.isHolding) {
        this.isHolding = false;
        this.body.type = CANNON.Body.DYNAMIC;
        this.body.mass = 2;
        this.body.wakeUp();
      }
      if (this.isDragging) {
        const f = new THREE.Vector3()
          .copy(this.dragStart)
          .sub(this.dragCurrent);
        const power = 15;
        this.inProgress = true;
        this.body.wakeUp();
        this.body.applyImpulse(
          new CANNON.Vec3(f.x * power, f.y * power, f.z * power),
          new CANNON.Vec3(0, 0, 0)
        );
        this.isDragging = false;
        this.arrow.visible = false;
      }
    });
  }
}

class App {
  constructor() {
    this.assets = new Assets();
    this.sceneMgr = new SceneManager();
    this.physics = new PhysicsWorld();
    this.dashboard = new Dashboard();

    const scene = this.sceneMgr.scene;
    this.court = new Court(scene, this.physics, this.assets);
    this.vision = new VisionSystem(
      this.sceneMgr.renderer,
      scene,
      CONFIG.batchSize
    );
    this.trajectory = new TrajectoryTrails(
      scene,
      this.assets,
      CONFIG.maxHistory
    );
    this.showcase = new Showcase(scene, this.assets, this.trajectory, {
      count: CONFIG.showcaseCount,
      refreshMs: CONFIG.showcaseRefreshMs
    });
    this.manual = new ManualBall(
      scene,
      this.physics,
      this.assets,
      this.sceneMgr.camera,
      CONFIG.ballRadius
    );

    this.agent = null;
    this.arena = null;
    this.trainingMode = false;
    this.lastFrame = performance.now();
    this._lastRender = 0;
    this._pendingDt = 0;

    this._wireContacts();
    this._wireButtons();
  }

  _wireContacts() {
    this.physics.onBeginContact((e) => {
      const { bodyA, bodyB } = e;

      if (bodyA === this.manual.body || bodyB === this.manual.body) {
        const other = bodyA === this.manual.body ? bodyB : bodyA;
        this.manual.onContact(other, this.court);
        return;
      }

      if (!this.arena) return;
      let ball = this.arena.bodyToBall.get(bodyA);
      let other = bodyB;
      if (!ball) {
        ball = this.arena.bodyToBall.get(bodyB);
        other = bodyA;
      }
      if (ball && ball.active) ball.onContact(other, this.court);
    });
  }

  _wireButtons() {
    const btnTrain = document.getElementById("toggle-train");
    const btnExport = document.getElementById("export-btn");

    btnTrain.addEventListener("click", () => {
      this.trainingMode = !this.trainingMode;
      this.dashboard.setStatus(
        this.trainingMode ? "Training (Fast Fwd)" : "Manual Mode",
        this.trainingMode ? "#0f0" : "white"
      );
      btnTrain.innerText = this.trainingMode
        ? "STOP TRAINING"
        : "START TRAINING";
      btnTrain.classList.toggle("active");
      this.manual.setEnabled(!this.trainingMode);
      this.manual.reset();
      if (this.arena) {
        if (this.trainingMode) {
          // Unawaited: startBatch latches isTrainingStep synchronously, so the
          // frame loop stays off the batch until it is launched — and it
          // declines outright if a cycle from before the last STOP is still
          // resolving, rather than racing it for the state tensor.
          this.arena.startBatch();
        } else {
          for (const b of this.arena.balls) b.retire();
          // Only when idle: an in-flight capture or training step is still
          // reading the batch tensor, and the next _spawnAndLaunch() disposes it
          // first thing regardless, so deferring costs residency, not a leak.
          if (!this.arena.isTrainingStep) this.arena.release();
        }
      }
      if (!this.trainingMode) this.showcase.hideAll();
    });

    btnExport.addEventListener("click", async () => {
      if (this.agent && this.agent.actor) await this.agent.saveActor();
    });
  }

  async start() {
    try {
      await tf.setBackend("webgpu");
      await tf.ready();
      this.dashboard.setBackend(tf.getBackend().toUpperCase(), true);
    } catch (e) {
      console.warn("WebGPU failed, fallback", e);
      try {
        await tf.setBackend("webgl");
        await tf.ready();
        this.dashboard.setBackend(tf.getBackend().toUpperCase(), true);
      } catch (e2) {
        await tf.setBackend("cpu");
        this.dashboard.setBackend("CPU", false);
      }
    }

    this.agent = new CNNAgent({
      learningRate: CONFIG.learningRate,
      batchSize: CONFIG.batchSize,
      l2: CONFIG.l2
    });
    this.arena = new TrainingArena({
      agent: this.agent,
      physics: this.physics,
      court: this.court,
      vision: this.vision,
      dashboard: this.dashboard,
      showcase: this.showcase
    });
    // Everything the agents must not see. The training balls themselves are no
    // longer in this list because they no longer have meshes at all.
    this.arena.hiddenDuringCapture = [
      this.manual.mesh,
      this.manual.arrow,
      this.trajectory.group,
      this.showcase.group
    ];

    // Don't spawn the training batch at startup — the app opens in manual mode
    // and the balls stay hidden until the user starts training (startBatch
    // spawns them). This avoids a startup flash of the whole batch.
    this.manual.reset();
    this._loop();
  }

  // One rendered frame. Simulation and rendering are deliberately on different
  // clocks here: the frame draws whatever the showcase is replaying, while the
  // batch underneath is advanced as many steps as the budget allows. A batch
  // therefore takes as long as it takes to *compute*, not as long as the balls
  // take to fall.
  _loop() {
    requestAnimationFrame(() => this._loop());

    const now = performance.now();
    // Accumulated rather than consumed every frame: with rendering throttled,
    // a rAF tick can pass without the showcase advancing, and the time it
    // covered still has to reach the replay or playback runs slow.
    this._pendingDt += (now - this.lastFrame) / 1000;
    this.lastFrame = now;

    let retired = 0;
    if (this.trainingMode && this.arena) {
      if (!this.arena.isTrainingStep) {
        // Fast-forward the batch. Bounded by wall clock rather than a fixed
        // step count so the budget holds across machines: a fast GPU gets more
        // simulation per frame, a slow one still yields in time to paint.
        const deadline = now + CONFIG.simBudgetMs;
        let steps = 0;
        while (
          steps < CONFIG.maxStepsPerFrame &&
          this.arena.activeCount > 0 &&
          performance.now() < deadline
        ) {
          this.physics.step();
          this.arena.update();
          steps++;
        }
        if (this.arena.activeCount === 0) {
          retired = CONFIG.batchSize;
          // Deliberately not awaited: training resolves across later frames
          // while rendering keeps running, and isTrainingStep guards re-entry.
          this.arena.finishBatch();
        } else {
          this.dashboard.setBatchProgress(`${this.arena.activeCount} Active`);
        }
      }
    } else {
      // Manual mode runs at real time — it's a person aiming a ball.
      this.physics.step();
    }

    // Render on its own clock. Every rAF tick used to draw, so a tick's work
    // was simBudgetMs of simulation plus a full scene render, and the reel got
    // a repaint it has no use for — the showcase only promotes new shots once
    // a second. Skipping repaints hands that share back to the batch. Manual
    // mode is exempt: there the picture is the point.
    const shouldRender =
      !this.trainingMode ||
      now - this._lastRender >= CONFIG.renderIntervalMs;

    if (shouldRender) {
      this._lastRender = now;
      // Clamped so a long training stall (or a backgrounded tab) doesn't make
      // playback jump — the reel should always look like real-time basketball.
      const dt = Math.min(0.1, this._pendingDt);
      this._pendingDt = 0;

      this.manual.update();
      this.showcase.refresh(now);
      this.showcase.update(dt);
      this.sceneMgr.render();
    }

    this.dashboard.tickRates(retired, shouldRender ? 1 : 0, now);
  }
}

new App().start();
