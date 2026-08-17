import * as THREE from "https://esm.sh/three@0.132.2";
import { OrbitControls } from "https://esm.sh/three@0.132.2/examples/jsm/controls/OrbitControls.js";
import * as CANNON from "https://esm.sh/cannon-es@0.19.0";

const CONFIG = {
  batchSize: 1024,
  visionWidth: 256,
  visionHeight: 256,
  ballRadius: 0.4,
  learningRate: 0.001,
  l2: 0.001,
  maxHistory: 100,
  accuracyWindow: 1024,
  exploreNoise: 0.4,
  exploreNoiseMin: 0.15,
  exploreNoiseDecay: 0.999,
  advantageTemp: 1.0,
  advantageClip: 20.0,
  ipd: 0.2067,
  visionFov: 60,

  // --- Throughput -----------------------------------------------------------
  // Training no longer runs at display rate. The old loop stepped physics once
  // per requestAnimationFrame, so a batch could not finish faster than the
  // slowest ball's real-time flight — roughly ten seconds of wall clock spent
  // watching balls fall. These bound how much simulation is fast-forwarded per
  // rendered frame: keep stepping until either the step cap or the wall-clock
  // budget is hit, then yield so the browser can paint.
  //
  // simBudgetMs is the knob that trades framerate for learning rate. 20ms
  // settles around 30fps of playback, which is plenty for watching ten balls
  // arc toward a hoop; lower it if the view matters more than the learning
  // rate, raise it if the reverse.
  simBudgetMs: 20,
  maxStepsPerFrame: 400,
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
  // pages of (visionAtlasMax / visionWidth)^2 tiles. At 256px tiles that is
  // 64 balls per page, which caps the readback buffers at 16MB per eye
  // instead of the 268MB an all-in-one 8192x8192 atlas demanded.
  visionAtlasMax: 2048,
  // Samples per GPU upload during training. The state buffer is held on the
  // CPU as bytes and converted to float a chunk at a time, so peak GPU
  // residency is chunk * visionWidth * visionHeight * 2 * 4 bytes (~33MB at
  // 64 x 256px) rather than the whole 537MB batch at once.
  gpuChunk: 64,

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
    this.actor = this._buildActor();
    this.critic = this._buildCritic();
  }

  // Uploads samples [start, start+count) of a byte-packed state buffer as a
  // float tensor in [0, 1].
  //
  // States live on the CPU as one Uint8Array because that is exactly what the
  // framebuffer readback produces — promoting luma to float32 at capture time
  // quadrupled the buffer (537MB per batch at 256x256x1024) for zero extra
  // information. The conversion is deferred to here and done a chunk at a
  // time, so the float expansion only ever exists for CONFIG.gpuChunk samples.
  //
  // The subarray is a view, not a copy, and tf.tensor's Uint8Array ->
  // Float32Array widening is a single native typed-array conversion; the
  // divide happens on the GPU. Chunks are contiguous rather than shuffled:
  // every sample in a batch is an independent random spawn, so batch order is
  // already arbitrary and gathering scattered rows would force a slow
  // element-by-element copy back on the CPU.
  _chunkTensor(stateBytes, start, count) {
    const fs = this.frameSize;
    const sub = stateBytes.subarray(start * fs, (start + count) * fs);
    return tf.tidy(() =>
      tf
        .tensor(
          sub,
          [count, this.visionH, this.visionW, this.channels],
          "float32"
        )
        .div(255)
    );
  }

  // Runs fn(chunkTensor, start, count) over the whole buffer in gpuChunk-sized
  // pieces, disposing each chunk before the next is uploaded.
  async _overChunks(stateBytes, count, fn) {
    const chunk = CONFIG.gpuChunk;
    for (let start = 0; start < count; start += chunk) {
      const n = Math.min(chunk, count - start);
      const x = this._chunkTensor(stateBytes, start, n);
      try {
        await fn(x, start, n);
      } finally {
        x.dispose();
      }
    }
  }

  // Shared conv stack: conv(8) -> conv(16) -> flatten -> dense(64).
  // Layer indices [0],[1] are the two conv layers in BOTH nets — the weight
  // sync in train() depends on that alignment.
  //
  // denseReg is null for the actor: dense(64) and the output head are the only
  // trainable parts of the policy, and L2 on them pulls the policy toward
  // constant zero output — i.e. the same shot from every position — which
  // competes directly with an already-weak improvement signal. The critic keeps
  // L2 everywhere, and since the critic owns the conv backbone the filters stay
  // regularized regardless.
  _convBase(model, denseReg = this.l2Reg) {
    model.add(
      tf.layers.conv2d({
        inputShape: [this.visionH, this.visionW, this.channels],
        filters: 8,
        kernelSize: 8,
        strides: 4,
        activation: "relu",
        kernelRegularizer: this.l2Reg
      })
    );
    model.add(
      tf.layers.conv2d({
        filters: 16,
        kernelSize: 4,
        strides: 2,
        activation: "relu",
        kernelRegularizer: this.l2Reg
      })
    );
    model.add(tf.layers.flatten());
    model.add(
      tf.layers.dense({
        units: 64,
        activation: "relu",
        kernelRegularizer: denseReg
      })
    );
  }

  _buildActor() {
    const m = tf.sequential();
    this._convBase(m, null);
    // The critic owns the shared conv backbone: train() copies the critic's
    // conv weights into the actor after every batch. Freeze the actor's two
    // conv layers so the actor update doesn't waste work computing gradients
    // that get overwritten, and so the actor's dense head trains against a
    // stable feature extractor.
    m.layers[0].trainable = false;
    m.layers[1].trainable = false;
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

  // stateBytes: Uint8Array of count * frameSize luma samples (frameSize=W*H*2).
  // Writes count * 3 action components into `out` (a Float32Array). Chunked so
  // inference on a full batch never needs the whole float expansion resident
  // at once, and flat so a batch of launches costs no per-ball allocation.
  predictBatch(stateBytes, count, out, noiseScale = 0) {
    // Clamp to [-1, 1] after adding exploration noise: the actor's output
    // is tanh-bounded, and these actions are later stored as regression
    // targets, so out-of-range values would be unreachable training goals.
    const clamp = (v) => Math.max(-1, Math.min(1, v));
    const chunk = CONFIG.gpuChunk;
    for (let start = 0; start < count; start += chunk) {
      const n = Math.min(chunk, count - start);
      const data = tf.tidy(() => {
        const x = this._chunkTensor(stateBytes, start, n);
        return this.actor.predict(x).dataSync(); // n * 3
      });
      for (let i = 0; i < n * 3; i++)
        out[start * 3 + i] = clamp(
          data[i] + (Math.random() - 0.5) * noiseScale
        );
    }
    return out;
  }

  // Activations for a single sample, addressed by its index in the batch.
  getActivations(stateBytes, index) {
    return tf.tidy(() => {
      let t = this._chunkTensor(stateBytes, index, 1);
      for (let i = 0; i <= 3; i++) t = this.actor.layers[i].apply(t);
      const dense64 = t;
      const output3 = this.actor.layers[4].apply(dense64);
      return { dense: dense64.dataSync(), output: output3.dataSync() };
    });
  }

  // Trains critic on returns, then the actor on advantage-weighted actions.
  // Returns the critic loss (or null if nothing to train on).
  //
  // stateBytes/actions/rewards describe one batch; nothing is retained
  // afterwards, so the caller owns (and may reuse) the state buffer.
  //
  // The whole step is chunked. Previously each of the three passes built a
  // tensor over the entire batch, which at 1024 x 256x256x2 is 537MB of GPU
  // residency per tensor — enough to thrash or fail outright. Now each pass
  // walks the batch in gpuChunk pieces and the per-sample bookkeeping
  // (values, advantages, weights) stays in plain typed arrays on the CPU,
  // where it costs 4 bytes a sample.
  async trainOnBatch(stateBytes, actions, rewards, count) {
    if (count === 0) return null;

    // --- Pass 1: critic regression onto realized reward ---------------------
    // fit() is still used per chunk rather than a hand-rolled minimize() so
    // the L2 regularizer losses and the Adam slot state keep behaving exactly
    // as they did. batchSize 32 preserves the original optimizer step count.
    let lossSum = 0;
    let lossChunks = 0;
    await this._overChunks(stateBytes, count, async (x, start, n) => {
      const yb = tf.tensor2d(rewards.subarray(start, start + n), [n, 1]);
      const history = await this.critic.fit(x, yb, {
        epochs: 1,
        batchSize: 32,
        shuffle: false,
        verbose: 0
      });
      yb.dispose();
      if (history.history.loss) {
        lossSum += history.history.loss[0] * n;
        lossChunks += n;
      }
    });
    const loss = lossChunks > 0 ? lossSum / lossChunks : 0;

    // --- Pass 2: value estimates for the whole batch ------------------------
    // Read back as a CPU Float32Array so the advantage statistics below can be
    // computed without holding a batch-wide tensor.
    const values = new Float32Array(count);
    await this._overChunks(stateBytes, count, (x, start, n) => {
      const v = tf.tidy(() => this.critic.predict(x).dataSync());
      values.set(v.subarray(0, n), start);
    });

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
      for (let i = 0; i < 2; i++)
        this.actor.layers[i].setWeights(this.critic.layers[i].getWeights());
    });

    // --- Pass 3: advantage-weighted actor update ----------------------------
    await this._overChunks(stateBytes, count, (x, start, n) => {
      this._fitActorWeighted(x, actions, weightData, start, n);
    });

    return loss;
  }

  // One epoch of minibatch Adam over a per-sample weighted MSE, restricted to
  // samples [start, start+n) of the batch. model.fit() cannot do this: passing
  // sampleWeight throws "Support sampleWeight is not implemented yet" in tfjs
  // 4.x, so the update is driven explicitly. Minibatch size matches fit()'s
  // default of 32 so the number of optimizer steps per batch is unchanged.
  // Only trainableWeights are passed to minimize(), which already excludes the
  // two frozen conv layers.
  //
  // Minibatches are cut out of the already-uploaded chunk with tf.slice, so
  // splitting the batch into chunks costs no extra CPU->GPU traffic.
  _fitActorWeighted(chunk, actions, weights, start, n, minibatch = 32) {
    const vars = this.actor.trainableWeights.map((w) => w.val);
    for (let off = 0; off < n; off += minibatch) {
      const m = Math.min(minibatch, n - off);
      const base = (start + off) * 3;
      tf.tidy(() => {
        const xb = chunk.slice([off, 0, 0, 0], [m, -1, -1, -1]);
        const yb = tf.tensor2d(actions.subarray(base, base + m * 3), [m, 3]);
        const wb = tf.tensor1d(weights.subarray(start + off, start + off + m));
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
  // samples into `out`, a caller-owned Uint8Array of balls.length * frameSize.
  //
  // Output is 8-bit because the framebuffer it comes from is 8-bit: widening to
  // float here bought no precision and quadrupled every downstream buffer. The
  // agent converts to float per GPU chunk instead (CNNAgent._chunkTensor).
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

    for (let start = 0; start < balls.length; start += this.pageSize) {
      const n = Math.min(this.pageSize, balls.length - start);
      this._renderPage(balls, start, n, hoopPos);
      this._readPage(out, start, n);
    }

    // Restore.
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

    // Integer luma: 77/150/29 are the Rec.601 weights scaled by 256, so the
    // shift replaces three float multiplies and a divide per pixel per eye.
    // At 256x256x1024 this inner loop runs 67M times a batch, which is enough
    // for the difference to be worth the squint.
    for (let i = 0; i < n; i++) {
      const tileX = (i % cols) * W;
      const tileY = Math.floor(i / cols) * H;
      let o = (start + i) * frameSize;
      for (let y = 0; y < H; y++) {
        let a = ((tileY + y) * atlasW + tileX) * 4;
        for (let x = 0; x < W; x++) {
          out[o] = (77 * bl[a] + 150 * bl[a + 1] + 29 * bl[a + 2]) >> 8;
          out[o + 1] = (77 * br[a] + 150 * br[a + 1] + 29 * br[a + 2]) >> 8;
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

  // Called once per rendered frame with the shots retired since the last call.
  // Reporting the two rates separately is the point: they are now independent,
  // and seeing shots/sec hold steady while fps moves (or vice versa) is how you
  // confirm the visualization really is off the training path.
  tickRates(shots, now) {
    this._shots += shots;
    this._frames++;
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

  // Draws sample `index` out of a byte-packed state batch as a side-by-side
  // L|R pair. Values are already 0-255 luma, so they go straight into the
  // ImageData with no rescaling.
  drawAgentView(stateBytes, index) {
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
      const valL = stateBytes[base + i * 2];
      img.data[idxL] = valL;
      img.data[idxL + 1] = valL;
      img.data[idxL + 2] = valL;
      img.data[idxL + 3] = 255;

      const idxR = (invRow * dW + (col + W)) * 4;
      const valR = stateBytes[base + i * 2 + 1];
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

    const TILE = 32;
    const PAD = 2;
    const cv = this.kernelCanvas;
    const needW = numFilters * TILE;
    const needH = inChannels * TILE;
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
          f * TILE,
          c * TILE,
          TILE - PAD,
          TILE - PAD
        );
      }
    }
  }

  visualizeActivations({ dense, output }) {
    const ctx = this.actCtx;
    const w = this.actCanvas.width;
    const h = this.actCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const nodeSize = 6;
    const gap = 10;
    const startX = 10;
    const startY = 10;
    const outX = w - 30;
    const outYStart = h / 2 - 20;
    const outGap = 20;

    const densePositions = [];
    for (let i = 0; i < 64; i++) {
      const row = Math.floor(i / 8);
      const col = i % 8;
      densePositions.push({
        x: startX + col * (nodeSize + gap),
        y: startY + row * (nodeSize + gap)
      });
    }
    const outPositions = [
      { x: outX, y: outYStart },
      { x: outX, y: outYStart + outGap },
      { x: outX, y: outYStart + outGap * 2 }
    ];

    ctx.lineWidth = 0.5;
    for (let i = 0; i < 64; i++) {
      if (dense[i] > 0.1) {
        for (let j = 0; j < 3; j++) {
          const alpha = Math.min(1, dense[i] * 0.5);
          ctx.strokeStyle = `rgba(100, 200, 255, ${alpha})`;
          ctx.beginPath();
          ctx.moveTo(densePositions[i].x, densePositions[i].y);
          ctx.lineTo(outPositions[j].x, outPositions[j].y);
          ctx.stroke();
        }
      }
    }
    for (let i = 0; i < 64; i++) {
      const intensity = Math.min(255, Math.floor(dense[i] * 255));
      ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity + 50})`;
      ctx.beginPath();
      ctx.arc(
        densePositions[i].x,
        densePositions[i].y,
        nodeSize / 2,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
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
    const fs = this.vision.frameSize;
    this.stateBytes = new Uint8Array(CONFIG.batchSize * fs);
    this.actions = new Float32Array(CONFIG.batchSize * 3);
    this.rewards = new Float32Array(CONFIG.batchSize);

    this.episodeStats = { count: 0, baskets: 0, shots: 0 };
    this.accuracyHistory = [];
    this.isTrainingStep = false;
    this.activeCount = 0;
    this.exploreNoise = CONFIG.exploreNoise;
    // Objects that must be hidden from the agents' cameras during capture.
    this.hiddenDuringCapture = [];
  }

  // Spawn, capture vision, predict, and launch a fresh batch.
  resetBatch() {
    for (const b of this.balls) b.spawn();
    this.activeCount = this.balls.length;
    this.dashboard.setBatchProgress("Simulating...");

    this.vision.captureBatch(
      this.balls,
      this.hoopPos,
      this.court,
      this.hiddenDuringCapture,
      this.stateBytes
    );
    this.agent.predictBatch(
      this.stateBytes,
      this.balls.length,
      this.actions,
      this.exploreNoise
    );
    for (let i = 0; i < this.balls.length; i++)
      this.balls[i].launch(this.actions, i * 3, this.hoopPos);
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
      this.accuracyHistory.push(b.scored ? 1 : 0);
      if (this.accuracyHistory.length > CONFIG.accuracyWindow)
        this.accuracyHistory.shift();
    }

    const rollingAcc =
      this.accuracyHistory.length > 0
        ? this.accuracyHistory.reduce((a, b) => a + b, 0) /
          this.accuracyHistory.length
        : 0;
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
    this.dashboard.drawAgentView(this.stateBytes, best);
    this.dashboard.visualizeActivations(
      this.agent.getActivations(this.stateBytes, best)
    );
    this.dashboard.visualizeKernels(this.agent);
    this.dashboard.setAgentDecision(
      this.actions,
      best * 3,
      this.rewards[best],
      this.balls[best].scored
    );

    // finally, not a trailing assignment: finishBatch is fired without being
    // awaited, so anything that escaped from here would leave isTrainingStep
    // latched and stall training permanently with no visible error.
    try {
      const loss = await this.agent.trainOnBatch(
        this.stateBytes,
        this.actions,
        this.rewards,
        n
      );
      if (loss != null) this.dashboard.pushLoss(loss);

      // Anneal exploration once per trained batch so the policy tightens toward
      // exploitation as it improves, with a floor that keeps a little jitter.
      this.exploreNoise = Math.max(
        CONFIG.exploreNoiseMin,
        this.exploreNoise * CONFIG.exploreNoiseDecay
      );
    } catch (e) {
      console.error("Train Error", e);
    } finally {
      // Relaunching in finally so a failed training step costs one batch
      // rather than wedging the simulation: the loop only advances again once
      // there are live balls to advance.
      this.resetBatch();
      this.isTrainingStep = false;
    }
    return n;
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
        if (this.trainingMode) this.arena.resetBatch();
        else for (const b of this.arena.balls) b.retire();
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
    // and the balls stay hidden until the user starts training (resetBatch
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
    // Clamped so a long training stall (or a backgrounded tab) doesn't make
    // playback jump — the reel should always look like real-time basketball.
    const dt = Math.min(0.1, (now - this.lastFrame) / 1000);
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

    this.manual.update();
    this.showcase.refresh(now);
    this.showcase.update(dt);
    this.dashboard.tickRates(retired, now);
    this.sceneMgr.render();
  }
}

new App().start();
