import * as THREE from "https://esm.sh/three@0.132.2";
import { OrbitControls } from "https://esm.sh/three@0.132.2/examples/jsm/controls/OrbitControls.js";
import * as CANNON from "https://esm.sh/cannon-es@0.19.0";

const CONFIG = {
  // Balls simulated, rendered and trained on per batch. The whole batch is on
  // the court at once — watching a thousand shots converge IS the demo — so
  // batchSize is the fixed quantity here and everything else is budgeted
  // around it. Two things make 1024 affordable:
  //   - the batch is drawn as a single InstancedMesh (see BallField), so 1024
  //     balls cost one draw call per pass instead of 1024;
  //   - the vision atlas is captured in bounded passes (see VisionSystem), so
  //     GPU memory no longer scales with batchSize.
  batchSize: 1024,
  // Retina resolution per eye. World units are FEET throughout (the court is
  // PlaneGeometry(94, 50), rim at 10ft, ball radius 0.4ft ~ 4.8in), which is
  // what makes the eye geometry below comparable to a real head.
  //
  // This is the knob that pays for the batch size: every buffer in the
  // pipeline scales with batchSize * W * H * 2, so resolution and ball count
  // trade off directly against each other. At 96px the state tensor is ~75MB
  // (1024 * 96 * 96 * 2 * 4B) — the same footprint the smaller-batch,
  // higher-resolution configs were tuned to.
  //
  // 96px is also about the floor for the stereo channel to mean anything. One
  // pixel of disparity from the 63mm IPD below needs ipd * f_px / d >= 1,
  // where f_px = (W/2) / tan(fov/2) = 83.1 at 96px / 60deg: that is ~1.7px at
  // 10ft and ~0.86px at 20ft. Mid-range shots clear it, the far corners do
  // not. Going below ~64px makes disparity sub-pixel everywhere and the second
  // eye becomes decorative.
  visionWidth: 96,
  visionHeight: 96,
  // Cap on either dimension of the stereo capture atlas. batchSize tiles of
  // visionWidth would be a 3072px atlas at these settings and an 8192px one at
  // 256px/eye — past MAX_TEXTURE_SIZE on plenty of GPUs (and 512MB of render
  // target). VisionSystem instead fills an atlas of at most this size and
  // repeats the fill/readback until the batch is covered, so no config choice
  // can blow out GPU memory.
  maxAtlasDim: 4096,
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
    // Per-instance tints multiplied over batchBall's color. Balls that have
    // finished their shot stay on the court until the next batch spawns —
    // the scatter of where a thousand shots ended up is the clearest picture
    // of what the policy currently does — so they need to read as spent
    // rather than in flight.
    this.ballTint = {
      live: new THREE.Color(1, 1, 1),
      spent: new THREE.Color(0.3, 0.22, 0.16)
    };

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
    // The pending batch, set by store() and consumed by train(). One object
    // for the whole batch, not one per sample.
    this.memory = null;
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

  // pixelDataBatch: Float32Array of size count * frameSize (frameSize = W*H*2).
  predictBatch(pixelDataBatch, noiseScale = 0) {
    return tf.tidy(() => {
      // Derive the batch count from the data length instead of assuming a fixed
      // batch size, so a partial or resized batch reshapes correctly.
      const count = pixelDataBatch.length / this.frameSize;
      const stateTensor = tf.tensor(pixelDataBatch, [
        count,
        this.visionH,
        this.visionW,
        this.channels
      ]);
      const data = this.actor.predict(stateTensor).dataSync(); // count * 3
      // Clamp to [-1, 1] after adding exploration noise: the actor's output
      // is tanh-bounded, and these actions are later stored as regression
      // targets, so out-of-range values would be unreachable training goals.
      const clamp = (v) => Math.max(-1, Math.min(1, v));
      const actions = [];
      for (let i = 0; i < count; i++) {
        actions.push([
          clamp(data[i * 3 + 0] + (Math.random() - 0.5) * noiseScale),
          clamp(data[i * 3 + 1] + (Math.random() - 0.5) * noiseScale),
          clamp(data[i * 3 + 2] + (Math.random() - 0.5) * noiseScale)
        ]);
      }
      return actions;
    });
  }

  getActivations(pixelData) {
    return tf.tidy(() => {
      let t = tf.tensor(pixelData, [
        1,
        this.visionH,
        this.visionW,
        this.channels
      ]);
      for (let i = 0; i <= 3; i++) t = this.actor.layers[i].apply(t);
      const dense64 = t;
      const output3 = this.actor.layers[4].apply(dense64);
      return { dense: dense64.dataSync(), output: output3.dataSync() };
    });
  }

  // Takes the batch whole: states is one contiguous Float32Array holding
  // count * frameSize pixels in the same order as actions and rewards. Storing
  // per-sample slices instead meant a full extra copy of the batch here and
  // another in train() to concatenate them back — 150MB of copying per batch
  // at batchSize 1024, for data that arrives contiguous already.
  store(states, actions, rewards) {
    this.memory = { states, actions, rewards, count: actions.length };
  }

  // Trains critic on returns, then actor on advantage-positive samples.
  // Returns the critic loss (or null if nothing to train on).
  async train() {
    if (!this.memory) return null;
    const { states, actions, rewards, count: batchSize } = this.memory;

    const stateTensor = tf.tensor(states, [
      batchSize,
      this.visionH,
      this.visionW,
      this.channels
    ]);
    const actionTensor = tf.tensor2d(actions, [batchSize, 3]);
    const rewardTensor = tf.tensor2d(rewards, [batchSize, 1]);

    const criticHistory = await this.critic.fit(stateTensor, rewardTensor, {
      epochs: 1,
      verbose: 0
    });
    const loss = criticHistory.history.loss ? criticHistory.history.loss[0] : 0;

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
    const values = this.critic.predict(stateTensor);
    const advantages = rewardTensor.sub(values);
    const advantageData = advantages.dataSync();

    let advMean = 0;
    for (let i = 0; i < batchSize; i++) advMean += advantageData[i];
    advMean /= batchSize;
    let advVar = 0;
    for (let i = 0; i < batchSize; i++)
      advVar += (advantageData[i] - advMean) ** 2;
    const advStd = Math.sqrt(advVar / batchSize) || 1;

    const weightData = new Float32Array(batchSize);
    for (let i = 0; i < batchSize; i++) {
      const z = (advantageData[i] - advMean) / advStd;
      weightData[i] = Math.min(
        CONFIG.advantageClip,
        Math.exp(z / CONFIG.advantageTemp)
      );
    }
    const weightTensor = tf.tensor1d(weightData);

    // Sync the shared conv layers (critic -> actor) BEFORE fitting the actor.
    // The actor's conv layers are frozen, so actor.fit() only adjusts its dense
    // head — and that head should be trained against the same features the
    // actor will use at inference. Syncing after the fit would instead train
    // the head on the previous batch's stale features, then swap the backbone
    // out from under it.
    tf.tidy(() => {
      for (let i = 0; i < 2; i++)
        this.actor.layers[i].setWeights(this.critic.layers[i].getWeights());
    });

    this._fitActorWeighted(stateTensor, actionTensor, weightTensor);

    stateTensor.dispose();
    actionTensor.dispose();
    rewardTensor.dispose();
    values.dispose();
    advantages.dispose();
    weightTensor.dispose();
    this.memory = null;
    return loss;
  }

  // One epoch of minibatch Adam over a per-sample weighted MSE. model.fit()
  // cannot do this: passing sampleWeight throws "Support sampleWeight is not
  // implemented yet" in tfjs 4.x, so the update is driven explicitly. Minibatch
  // size matches fit()'s default of 32 so the number of optimizer steps per
  // batch is unchanged. Only trainableWeights are passed to minimize(), which
  // already excludes the two frozen conv layers.
  _fitActorWeighted(states, actions, weights, minibatch = 32) {
    const n = states.shape[0];
    const order = new Int32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const t = order[i];
      order[i] = order[j];
      order[j] = t;
    }

    const vars = this.actor.trainableWeights.map((w) => w.val);
    for (let start = 0; start < n; start += minibatch) {
      const slice = order.slice(start, Math.min(start + minibatch, n));
      tf.tidy(() => {
        const idx = tf.tensor1d(slice, "int32");
        const xb = tf.gather(states, idx);
        const yb = tf.gather(actions, idx);
        const wb = tf.gather(weights, idx);
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
    // Finished balls stay on the court until the next batch spawns. Ball.retire
    // sleeps them, and cannon-es skips both integration and broadphase pairs
    // for sleeping bodies — so as a batch plays out its physics cost drains
    // away instead of staying at batchSize until the last shot lands.
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

// Every ball in the batch is on screen at once, so the batch is drawn as one
// InstancedMesh: a single draw call (plus one per shadow map) for the whole
// batch instead of one per ball. A Mesh per ball put the batch through the
// draw-call path batchSize times per pass, and with two shadow-casting lights
// there are three passes — so the cost of the thing this demo most wants to
// scale up grew with exactly the number it wants to make large. Instanced, the
// entire scene renders in 19 draw calls at batchSize 1024.
//
// Instances are hidden by zero-scaling them rather than by a visible flag,
// which instancing has no per-instance equivalent of.
class BallField {
  constructor(scene, assets, radius, count) {
    this.mesh = new THREE.InstancedMesh(
      new THREE.SphereGeometry(radius, 12, 12),
      assets.batchBall,
      count
    );
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.castShadow = true;
    // Instances move every frame and the mesh's bounding sphere is computed
    // once from the initial matrices, so culling against it would pop the
    // whole batch out of view. One field, always drawn.
    this.mesh.frustumCulled = false;
    this.tint = assets.ballTint;
    this._xf = new THREE.Object3D();
    this._matrixDirty = false;
    this._colorDirty = false;

    // Dormant until the first spawn(). Without this every instance defaults to
    // an identity matrix, so the whole batch would clump at center court until
    // training first spawns them.
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let i = 0; i < count; i++) {
      this.mesh.setMatrixAt(i, zero);
      this.mesh.setColorAt(i, this.tint.live);
    }
    scene.add(this.mesh);
  }

  setTransform(i, position, quaternion) {
    const xf = this._xf;
    xf.position.copy(position);
    xf.quaternion.copy(quaternion);
    xf.scale.set(1, 1, 1);
    xf.updateMatrix();
    this.mesh.setMatrixAt(i, xf.matrix);
    this._matrixDirty = true;
  }

  setLive(i, live) {
    this.mesh.setColorAt(i, live ? this.tint.live : this.tint.spent);
    this._colorDirty = true;
  }

  setVisible(on) {
    this.mesh.visible = on;
  }

  // Instance buffers are uploaded once per frame, not once per ball — and not
  // at all on frames where no ball moved, which is every frame between the
  // last shot landing and the next batch spawning.
  flush() {
    if (this._matrixDirty) {
      this.mesh.instanceMatrix.needsUpdate = true;
      this._matrixDirty = false;
    }
    if (this._colorDirty) {
      this.mesh.instanceColor.needsUpdate = true;
      this._colorDirty = false;
    }
  }
}

class Ball {
  constructor(id, field, physics, radius) {
    this.id = id;
    this.field = field;
    // Instance transform, mirrored here because the instance matrix is
    // write-only as far as the rest of the app is concerned.
    this.position = new THREE.Vector3();
    this.quaternion = new THREE.Quaternion();

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
    this.action = [0, 0, 0];
    this.path = [];
    this.minDist = 100;
    this.scored = false;
    this.hitBackboard = false;
    this.hitRim = false;
  }

  // Position at a random launch spot and mark active/visible.
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
    this.syncMesh();
    this.field.setLive(this.id, true);
  }

  syncMesh() {
    this.position.copy(this.body.position);
    this.quaternion.copy(this.body.quaternion);
    this.field.setTransform(this.id, this.position, this.quaternion);
  }

  // The shot is over. The ball keeps its landing spot on the court until the
  // next batch spawns — it just stops being simulated and dims. Sleeping the
  // body is what freezes it: cannon-es skips integration and broadphase for
  // sleeping bodies, so a finished ball costs nothing while the rest of the
  // batch is still in the air.
  retire() {
    this.active = false;
    this.body.sleep();
    this.field.setLive(this.id, false);
  }

  // Apply the launch impulse derived from the agent's 3-vector action.
  launch(action, hoopPos) {
    this.action = action;
    const start = this.position.clone();
    const dirToHoop = new THREE.Vector3().subVectors(hoopPos, start);
    dirToHoop.y = 0;
    dirToHoop.normalize();
    const dirSide = new THREE.Vector3().crossVectors(UP, dirToHoop).normalize();

    const magFwd = (action[0] + 1) * 50;
    const magUp = (action[1] + 1) * 35 + 20;
    const magSide = action[2] * 20;

    const impulse = new THREE.Vector3()
      .add(dirToHoop.multiplyScalar(magFwd))
      .add(new THREE.Vector3(0, 1, 0).multiplyScalar(magUp))
      .add(dirSide.multiplyScalar(magSide));

    this.body.wakeUp();
    this.body.applyImpulse(
      new CANNON.Vec3(impulse.x, impulse.y, impulse.z),
      new CANNON.Vec3(0, 0, 0)
    );
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
    // grid packed into ONE render target per eye, so a single readback per eye
    // covers many balls at once. The alternative — two synchronous
    // readRenderTargetPixels() per ball — was 2 * batchSize GPU->CPU pipeline
    // flushes per batch and the dominant per-batch stall.
    //
    // The atlas is capped at CONFIG.maxAtlasDim rather than sized to hold the
    // whole batch. Sizing it to batchSize made GPU memory scale with the one
    // number this demo wants to push: a 1024-ball atlas at 256px/eye is
    // 8192x8192, i.e. 256MB per eye of render target plus the same again in
    // readback buffers, and past MAX_TEXTURE_SIZE on a lot of hardware. A
    // capped atlas is filled and read back as many times as it takes
    // (this.passes), which costs a few extra flushes per batch and makes the
    // footprint independent of batchSize.
    const maxCols = Math.floor(CONFIG.maxAtlasDim / W);
    const maxRows = Math.floor(CONFIG.maxAtlasDim / H);
    this.cols = Math.max(1, Math.min(Math.ceil(Math.sqrt(batchSize)), maxCols));
    this.rows = Math.max(
      1,
      Math.min(Math.ceil(batchSize / this.cols), maxRows)
    );
    this.tilesPerPass = this.cols * this.rows;
    this.passes = Math.ceil(batchSize / this.tilesPerPass);
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
  }

  // Renders each ball's stereo view toward the hoop and packs grayscale L/R
  // channels into one flat Float32Array [balls.length * frameSize]. The output
  // packing order is identical to the old per-target path, so downstream
  // consumers (agent state storage, drawAgentView) are unaffected.
  captureBatch({
    balls,
    ballField,
    hoopPos,
    court,
    manualMesh,
    trajectoryGroup
  }) {
    const { tilesPerPass } = this;
    const batch = new Float32Array(balls.length * this.frameSize);

    // Hide everything that shouldn't appear in the agents' vision. The whole
    // batch is one instanced mesh, so hiding the balls is a single flag.
    const wasVizVisible = trajectoryGroup.visible;
    const wasFieldVisible = ballField.visible;
    trajectoryGroup.visible = false;
    manualMesh.visible = false;
    ballField.visible = false;
    court.setHighContrast(true);

    // Freeze the shadow maps for the duration of the capture. Every render()
    // redraws them by default, and the capture issues 2 * batchSize renders —
    // 2048 full shadow passes over a scene that does not move, since the balls
    // are hidden and the court is static throughout. autoUpdate = false with
    // needsUpdate = true refreshes them exactly once, on the first render
    // below, so the agents see shadows consistent with the high-contrast scene
    // they're being shown rather than 2047 redundant redraws of it.
    const wasAutoUpdate = this.renderer.shadowMap.autoUpdate;
    this.renderer.shadowMap.autoUpdate = false;
    this.renderer.shadowMap.needsUpdate = true;

    for (let start = 0; start < balls.length; start += tilesPerPass) {
      const end = Math.min(start + tilesPerPass, balls.length);
      this._renderPass(balls, start, end, hoopPos);
      this._readPass(batch, start, end);
    }

    // Restore.
    this.renderer.shadowMap.autoUpdate = wasAutoUpdate;
    this.renderer.shadowMap.needsUpdate = true;
    court.setHighContrast(false);
    trajectoryGroup.visible = wasVizVisible;
    ballField.visible = wasFieldVisible;
    manualMesh.visible = true;
    // Unbinding restores the canvas viewport/scissor from the renderer globals,
    // which this path never touches, so there is nothing else to put back.
    this.renderer.setRenderTarget(null);

    return batch;
  }

  // Fills both atlases with the stereo views of balls[start, end).
  _renderPass(balls, start, end, hoopPos) {
    const { W, H, cols } = this;
    const half = CONFIG.ipd / 2;
    const dir = new THREE.Vector3();
    const rightVec = new THREE.Vector3();
    const offset = new THREE.Vector3();

    for (let i = start; i < end; i++) {
      const pos = balls[i].position;
      dir.subVectors(hoopPos, pos).normalize();
      rightVec.crossVectors(dir, UP).normalize();
      offset.copy(rightVec).multiplyScalar(half);

      this.camLeft.position.copy(pos).sub(offset);
      this.camRight.position.copy(pos).add(offset);
      this.camLeft.lookAt(hoopPos);
      this.camRight.lookAt(hoopPos);

      const tile = i - start;
      const tileX = (tile % cols) * W;
      const tileY = Math.floor(tile / cols) * H;

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

  // Reads both atlases back and unpacks the tiles for balls[start, end) into
  // the flat batch array. One readback per eye covers the whole pass.
  _readPass(batch, start, end) {
    const { W, H, cols, atlasW, atlasH, frameSize } = this;

    this.renderer.readRenderTargetPixels(
      this.rtLeft,
      0,
      0,
      atlasW,
      atlasH,
      this.bufLeft
    );
    this.renderer.readRenderTargetPixels(
      this.rtRight,
      0,
      0,
      atlasW,
      atlasH,
      this.bufRight
    );

    for (let i = start; i < end; i++) {
      const tile = i - start;
      const tileX = (tile % cols) * W;
      const tileY = Math.floor(tile / cols) * H;
      const offset = i * frameSize;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const p = y * W + x;
          const a = ((tileY + y) * atlasW + (tileX + x)) * 4;
          const grayL =
            (0.299 * this.bufLeft[a] +
              0.587 * this.bufLeft[a + 1] +
              0.114 * this.bufLeft[a + 2]) /
            255.0;
          const grayR =
            (0.299 * this.bufRight[a] +
              0.587 * this.bufRight[a + 1] +
              0.114 * this.bufRight[a + 2]) /
            255.0;
          batch[offset + p * 2] = grayL;
          batch[offset + p * 2 + 1] = grayR;
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

  add(points, type, reward) {
    if (points.length < 2) return;

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
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
    sprite.position.copy(points[points.length - 1]);
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
    this.uiAcc = document.getElementById("accuracy");
    this.uiStatus = document.getElementById("status");
    this.uiBatch = document.getElementById("batch-progress");

    this.lossHistory = [];
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

  setStats({ accuracy, baskets, episodes }) {
    this.uiAcc.innerText = Math.round(accuracy * 100) + "%";
    this.uiBaskets.innerText = baskets;
    this.uiEp.innerText = episodes;
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

  drawAgentView(batch) {
    const W = CONFIG.visionWidth;
    const H = CONFIG.visionHeight;
    const dW = W * 2; // combined L|R width
    const img = this.agentViewCtx.createImageData(dW, H);
    for (let i = 0; i < W * H; i++) {
      const grayL = batch[i * 2];
      const grayR = batch[i * 2 + 1];
      const row = Math.floor(i / W);
      const col = i % W;
      const invRow = H - 1 - row;

      const idxL = (invRow * dW + col) * 4;
      const valL = grayL * 255;
      img.data[idxL] = valL;
      img.data[idxL + 1] = valL;
      img.data[idxL + 2] = valL;
      img.data[idxL + 3] = 255;

      const idxR = (invRow * dW + (col + W)) * 4;
      const valR = grayR * 255;
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
    const labels = ["V", "F", "A"];
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
  constructor({
    agent,
    scene,
    physics,
    assets,
    court,
    vision,
    dashboard,
    trajectory
  }) {
    this.agent = agent;
    this.court = court;
    this.vision = vision;
    this.dashboard = dashboard;
    this.trajectory = trajectory;

    this.hoopPos = court.rimPosition;
    this.hoopPosCannon = court.rimPositionCannon;

    this.field = new BallField(
      scene,
      assets,
      CONFIG.ballRadius,
      CONFIG.batchSize
    );
    this.balls = [];
    this.bodyToBall = new Map();
    for (let i = 0; i < CONFIG.batchSize; i++) {
      const b = new Ball(i, this.field, physics, CONFIG.ballRadius);
      this.balls.push(b);
      this.bodyToBall.set(b.body, b);
    }

    // The current batch's stereo frames: one contiguous Float32Array holding
    // every ball's launch view, in ball order, handed to the agent as-is at the
    // end of the batch. Keeping per-ball copies of the same pixels cost a
    // second full copy of the batch here and a third in the training path,
    // which is 150MB of pointless copying per batch at batchSize 1024.
    this.batchPixels = null;

    this.episodeStats = { count: 0, baskets: 0, shots: 0 };
    this.accuracyHistory = [];
    this.isTrainingStep = false;
    this.exploreNoise = CONFIG.exploreNoise;
  }

  // Position all balls but don't launch (initial state / warm-up).
  spawnAll() {
    this.field.setVisible(true);
    for (const b of this.balls) b.spawn();
    this.field.flush();
    this.dashboard.setBatchProgress("Simulating...");
  }

  // Leaving training mode. Retire whatever is still in the air so 1024 bodies
  // aren't simulated invisibly, and take the batch off the court so manual
  // mode gets a clean floor instead of the last batch's landing scatter.
  halt() {
    for (const b of this.balls) if (b.active) b.retire();
    this.field.setVisible(false);
  }

  // Spawn, capture vision, predict, and launch a fresh batch.
  resetBatch(manualMesh) {
    this.spawnAll();

    const batch = this.vision.captureBatch({
      balls: this.balls,
      ballField: this.field.mesh,
      hoopPos: this.hoopPos,
      court: this.court,
      manualMesh,
      trajectoryGroup: this.trajectory.group
    });
    this.batchPixels = batch;
    const actions = this.agent.predictBatch(batch, this.exploreNoise);
    for (let i = 0; i < this.balls.length; i++)
      this.balls[i].launch(actions[i], this.hoopPos);
    this.field.flush();

    // Ball 0 stands in for the batch in the activation panel.
    this.dashboard.visualizeActivations(
      this.agent.getActivations(batch.slice(0, this.vision.frameSize))
    );
    this.dashboard.visualizeKernels(this.agent);
    this.dashboard.drawAgentView(batch);
  }

  // Per-frame update; returns the number of finished (inactive) balls.
  update() {
    let finished = 0;
    for (const b of this.balls) {
      if (!b.active) {
        finished++;
        continue;
      }
      b.syncMesh();
      b.path.push(b.position.clone());

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
      const isTimeout = b.path.length > 600;

      if (
        b.scored ||
        isOnFloor ||
        isOOB ||
        (isStopped && b.path.length > 10) ||
        isTimeout
      ) {
        b.retire();
      }
    }
    // One instance-buffer upload per frame covers every ball that moved.
    this.field.flush();
    return finished;
  }

  _reward(b) {
    let shotDistance = 0;
    if (b.path.length > 0) {
      const launchPos = b.path[0];
      shotDistance = Math.sqrt(
        Math.pow(launchPos.x - this.hoopPos.x, 2) +
          Math.pow(launchPos.z - this.hoopPos.z, 2)
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

  // Compute rewards, store transitions, train, then relaunch. Re-entrant-guarded.
  async finishBatch(manualMesh) {
    if (this.isTrainingStep) return;
    this.isTrainingStep = true;

    const actions = [];
    const rewards = [];
    for (const b of this.balls) {
      const { reward, type } = this._reward(b);
      if (b.scored) this.episodeStats.baskets++;
      this.trajectory.add(b.path, type, reward);
      actions.push(b.action);
      rewards.push(reward);
      this.episodeStats.shots++;
      this.episodeStats.count++;
      this.accuracyHistory.push(b.scored ? 1 : 0);
      if (this.accuracyHistory.length > CONFIG.accuracyWindow)
        this.accuracyHistory.shift();
    }
    // The states are already contiguous and in ball order in batchPixels, so
    // the agent takes the whole array by reference instead of rebuilding it.
    this.agent.store(this.batchPixels, actions, rewards);

    const rollingAcc =
      this.accuracyHistory.length > 0
        ? this.accuracyHistory.reduce((a, b) => a + b, 0) /
          this.accuracyHistory.length
        : 0;
    this.dashboard.setStats({
      accuracy: rollingAcc,
      baskets: this.episodeStats.baskets,
      episodes: this.episodeStats.count
    });
    this.dashboard.setBatchProgress("Training GPU...");

    try {
      const loss = await this.agent.train();
      if (loss != null) this.dashboard.pushLoss(loss);
    } catch (e) {
      console.error("Train Error", e);
    }

    // Anneal exploration once per trained batch so the policy tightens toward
    // exploitation as it improves, with a floor that keeps a little jitter.
    this.exploreNoise = Math.max(
      CONFIG.exploreNoiseMin,
      this.exploreNoise * CONFIG.exploreNoiseDecay
    );

    this.resetBatch(manualMesh);
    this.isTrainingStep = false;
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

  setEnabled(v) {
    this.enabled = v;
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
      if (!this.arena) return;
      if (this.trainingMode) this.arena.resetBatch(this.manual.mesh);
      else this.arena.halt();
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
      scene: this.sceneMgr.scene,
      physics: this.physics,
      assets: this.assets,
      court: this.court,
      vision: this.vision,
      dashboard: this.dashboard,
      trajectory: this.trajectory
    });

    // Don't spawn the training batch at startup — the app opens in manual mode
    // and the balls stay hidden until the user starts training (resetBatch
    // spawns them). This avoids a startup flash of the whole batch.
    this.manual.reset();
    this._loop();
  }

  _loop() {
    requestAnimationFrame(() => this._loop());
    this.physics.step();

    if (this.arena) {
      const finished = this.arena.update();
      if (this.trainingMode && !this.arena.isTrainingStep) {
        if (finished === CONFIG.batchSize)
          this.arena.finishBatch(this.manual.mesh);
        else
          this.dashboard.setBatchProgress(
            `${CONFIG.batchSize - finished} Active`
          );
      }
    }

    this.manual.update();
    this.sceneMgr.render();
  }
}

new App().start();
