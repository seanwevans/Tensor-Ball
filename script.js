import * as THREE from "https://esm.sh/three@0.132.2";
import { OrbitControls } from "https://esm.sh/three@0.132.2/examples/jsm/controls/OrbitControls.js";
import * as CANNON from "https://esm.sh/cannon-es@0.19.0";

const CONFIG = {
  batchSize: 2048,
  visionWidth: 64,
  visionHeight: 64,
  ballRadius: 0.4,
  learningRate: 0.001,
  l2: 0.001,
  maxHistory: 100,
  accuracyWindow: 1024,
  exploreNoise: 0.4,
  ipd: 0.2, // stereo interpupillary distance
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
    this.actor = this._buildActor();
    this.critic = this._buildCritic();
    this.memory = [];
  }

  // Shared conv stack: conv(8) -> conv(16) -> flatten -> dense(64).
  // Layer indices [0],[1] are the two conv layers in BOTH nets — the weight
  // sync in train() depends on that alignment.
  _convBase(model) {
    model.add(
      tf.layers.conv2d({
        inputShape: [64, 64, 2],
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
        kernelRegularizer: this.l2Reg
      })
    );
  }

  _buildActor() {
    const m = tf.sequential();
    this._convBase(m);
    // The critic owns the shared conv backbone: train() copies the critic's
    // conv weights into the actor after every batch. Freeze the actor's two
    // conv layers so actor.fit() doesn't waste work computing gradients that
    // get overwritten, and so the actor's dense head trains against a stable
    // feature extractor.
    m.layers[0].trainable = false;
    m.layers[1].trainable = false;
    m.add(
      tf.layers.dense({
        units: 3,
        activation: "tanh",
        kernelRegularizer: this.l2Reg
      })
    );
    m.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: "meanSquaredError"
    });
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

  // pixelDataBatch: Float32Array of size batchSize * 64 * 64 * 2
  predictBatch(pixelDataBatch, noiseScale = 0) {
    return tf.tidy(() => {
      const stateTensor = tf.tensor(pixelDataBatch, [
        this.batchSize,
        64,
        64,
        2
      ]);
      const data = this.actor.predict(stateTensor).dataSync(); // batchSize * 3
      // Clamp to [-1, 1] after adding exploration noise: the actor's output
      // is tanh-bounded, and these actions are later stored as regression
      // targets, so out-of-range values would be unreachable training goals.
      const clamp = (v) => Math.max(-1, Math.min(1, v));
      const actions = [];
      for (let i = 0; i < this.batchSize; i++) {
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
      let t = tf.tensor(pixelData, [1, 64, 64, 2]);
      for (let i = 0; i <= 3; i++) t = this.actor.layers[i].apply(t);
      const dense64 = t;
      const output3 = this.actor.layers[4].apply(dense64);
      return { dense: dense64.dataSync(), output: output3.dataSync() };
    });
  }

  store(pixelData, action, reward) {
    this.memory.push({ state: pixelData, action, reward });
  }

  // Trains critic on returns, then actor on advantage-positive samples.
  // Returns the critic loss (or null if nothing to train on).
  async train() {
    if (this.memory.length === 0) return null;
    const batchSize = this.memory.length;

    const stateData = new Float32Array(batchSize * 64 * 64 * 2);
    for (let i = 0; i < batchSize; i++)
      stateData.set(this.memory[i].state, i * 64 * 64 * 2);
    const stateTensor = tf.tensor(stateData, [batchSize, 64, 64, 2]);
    const actionTensor = tf.tensor2d(
      this.memory.map((m) => m.action),
      [batchSize, 3]
    );
    const rewardTensor = tf.tensor2d(
      this.memory.map((m) => m.reward),
      [batchSize, 1]
    );

    const criticHistory = await this.critic.fit(stateTensor, rewardTensor, {
      epochs: 1,
      verbose: 0
    });
    const loss = criticHistory.history.loss ? criticHistory.history.loss[0] : 0;

    // Actor: fit toward actions that beat the critic's value estimate.
    const values = this.critic.predict(stateTensor);
    const advantages = rewardTensor.sub(values);
    const advantageData = advantages.dataSync();
    const goodIndices = [];
    for (let i = 0; i < batchSize; i++)
      if (advantageData[i] > 0) goodIndices.push(i);

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

    if (goodIndices.length > 0) {
      const idx = tf.tensor1d(goodIndices, "int32");
      const goodStateTensor = tf.gather(stateTensor, idx);
      const goodActionTensor = tf.gather(actionTensor, idx);
      await this.actor.fit(goodStateTensor, goodActionTensor, {
        epochs: 1,
        verbose: 0
      });
      idx.dispose();
      goodStateTensor.dispose();
      goodActionTensor.dispose();
    }

    stateTensor.dispose();
    actionTensor.dispose();
    rewardTensor.dispose();
    values.dispose();
    advantages.dispose();
    this.memory = [];
    return loss;
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

class Ball {
  constructor(id, scene, physics, assets, radius) {
    this.id = id;
    this.mesh = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 16, 16),
      assets.batchBall
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

    this._resetState();
  }

  _resetState() {
    this.active = false;
    this.startPixels = null;
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
    this.mesh.position.copy(this.body.position);
    this.mesh.quaternion.copy(this.body.quaternion);

    this._resetState();
    this.active = true;
    this.mesh.visible = true;
  }

  syncMesh() {
    this.mesh.position.copy(this.body.position);
    this.mesh.quaternion.copy(this.body.quaternion);
  }

  // Apply the launch impulse derived from the agent's 3-vector action.
  launch(action, hoopPos) {
    this.action = action;
    const start = this.mesh.position.clone();
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
  constructor(renderer, scene) {
    this.renderer = renderer;
    this.scene = scene;
    const W = CONFIG.visionWidth;
    const H = CONFIG.visionHeight;
    this.W = W;
    this.H = H;
    this.frameSize = W * H * 2;

    this.rtLeft = new THREE.WebGLRenderTarget(W, H);
    this.rtRight = new THREE.WebGLRenderTarget(W, H);
    this.camLeft = new THREE.PerspectiveCamera(90, 1, 0.1, 200);
    this.camRight = new THREE.PerspectiveCamera(90, 1, 0.1, 200);
    this.bufLeft = new Uint8Array(W * H * 4);
    this.bufRight = new Uint8Array(W * H * 4);
  }

  // Renders each ball's stereo view toward the hoop and packs grayscale L/R
  // channels into one flat Float32Array [balls.length * frameSize].
  captureBatch(balls, hoopPos, court, manualMesh, trajectoryGroup) {
    const { W, H, frameSize } = this;
    const batch = new Float32Array(balls.length * frameSize);

    // Hide everything that shouldn't appear in the agents' vision.
    const wasVizVisible = trajectoryGroup.visible;
    trajectoryGroup.visible = false;
    manualMesh.visible = false;
    for (const b of balls) b.mesh.visible = false;
    court.setHighContrast(true);

    const half = CONFIG.ipd / 2;
    for (let i = 0; i < balls.length; i++) {
      const pos = balls[i].mesh.position;
      const dir = new THREE.Vector3().subVectors(hoopPos, pos).normalize();
      const rightVec = new THREE.Vector3().crossVectors(dir, UP).normalize();

      this.camLeft.position
        .copy(pos)
        .sub(rightVec.clone().multiplyScalar(half));
      this.camRight.position
        .copy(pos)
        .add(rightVec.clone().multiplyScalar(half));
      this.camLeft.lookAt(hoopPos);
      this.camRight.lookAt(hoopPos);

      this.renderer.setRenderTarget(this.rtLeft);
      this.renderer.render(this.scene, this.camLeft);
      this.renderer.setRenderTarget(this.rtRight);
      this.renderer.render(this.scene, this.camRight);

      this.renderer.readRenderTargetPixels(
        this.rtLeft,
        0,
        0,
        W,
        H,
        this.bufLeft
      );
      this.renderer.readRenderTargetPixels(
        this.rtRight,
        0,
        0,
        W,
        H,
        this.bufRight
      );

      const offset = i * frameSize;
      for (let p = 0; p < W * H; p++) {
        const grayL =
          (0.299 * this.bufLeft[p * 4] +
            0.587 * this.bufLeft[p * 4 + 1] +
            0.114 * this.bufLeft[p * 4 + 2]) /
          255.0;
        const grayR =
          (0.299 * this.bufRight[p * 4] +
            0.587 * this.bufRight[p * 4 + 1] +
            0.114 * this.bufRight[p * 4 + 2]) /
          255.0;
        batch[offset + p * 2] = grayL;
        batch[offset + p * 2 + 1] = grayR;
      }
    }

    // Restore.
    court.setHighContrast(false);
    trajectoryGroup.visible = wasVizVisible;
    for (const b of balls) b.mesh.visible = true;
    manualMesh.visible = true;
    this.renderer.setRenderTarget(null);

    return batch;
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
    this.agentViewCtx = document
      .getElementById("agent-view-canvas")
      .getContext("2d");
    this.kernelCtx = document.getElementById("kernel-canvas").getContext("2d");
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
    const img = this.agentViewCtx.createImageData(128, 64);
    for (let i = 0; i < W * H; i++) {
      const grayL = batch[i * 2];
      const grayR = batch[i * 2 + 1];
      const row = Math.floor(i / W);
      const col = i % W;
      const invRow = H - 1 - row;

      const idxL = (invRow * 128 + col) * 4;
      const valL = grayL * 255;
      img.data[idxL] = valL;
      img.data[idxL + 1] = valL;
      img.data[idxL + 2] = valL;
      img.data[idxL + 3] = 255;

      const idxR = (invRow * 128 + (col + 64)) * 4;
      const valR = grayR * 255;
      img.data[idxR] = valR;
      img.data[idxR + 1] = valR;
      img.data[idxR + 2] = valR;
      img.data[idxR + 3] = 255;
    }
    this.agentViewCtx.putImageData(img, 0, 0);
  }

  visualizeKernels(agent) {
    const wData = agent.actor.layers[0].getWeights()[0].dataSync();
    const numFilters = 8;
    const kSize = 8;
    const inChannels = 2;
    this.kernelCtx.clearRect(0, 0, 256, 32);

    let min = 9999;
    let max = -9999;
    for (let i = 0; i < wData.length; i++) {
      if (wData[i] < min) min = wData[i];
      if (wData[i] > max) max = wData[i];
    }
    const range = max - min || 1;

    for (let f = 0; f < numFilters; f++) {
      const imgData = this.kernelCtx.createImageData(kSize, kSize);
      for (let y = 0; y < kSize; y++) {
        for (let x = 0; x < kSize; x++) {
          const idx =
            y * (kSize * inChannels * numFilters) +
            x * (inChannels * numFilters) +
            0 * numFilters +
            f;
          const norm = (wData[idx] - min) / range;
          const pxIdx = (y * kSize + x) * 4;
          const c = Math.floor(norm * 255);
          imgData.data[pxIdx] = c;
          imgData.data[pxIdx + 1] = c;
          imgData.data[pxIdx + 2] = c;
          imgData.data[pxIdx + 3] = 255;
        }
      }
      const tempCvs = document.createElement("canvas");
      tempCvs.width = kSize;
      tempCvs.height = kSize;
      tempCvs.getContext("2d").putImageData(imgData, 0, 0);
      this.kernelCtx.drawImage(tempCvs, f * 32, 0, 30, 30);
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

    this.balls = [];
    this.bodyToBall = new Map();
    for (let i = 0; i < CONFIG.batchSize; i++) {
      const b = new Ball(i, scene, physics, assets, CONFIG.ballRadius);
      this.balls.push(b);
      this.bodyToBall.set(b.body, b);
    }

    this.episodeStats = { count: 0, baskets: 0, shots: 0 };
    this.accuracyHistory = [];
    this.isTrainingStep = false;
  }

  // Position all balls but don't launch (initial state / warm-up).
  spawnAll() {
    for (const b of this.balls) b.spawn();
    this.dashboard.setBatchProgress("Simulating...");
  }

  // Spawn, capture vision, predict, and launch a fresh batch.
  resetBatch(manualMesh) {
    this.spawnAll();

    const batch = this.vision.captureBatch(
      this.balls,
      this.hoopPos,
      this.court,
      manualMesh,
      this.trajectory.group
    );
    const actions = this.agent.predictBatch(batch, CONFIG.exploreNoise);
    const fs = this.vision.frameSize;
    for (let i = 0; i < this.balls.length; i++) {
      const b = this.balls[i];
      b.startPixels = batch.slice(i * fs, (i + 1) * fs);
      b.launch(actions[i], this.hoopPos);
    }

    this.dashboard.visualizeActivations(
      this.agent.getActivations(this.balls[0].startPixels)
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
      b.path.push(b.mesh.position.clone());

      const dist = b.body.position.distanceTo(this.hoopPosCannon);
      if (dist < b.minDist) b.minDist = dist;

      const isStopped = b.body.velocity.length() < 0.2;
      const isOOB =
        Math.abs(b.body.position.x) > 50 || Math.abs(b.body.position.z) > 30;
      const isFallen = b.body.position.y < -2;
      const isTimeout = b.path.length > 600;

      if (
        b.scored ||
        isFallen ||
        isOOB ||
        (isStopped && b.path.length > 10) ||
        isTimeout
      ) {
        b.active = false;
        b.mesh.visible = false;
      }
    }
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

    for (const b of this.balls) {
      const { reward, type } = this._reward(b);
      if (b.scored) this.episodeStats.baskets++;
      this.trajectory.add(b.path, type, reward);
      this.agent.store(b.startPixels, b.action, reward);
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
      episodes: this.episodeStats.count
    });
    this.dashboard.setBatchProgress("Training GPU...");

    try {
      const loss = await this.agent.train();
      if (loss != null) this.dashboard.pushLoss(loss);
    } catch (e) {
      console.error("Train Error", e);
    }

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
    this.vision = new VisionSystem(this.sceneMgr.renderer, scene);
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
      if (this.trainingMode && this.arena)
        this.arena.resetBatch(this.manual.mesh);
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

    this.arena.spawnAll();
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
