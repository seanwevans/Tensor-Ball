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
  // Shots the reported accuracy averages over. This was 1024 — exactly one
  // batch at the default batchSize — so the "rolling" accuracy was just the
  // last batch's rate, and a run's accuracy trace carried a full batch's worth
  // of binomial noise on every point. Over eight batches the same trace is
  // readable, which matters when the question being asked of it is whether a
  // slow climb has flattened out.
  accuracyWindow: 8192,
  // Closest a ball may spawn to the rim's axis, in feet.
  //
  // Spawns were drawn from the whole half court, rim included. Two things go
  // wrong in the column right under the hoop. dirToHoop is the difference of
  // two nearly equal positions, and three.js normalize() returns the zero
  // vector rather than NaN when that difference is ~0, so the forward
  // component of the launch silently vanishes and the ball goes straight up.
  // And a ball starting under the rim can only ever pass through it from
  // below, which is an illegalEntry: the largest negative reward in the table,
  // handed out for a state with no action that avoids it.
  //
  // Those samples are pure noise in the advantage — a big penalty uncorrelated
  // with anything the policy could have done differently. 2ft clears the rim
  // column (0.55) with room to spare, and spawns at 2ft still have a make
  // available: sweeping the action space per distance band, 100% of 2-4ft
  // spawns have a clean swish reachable.
  minSpawnDistance: 2.0,
  // The policy is Gaussian and learns its own spread, so there is no
  // exploration schedule to tune — exploreNoise / exploreNoiseMin /
  // exploreNoiseDecay are gone. What replaces them:
  //
  // A global annealed noise is the wrong shape for this problem. It is one
  // number for every state, so it cannot be small where the policy is good and
  // large where it is not, and being on a schedule it stops shrinking whether
  // or not the agent has earned it. The schedule this replaces has since been
  // retuned — 0.99/batch to a floor of 0.04, rather than 0.999 to 0.15 — which
  // fixes how long it takes to arrive but not its shape: it is still one number
  // for every state, and it still stops moving on a fixed date. And a floor is
  // still a hard ceiling, because it is added to whatever the policy outputs
  // and never goes away. Modelling the best action available to any policy
  // under a fixed jitter, on the current launch envelope, 0.15 caps accuracy
  // near 56% and 0.04 near 70% — better, but still a bound the agent cannot
  // argue with.
  //
  // So make the spread part of the policy. The actor emits a mean and a log
  // standard deviation per action dimension, actions are sampled from that
  // Gaussian, and the advantage-weighted objective becomes the likelihood of
  // the action taken rather than a regression onto it. Sigma then shrinks by
  // itself wherever the high-advantage actions cluster tightly, and stays wide
  // wherever they do not — per state, on the agent's own evidence.
  //
  // logStdInit is exp()'d, so 0.25 starts the policy sampling about as widely
  // as the old schedule's opening +/-0.2 uniform. logStdMin is the floor that
  // keeps a collapsed policy still exploring; logStdMax stops a policy that is
  // learning nothing from widening without bound.
  policy: {
    logStdInit: Math.log(0.25),
    logStdMin: Math.log(0.02),
    logStdMax: Math.log(0.8)
  },
  // Balls at the front of each batch launched on the policy's mean action
  // rather than a draw from its distribution, so the dashboard can report what
  // the policy actually does rather than what the policy plus its exploration
  // does.
  //
  // Every other accuracy number in this app is measured on the behaviour
  // policy: predictBatch samples each action from N(mean, sigma) and the shot
  // that gets graded is the sampled one. That conflates two things that move
  // independently — how good the policy is, and how widely it is currently
  // exploring. An agent improving while its spread holds it back looks exactly
  // like an agent that has stopped learning, which is the whole reason the
  // spread is now something the policy chooses rather than something a
  // schedule imposes.
  //
  // These balls are still stored and trained on: a greedy action with a good
  // reward is the best imitation target in the batch, not a sample to discard.
  // 128 of 1024 costs an eighth of the batch's exploration and gives the eval
  // rate a large enough sample to be readable batch to batch.
  evalBalls: 128,
  advantageTemp: 1.0,
  advantageClip: 20.0,
  // Passes the critic makes over the batch it just collected. The critic is
  // what turns a reward into an advantage, so an underfit critic hands the
  // actor something close to "reward minus batch mean" and the imitation
  // weights stop depending on the state at all.
  criticEpochs: 1,
  // Minibatch for the actor's weighted-regression pass. The batch is walked
  // once either way, so this trades the size of each Adam step against how many
  // of them a batch buys.
  actorMinibatch: 32,
  // Self-imitation replay.
  //
  // train() ends with `this.memory = null`, so every shot the agent has ever
  // taken is seen by exactly one gradient step and then thrown away. That is
  // very expensive for the samples that matter: made baskets are a low
  // single-digit percentage of a batch early on, they are the only samples
  // carrying any information about what a good shot looks like, and each one
  // gets a single weighted regression step before it is gone forever.
  //
  // Keep the best shots and replay them. The buffer is admitted to by
  // advantage, not by reward, because reward is dominated by shot distance —
  // a made three is worth more than a made layup regardless of which was the
  // better decision from where the ball was standing.
  //
  // Costs a Float32Array of capacity * visionW * visionH * 2 * 4B, which is
  // 37MB at these defaults — half a batch's state tensor.
  replay: {
    enabled: true,
    capacity: 512,
    // Best shots of each batch offered to the buffer.
    admitPerBatch: 64,
    // Shots drawn back out of it into each actor update.
    samplesPerBatch: 128
  },
  // The launch envelope: what the actor's tanh-bounded 3-vector means in
  // ft/s. action[0] and action[1] map linearly onto [fwdMin, fwdMax] and
  // [upMin, upMax], action[2] onto +/-side.
  //
  // These were inline constants working out to fwd 0-50, up 10-45 and side
  // +/-10 ft/s, and all three were far wider than the physics needs. That
  // costs accuracy directly, because the exploration noise is a fixed fraction
  // of the action range: at the exploration floor every shot carries
  // +/-0.075 of action, which against the old fwd range was +/-1.9ft/s of
  // launch speed — roughly a 10% speed error on a mid-range shot, when a make
  // wants single-digit percent.
  //
  // The side channel was the worst of the three. launch() already aims the
  // shot down dirToHoop, so action[2] is only ever a correction, yet it spanned
  // +/-10ft/s — over a ~1.4s flight the noise alone threw the ball more than a
  // foot sideways, against a rim that leaves ~0.35ft of room around the ball.
  //
  // Narrowing the ranges onto the band the physics actually uses multiplies
  // the effective resolution of the action against the same noise. Modelled
  // over the spawn distribution (the best action available to any policy, with
  // the noise added to it), the ceiling on accuracy goes 27% -> 56% at the
  // current 0.15 floor and 61% -> 85% at 0.05. Reach is unchanged: every spawn
  // that had a swish available under the old envelope still has one, checked
  // per distance band down to 2ft.
  //
  // Scalars rather than pairs so every bound is reachable from a
  // tools/hpsearch `--set launch.fwdMax=34` overlay.
  launch: {
    fwdMin: 3,
    fwdMax: 30,
    upMin: 13,
    upMax: 39,
    side: 3
  },
  ipd: 0.2067,
  visionFov: 60,
  groups: { court: 1, ball: 2 },
  rim: { x: 41.75, y: 10, z: 0 },
  // Spawn-distance curriculum.
  //
  // Spawns are drawn from the whole half court from the first batch, and the
  // far half of it is the worst possible place to start learning. Long shots
  // are both the hardest to hit and the hardest to *see*: the README's own
  // disparity arithmetic puts one pixel of stereo disparity at ~1.7px at 10ft
  // and ~0.86px at 20ft, so past mid-range the second eye stops carrying
  // usable depth — and conv1 (8x8, stride 4) throws away most of what is left.
  // Modelled over the action space, the best accuracy available to any policy
  // falls from ~51% at 0-10ft to ~25% at 30-50ft at the current noise floor.
  //
  // So the far court contributes samples that are near-unhittable, poorly
  // ranged, and a permanent drag on the average — while the near court, where
  // the signal is good, is only a fraction of each batch. Start close and open
  // the court up as the policy earns it.
  //
  // Expansion is one-way. A curriculum that also contracts makes the reported
  // accuracy non-comparable over time (the task gets easier exactly when the
  // agent gets worse); one-way means any accuracy number is measured on a task
  // at least as hard as every number before it.
  //
  // At maxRadius the disc covers the whole half court — the farthest corner is
  // 47.7ft from the rim — so the spawn distribution at full radius is exactly
  // the uniform-over-the-court one this replaces.
  curriculum: {
    enabled: true,
    startRadius: 12,
    maxRadius: 48,
    // Rolling accuracy that has to be cleared before the radius grows.
    expandAbove: 0.3,
    // Growth per batch while above the threshold: 12ft -> 48ft in ~47 batches.
    expandRate: 1.03,
    // Shots that must be in the accuracy window before it is trusted to gate
    // anything, so the radius cannot run away on one lucky early batch.
    minSamples: 512
  },
  // Which way the ball went through the hoop is the whole game, so the two
  // directions sit at opposite ends of the reward scale rather than a hair
  // apart. Dropping in from above is the only outcome worth learning. Sailing
  // up through the rim from underneath is the outcome most easily mistaken for
  // it: the ball passes dead through the middle, so its closest approach is ~0
  // and it usually clips the rim on the way, which under proximity shaping
  // alone made it score better than every honest miss in the batch — a shape
  // the policy can exploit by learning to fire straight up through the net.
  reward: {
    // Made basket, bonused by how far out the shot was launched from.
    score: 25.0,
    scoreDistanceScale: 20.0,
    // Flat penalty for entering the hoop from below, applied instead of the
    // rim/backboard bonuses. Deep enough to sit well below the worst honest
    // miss, whose proximity term bottoms out near -5.
    illegalEntry: -20.0,
    rim: 2.0,
    backboard: 0.5,
    missDistanceScale: 10.0
  },
  // Calling a sensor crossing "came up through the hoop" takes more than an
  // upward velocity: a ball rattling on the rim from above can clip the sensor
  // while bouncing back up. Require real ascent, and require the ball to be in
  // the column of the hoop rather than hanging off its edge.
  //
  // scoreRadius is how close to the rim's axis the ball's centre has to be, at
  // the moment it crosses the rim's plane going down, for the shot to have gone
  // through the hoop. It is the hole minus the ball: the rim is a torus of
  // radius 0.75 and tube 0.05, so the clear opening is 0.70, and a ball of
  // radius 0.40 only fits while its centre is within 0.30 of the axis.
  //
  // Nothing enforced this before. Scoring was a contact against the scoring
  // sensor, and that sensor is a Cylinder(0.5) against a Sphere(0.4), so cannon
  // reported a hit as soon as the ball's centre came within ~0.9ft of the axis
  // — more than a foot of slack around a hole the ball barely fits through.
  // Measured over twelve batches, 75% of the shots credited as baskets had
  // their centre further than 0.30ft from the axis when they were credited, and
  // the median was 0.63ft: balls bouncing off the outside of the rim, each one
  // collecting +25 and the largest positive advantage in the batch.
  hoopEntry: { minAscentSpeed: 1.0, columnRadius: 0.55, scoreRadius: 0.3 }
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
    // The same backboard, minus refraction, for the agents' capture pass.
    //
    // A material with transmission > 0 anywhere in the render list makes
    // WebGLRenderer run renderTransmissionPass() on every render(): the opaque
    // scene is drawn again into a 1024x1024 (multisampled, mipmapped) target so
    // the refraction shader has something to sample. That is a fixed ~1MP of
    // extra work per render, and vision capture issues two renders per ball —
    // 2048 of them per batch at batchSize 1024 — each for a 96x96 tile. The
    // backboard's refraction was therefore costing more than the entire rest of
    // training: measured under SwiftShader it was 160ms per view against 1.0ms
    // with this material swapped in, a 160x difference in capture time, for a
    // mean change of 0.002 in the grayscale the agent actually receives.
    //
    // The player's view still gets the refracting board — that is one render
    // per frame, not two per ball.
    this.glassCapture = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
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
      miss: trail(0xff0000, 0.15),
      // Wrong-way hoop entries get their own color: their trails run through
      // the rim exactly like a made basket's, so without a distinct magenta
      // there is no way to see the policy chasing the penalty from the court.
      illegal: trail(0xff00ff, 0.35)
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
      miss: endpoint(0xff0000, 0.4),
      illegal: endpoint(0xff00ff, 0.8)
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

// A fixed-capacity store of the highest-advantage shots the agent has taken,
// for replaying into the actor update (see CONFIG.replay).
//
// States are kept in one flat Float32Array rather than as per-sample arrays,
// matching how the batch arrives from VisionSystem: at 96px stereo a single
// frame is 18432 floats, so per-sample slices would mean an allocation and a
// copy per admitted shot, every batch, forever.
//
// Admission is "beat the weakest thing in here", which keeps the buffer at the
// running top-`capacity` by advantage without sorting it.
class ReplayBuffer {
  constructor(capacity, frameSize) {
    this.capacity = capacity;
    this.frameSize = frameSize;
    this.states = new Float32Array(capacity * frameSize);
    this.actions = new Float32Array(capacity * 3);
    this.rewards = new Float32Array(capacity);
    this.priority = new Float32Array(capacity);
    this.size = 0;
  }

  // Index of the weakest entry, or -1 while there is still room.
  _weakest() {
    if (this.size < this.capacity) return -1;
    let worst = 0;
    for (let i = 1; i < this.size; i++)
      if (this.priority[i] < this.priority[worst]) worst = i;
    return worst;
  }

  // states/actions are the batch's flat arrays; i indexes into them.
  offer(states, actions, reward, priority, i) {
    let slot;
    if (this.size < this.capacity) slot = this.size++;
    else {
      slot = this._weakest();
      if (priority <= this.priority[slot]) return false;
    }
    const f = this.frameSize;
    this.states.set(states.subarray(i * f, (i + 1) * f), slot * f);
    this.actions[slot * 3 + 0] = actions[i][0];
    this.actions[slot * 3 + 1] = actions[i][1];
    this.actions[slot * 3 + 2] = actions[i][2];
    this.rewards[slot] = reward;
    this.priority[slot] = priority;
    return true;
  }

  // `count` distinct entries at random, as flat arrays ready for tf.tensor.
  // Uniform rather than priority-weighted: the buffer is already the top of the
  // distribution, and weighting the draw too would train the actor almost
  // entirely on whichever handful of shots scored best.
  sample(count) {
    const n = Math.min(count, this.size);
    if (n === 0) return null;
    const idx = new Int32Array(this.size);
    for (let i = 0; i < this.size; i++) idx[i] = i;
    for (let i = 0; i < n; i++) {
      const j = i + Math.floor(Math.random() * (this.size - i));
      const t = idx[i];
      idx[i] = idx[j];
      idx[j] = t;
    }
    const f = this.frameSize;
    const states = new Float32Array(n * f);
    const actions = new Float32Array(n * 3);
    const rewards = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      const src = idx[k];
      states.set(this.states.subarray(src * f, (src + 1) * f), k * f);
      actions[k * 3 + 0] = this.actions[src * 3 + 0];
      actions[k * 3 + 1] = this.actions[src * 3 + 1];
      actions[k * 3 + 2] = this.actions[src * 3 + 2];
      rewards[k] = this.rewards[src];
    }
    return { states, actions, rewards, count: n };
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
    this.replay =
      CONFIG.replay.enabled && CONFIG.replay.capacity > 0
        ? new ReplayBuffer(CONFIG.replay.capacity, this.frameSize)
        : null;
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
    // Six linear outputs, not three tanh ones: the first three are the mean of
    // the action distribution (squashed by tanh in _headSplit, not here, so the
    // second three stay unsquashed) and the last three are its log standard
    // deviation. Keeping it a single sequential dense layer rather than two
    // heads keeps the actor serializable through EXPORT POLICY.
    m.add(tf.layers.dense({ units: 6 }));
    m.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: "meanSquaredError"
    });
    // Start the spread at CONFIG.policy.logStdInit. The layer's bias is zeros
    // by default, which would open at std 1 — a policy sampling almost
    // uniformly across the whole action range, for as long as it took the
    // gradient to pull the bias back down.
    // setWeights assigns into the layer's existing variables, so everything
    // created here is disposable once it returns.
    tf.tidy(() => {
      const head = m.layers[m.layers.length - 1];
      const [kernel, bias] = head.getWeights();
      const b = bias.arraySync().slice();
      for (let i = 3; i < 6; i++) b[i] = CONFIG.policy.logStdInit;
      head.setWeights([kernel, tf.tensor1d(b)]);
    });

    this.actorOptimizer = tf.train.adam(this.learningRate);
    return m;
  }

  // Split the actor's six raw outputs into (mean, logStd, std).
  //
  // tanh on the mean keeps it inside the action range the launcher accepts;
  // logStd is clipped rather than squashed, so its gradient is exactly zero
  // outside the range instead of merely small — which is what stops a policy
  // that is learning nothing from drifting the spread out forever.
  _headSplit(raw) {
    const mean = tf.tanh(raw.slice([0, 0], [-1, 3]));
    const logStd = raw
      .slice([0, 3], [-1, 3])
      .clipByValue(CONFIG.policy.logStdMin, CONFIG.policy.logStdMax);
    return { mean, logStd, std: tf.exp(logStd) };
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

  // One standard normal, Box-Muller. Built on Math.random so a seeded run
  // (tools/hpsearch replaces Math.random) still reproduces exactly.
  static _gauss() {
    // u must be non-zero for the log; Math.random() can return exactly 0.
    const u = 1 - Math.random();
    const v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // pixelDataBatch: Float32Array of size count * frameSize (frameSize = W*H*2).
  //
  // Actions are sampled from the policy's own Gaussian rather than perturbed by
  // an external schedule, so there is no noise scale to pass any more — the
  // spread is whatever the policy asked for at that state.
  //
  // The first greedyCount actions are the distribution's mean instead of a draw
  // from it. Those balls are the batch's eval sample (CONFIG.evalBalls): they
  // measure the policy itself, while the rest carry the exploration that
  // generates the batch's variety. Taking them off the front rather than at
  // random keeps the split stable across batches, so the eval rate is a fixed
  // sample of the state distribution rather than a fresh one every time.
  //
  // Records the batch's mean standard deviation on the agent, so the spread the
  // policy has settled on can be read off without a second forward pass.
  predictBatch(pixelDataBatch, greedyCount = 0) {
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
      const { mean, std } = this._headSplit(this.actor.predict(stateTensor));
      const meanData = mean.dataSync();
      const stdData = std.dataSync();

      let stdSum = 0;
      for (let i = 0; i < stdData.length; i++) stdSum += stdData[i];
      this.lastMeanStd = stdData.length ? stdSum / stdData.length : 0;

      // Clamp to [-1, 1]: the mean is tanh-bounded but a sample is not, and
      // these actions are later used as the targets whose likelihood the actor
      // is trained on, so out-of-range values would be unreachable goals.
      const clamp = (v) => Math.max(-1, Math.min(1, v));
      const actions = [];
      for (let i = 0; i < count; i++) {
        const a = [];
        for (let k = 0; k < 3; k++) {
          const j = i * 3 + k;
          // The draw happens either way: scaling it to zero rather than
          // skipping the call keeps the RNG stream identical whatever
          // greedyCount is, so a seeded run (tools/hpsearch) stays comparable
          // to one with a different eval split.
          const g = CNNAgent._gauss();
          const spread = i < greedyCount ? 0 : stdData[j];
          a.push(clamp(meanData[j] + spread * g));
        }
        actions.push(a);
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
      // The head is six wide now, but the activation panel draws the three
      // action channels — so hand it the mean, post-tanh, which is what the
      // three-output head used to return.
      const { mean } = this._headSplit(this.actor.layers[4].apply(dense64));
      return { dense: dense64.dataSync(), output: mean.dataSync() };
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
      epochs: CONFIG.criticEpochs,
      verbose: 0
    });
    // Report the last epoch's loss, so the number on the dashboard is the
    // critic's error after the update however many passes it took.
    const losses = criticHistory.history.loss;
    const loss = losses && losses.length ? losses[losses.length - 1] : 0;

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
    for (let i = 0; i < batchSize; i++)
      weightData[i] = this._advantageWeight(
        advantageData[i],
        advMean,
        advStd
      );
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

    // Replay the best shots the agent has ever taken, then offer this batch's
    // best to the buffer. Fresh first so the buffer's contribution is measured
    // against a head that has already seen the current batch.
    this._replayPass(advMean, advStd);
    this._admitToReplay(states, actions, rewards, advantageData);

    stateTensor.dispose();
    actionTensor.dispose();
    rewardTensor.dispose();
    values.dispose();
    advantages.dispose();
    weightTensor.dispose();
    this.memory = null;
    return loss;
  }

  // Weight an advantage the same way the fresh batch's were, so replayed
  // samples compete on the current batch's scale rather than on their own.
  _advantageWeight(advantage, advMean, advStd) {
    const z = (advantage - advMean) / advStd;
    return Math.min(CONFIG.advantageClip, Math.exp(z / CONFIG.advantageTemp));
  }

  // A second pass of the actor's objective over a draw from the replay
  // buffer. It shares _fitActorWeighted, so replayed samples are fitted by
  // the same advantage-weighted likelihood as the fresh batch — including
  // the gradient on sigma, which they should contribute to as much as any
  // other sample the policy is being judged on.
  //
  // The advantages are recomputed against the *current* critic rather than
  // reused from when the shot was taken: the critic is what a sample's
  // advantage is relative to, and it has moved since. A shot that beat a naive
  // early critic by a mile may be unremarkable against a better one, and should
  // stop pulling as hard when that happens.
  _replayPass(advMean, advStd) {
    if (!this.replay) return;
    const draw = this.replay.sample(CONFIG.replay.samplesPerBatch);
    if (!draw) return;

    const states = tf.tensor(draw.states, [
      draw.count,
      this.visionH,
      this.visionW,
      this.channels
    ]);
    const actions = tf.tensor2d(draw.actions, [draw.count, 3]);
    const values = this.critic.predict(states);
    const valueData = values.dataSync();

    const weightData = new Float32Array(draw.count);
    for (let i = 0; i < draw.count; i++)
      weightData[i] = this._advantageWeight(
        draw.rewards[i] - valueData[i],
        advMean,
        advStd
      );
    const weights = tf.tensor1d(weightData);

    this._fitActorWeighted(states, actions, weights);

    states.dispose();
    actions.dispose();
    values.dispose();
    weights.dispose();
  }

  // Offer this batch's highest-advantage shots to the buffer.
  //
  // Admission is by advantage rather than by reward: reward is dominated by
  // shot distance, so ranking on it would fill the buffer with long makes and
  // starve it of the close-range shots that are the only thing a policy this
  // early can reliably repeat.
  _admitToReplay(states, actions, rewards, advantageData) {
    if (!this.replay) return;
    const n = rewards.length;
    const ranked = new Array(n);
    for (let i = 0; i < n; i++) ranked[i] = i;
    ranked.sort((a, b) => advantageData[b] - advantageData[a]);
    const take = Math.min(CONFIG.replay.admitPerBatch, n);
    for (let k = 0; k < take; k++) {
      const i = ranked[k];
      this.replay.offer(states, actions, rewards[i], advantageData[i], i);
    }
  }

  // One epoch of minibatch Adam over a per-sample advantage-weighted negative
  // log-likelihood. model.fit() cannot do this: passing sampleWeight throws
  // "Support sampleWeight is not implemented yet" in tfjs 4.x, so the update is
  // driven explicitly. Minibatch size matches fit()'s default of 32 so the
  // number of optimizer steps per batch is unchanged (CONFIG.actorMinibatch
  // retunes it). Only trainableWeights are passed to minimize(), which already
  // excludes the two frozen conv layers.
  //
  // The objective was a weighted squared error onto the action taken, which is
  // the same thing as this likelihood at a fixed spread — so with sigma now
  // part of the policy, keeping MSE would have trained the mean and left the
  // spread with no gradient at all. Under the likelihood, a dimension whose
  // high-advantage actions cluster tightly is fit better by a narrow Gaussian
  // and sigma is pulled down; a dimension whose good actions are all over the
  // place keeps a wide one. The log(sigma) term is what stops the trivial
  // solution of shrinking sigma to nothing everywhere.
  _fitActorWeighted(states, actions, weights, minibatch = CONFIG.actorMinibatch) {
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
            const raw = this.actor.apply(xb, { training: true });
            const { mean, logStd, std } = this._headSplit(raw);
            // -log N(a | mean, std), dropping the constant 0.5*log(2*pi) per
            // dimension: it shifts every sample's loss by the same amount and
            // so contributes nothing to the gradient.
            const z = yb.sub(mean).div(std);
            const perSample = z.square().mul(0.5).add(logStd).sum(1);
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
    // Both boards, because the transmission pass is triggered by anything
    // transmissive in the render list — swapping only the near one would leave
    // the cost in place.
    this.boardMeshes = [];

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

    // The backboard is a solid slab rather than the 0.1ft sheet of glass it
    // used to be, because at 0.1ft the balls went straight through it.
    //
    // cannon steps at a fixed 1/60s and actions are clamped to [-1, 1], so a
    // ball leaves the launcher at up to the corner of CONFIG.launch's envelope
    // — 49ft/s, or 0.82ft of travel per step. (This was 51ft/s before the
    // envelope was narrowed, so the margin below only got wider.) A
    // sphere is only stopped if some step samples it before its center passes
    // the box's midplane; past that, the narrowphase finds the *back* face
    // closest and resolves the overlap out the back. That budget is
    // ballRadius + halfDepth, i.e. 0.45ft for real glass, so a hard shot can
    // clear the whole board between two samples: swept over every reachable
    // impact speed, phase and spot on the face, 14% of head-on shots passed
    // through. At halfDepth 0.6 the budget is 1.0ft against 0.85ft of travel
    // and nothing gets through.
    //
    // The mesh is built from the same numbers as the body so the board that is
    // drawn is exactly the board that collides — thickening only the physics
    // box would trade shots through the glass for shots bouncing off thin air
    // behind it. Only the back of the board moves: the court-facing face stays
    // where it was, so bank shots play exactly as before.
    const boardHalfDepth = 0.6;
    const boardFaceX = boardX - sign * 0.05;
    const boardCenterX = boardFaceX + sign * boardHalfDepth;

    const board = new THREE.Mesh(
      new THREE.BoxGeometry(boardHalfDepth * 2, 3.5, 6),
      this.assets.glass
    );
    board.position.set(boardCenterX, 10.75, 0);
    group.add(board);
    this.boardMeshes.push(board);

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
      boardCenterX,
      10.75,
      0,
      new CANNON.Box(new CANNON.Vec3(boardHalfDepth, 1.75, 3))
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

  // Switch the court between the way the player sees it and the way the agents
  // are shown it during vision capture: a flat white rim so the target reads
  // clearly at 96px, and a backboard that doesn't drag a full-screen refraction
  // pass through every one of the batch's renders (see Assets.glassCapture).
  setCaptureMode(on) {
    if (this.rimMesh)
      this.rimMesh.material = on ? this.assets.highContrast : this.assets.rim;
    for (const board of this.boardMeshes)
      board.material = on ? this.assets.glassCapture : this.assets.glass;
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

// Whether a touch of the scoring sensor is the ball coming up through the rim
// from underneath. It takes more than an upward velocity: a ball rattling on
// the rim from above can clip the sensor while bouncing back up, so require
// real ascent and require the ball to be in the column of the hoop rather than
// hanging off its edge.
//
// The other direction is no longer decided here. A sensor touch is a poor proxy
// for a basket — see CONFIG.hoopEntry.scoreRadius — so baskets are detected
// geometrically in trackHoopPass instead, and the sensor now only answers this
// one question.
//
// Shared by the batch balls and the manual ball so both agree on what counts.
function isEntryFromBelow(body, court) {
  const vy = body.velocity.y;
  if (vy <= CONFIG.hoopEntry.minAscentSpeed) return false;
  const dx = body.position.x - court.rimPositionCannon.x;
  const dz = body.position.z - court.rimPositionCannon.z;
  const r = CONFIG.hoopEntry.columnRadius;
  return dx * dx + dz * dz <= r * r;
}

// Did the ball just fall through the hoop? Called once per frame per live ball,
// for anything carrying the {body, scored, enteredFromBelow, prev*} shape —
// the batch balls and the manual ball both do.
//
// This replaces "the ball touched the scoring sensor going down". That test was
// wrong in both directions of precision: the sensor is a whole foot wider than
// the rim's opening, and cannon reports the contact once, at whatever point on
// the way in the shapes first overlapped, which for a flat shot is not where
// the ball crossed the rim.
//
// So find the crossing itself. The rim is a horizontal plane at rim.y; if the
// ball was above it last frame and is below it now, solve for where it cut the
// plane and ask whether that point is inside the hole. Interpolating rather
// than sampling makes the answer independent of how far the ball travels in a
// step, which at up to 0.8ft per step is otherwise most of the rim's diameter.
//
// A ball that has already come up through the hoop cannot score on the way back
// down, matching the real rule: entering from below kills the ball. Without it
// a shot fired straight up through the net collects the basket reward on its
// own descent.
function trackHoopPass(o, court) {
  const p = o.body.position;
  const px = o.prevX;
  const py = o.prevY;
  const pz = o.prevZ;
  o.prevX = p.x;
  o.prevY = p.y;
  o.prevZ = p.z;

  if (py === null || o.scored || o.enteredFromBelow) return;
  const rim = court.rimPositionCannon;
  // Descending crossing of the rim's plane, and only that: a ball still on its
  // way up, or one that never reaches the rim's height, is not a candidate.
  if (!(py >= rim.y && p.y < rim.y)) return;

  const t = (py - rim.y) / (py - p.y);
  const dx = px + (p.x - px) * t - rim.x;
  const dz = pz + (p.z - pz) * t - rim.z;
  const s = CONFIG.hoopEntry.scoreRadius;
  if (dx * dx + dz * dz <= s * s) o.scored = true;
}

class Ball {
  constructor(id, field, physics, radius) {
    this.id = id;
    this.field = field;
    // Balls at the front of the batch are the eval sample: launched greedily
    // and graded separately (see CONFIG.evalBalls). Fixed by index rather than
    // re-drawn each batch, and set here rather than in _resetState because it
    // is a property of the slot, not of the shot.
    this.isEval = id < CONFIG.evalBalls;
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
    this.enteredFromBelow = false;
    this.hitBackboard = false;
    this.hitRim = false;
    // Last frame's position, for the rim-plane crossing test in trackHoopPass.
    // Null rather than the spawn point so the first frame of a shot cannot be
    // read as a crossing from wherever the previous shot ended.
    this.prevX = null;
    this.prevY = null;
    this.prevZ = null;
  }

  // Position at a random launch spot and mark active/visible.
  //
  // Sampled from the annulus between CONFIG.minSpawnDistance and maxRadius:
  // the curriculum (CONFIG.curriculum) sets the outer bound and the rim's
  // column sets the inner one.
  //
  // The two bounds exist for opposite reasons and both have to hold. Inside
  // minSpawnDistance, dirToHoop degenerates and every upward shot is an illegal
  // entry, so those spawns are unlearnable. Outside maxRadius the shot is
  // simply harder than the policy has earned yet.
  //
  // r = sqrt(u * (R^2 - m^2) + m^2) is the area-uniform draw over that annulus,
  // so the near edge is not crowded — which would quietly make the
  // curriculum's early batches all point-blank — and it reduces to the plain
  // disc when m is 0. Rejecting back into the court bounds then leaves the
  // draw uniform over the intersection, so once the radius is large enough to
  // contain the court this is the uniform-over-the-court draw it replaces,
  // minus the excluded column.
  //
  // The loop is bounded and falls back to a point on the court-facing axis, so
  // a radius that somehow admits no legal spot cannot hang the batch.
  spawn(maxRadius = CONFIG.curriculum.maxRadius) {
    const rim = CONFIG.rim;
    const min = CONFIG.minSpawnDistance;
    const outer = Math.max(min, maxRadius);
    let x = 0;
    let z = 0;
    for (let attempt = 0; attempt < 24; attempt++) {
      const a = Math.random() * Math.PI * 2;
      const r = Math.sqrt(
        Math.random() * (outer * outer - min * min) + min * min
      );
      x = rim.x + Math.cos(a) * r;
      z = rim.z + Math.sin(a) * r;
      if (x >= 0 && x <= 42 && Math.abs(z) <= 23) break;
      if (attempt === 23) {
        // min wins over maxRadius here: the inner bound is the one that exists
        // to keep the ball out of an unlearnable state, so a misconfigured
        // outer bound must not be able to override it.
        const d = Math.min(Math.max(min, Math.min(outer, 41.75)), 41.75);
        x = Math.max(0, rim.x - d);
        z = rim.z;
      }
    }
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

    // The action is a launch *velocity* in CONFIG.launch's envelope, converted
    // to an impulse here. Writing the envelope in ft/s rather than in impulse
    // units keeps it in the same units as the rest of the world (the court is
    // in feet, gravity is 32.2ft/s^2), so the numbers can be checked against
    // the physics instead of against the ball's mass.
    const L = CONFIG.launch;
    const lerp = (lo, hi, a) => lo + ((a + 1) / 2) * (hi - lo);
    const vFwd = lerp(L.fwdMin, L.fwdMax, action[0]);
    const vUp = lerp(L.upMin, L.upMax, action[1]);
    const vSide = action[2] * L.side;

    const m = this.body.mass;
    const impulse = new THREE.Vector3()
      .add(dirToHoop.multiplyScalar(vFwd * m))
      .add(new THREE.Vector3(0, 1, 0).multiplyScalar(vUp * m))
      .add(dirSide.multiplyScalar(vSide * m));

    this.body.wakeUp();
    this.body.applyImpulse(
      new CANNON.Vec3(impulse.x, impulse.y, impulse.z),
      new CANNON.Vec3(0, 0, 0)
    );
  }

  onContact(other, court) {
    if (other === court.scoringSensor) {
      // Baskets are decided in trackHoopPass; the sensor's only remaining job
      // is catching the ball on its way up through the rim.
      if (isEntryFromBelow(this.body, court)) this.enteredFromBelow = true;
    } else if (other === court.backboardBody) this.hitBackboard = true;
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
    court.setCaptureMode(true);

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
    court.setCaptureMode(false);
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
      // Brightness tracks the distance bonus. Reward is score * (1 + d/scale),
      // so reward / score - 1 is the bonus in units of the base score, and the
      // ramp stays correct if the reward scale is retuned.
      const t = Math.max(
        0,
        Math.min(1, (reward / CONFIG.reward.score - 1) / 2)
      );
      opacity = 0.3 + t * 0.7;
    } else if (type === "illegal") {
      opacity = 0.35;
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
    this.uiIllegal = document.getElementById("illegal-entries");
    this.uiAcc = document.getElementById("accuracy");
    this.uiEvalAcc = document.getElementById("eval-accuracy");
    this.uiRadius = document.getElementById("spawn-radius");
    this.uiStd = document.getElementById("policy-std");
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

  setStats({
    accuracy,
    evalAccuracy,
    spawnRadius,
    policyStd,
    baskets,
    episodes,
    illegalEntries
  }) {
    this.uiAcc.innerText = Math.round(accuracy * 100) + "%";
    // Null until the first eval ball has been graded, and when a config turns
    // the eval split off entirely.
    if (this.uiEvalAcc)
      this.uiEvalAcc.innerText =
        evalAccuracy == null ? "--" : Math.round(evalAccuracy * 100) + "%";
    // Both accuracies above are only readable next to the radius they were
    // measured at, since the curriculum is what decides how hard the batch was.
    if (this.uiRadius && spawnRadius != null)
      this.uiRadius.innerText = spawnRadius.toFixed(1) + " ft";
    // The spread the policy chose, which used to be a number on a schedule and
    // is now the most direct readout of whether the agent thinks it knows what
    // it is doing.
    if (this.uiStd && policyStd != null)
      this.uiStd.innerText = policyStd.toFixed(3);
    this.uiBaskets.innerText = baskets;
    this.uiEp.innerText = episodes;
    if (this.uiIllegal) this.uiIllegal.innerText = illegalEntries;
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

// Fraction of the last `window` outcomes that were hits, kept as a ring buffer
// with a running sum.
//
// The accuracy window used to be a plain array pushed and shift()ed. shift() is
// O(n), and it ran once per ball per batch — fine while the window was one
// batch long, quadratic in the window otherwise, and the window wants to be
// several batches long to be readable at all.
class RollingRate {
  constructor(window) {
    this.buf = new Uint8Array(Math.max(1, window));
    this.i = 0;
    this.n = 0;
    this.sum = 0;
  }

  push(hit) {
    const v = hit ? 1 : 0;
    if (this.n === this.buf.length) this.sum -= this.buf[this.i];
    else this.n++;
    this.buf[this.i] = v;
    this.sum += v;
    this.i = (this.i + 1) % this.buf.length;
  }

  get value() {
    return this.n ? this.sum / this.n : 0;
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

    // Balls launched greedily each batch. Clamped to the batch so a config
    // with evalBalls >= batchSize degrades to "the whole batch is eval" rather
    // than silently leaving no ball exploring.
    this.evalBalls = Math.min(CONFIG.evalBalls, CONFIG.batchSize);

    // Current curriculum radius. Starts small and only ever grows — see
    // CONFIG.curriculum and _advanceCurriculum below.
    this.spawnRadius = CONFIG.curriculum.enabled
      ? CONFIG.curriculum.startRadius
      : CONFIG.curriculum.maxRadius;

    this.episodeStats = { count: 0, baskets: 0, shots: 0, illegal: 0 };
    this.accuracyHistory = new RollingRate(CONFIG.accuracyWindow);
    // Graded separately from accuracyHistory: only the shots the exploration
    // noise never touched. Sized by evalBalls/batchSize so it spans the same
    // number of *batches* as the behaviour window rather than the same number
    // of shots — otherwise the eval rate would average over eight times as much
    // history and visibly lag.
    this.evalHistory = new RollingRate(
      Math.max(
        1,
        Math.round((CONFIG.accuracyWindow * this.evalBalls) / CONFIG.batchSize)
      )
    );
    this.isTrainingStep = false;
    // Not a schedule any more — the mean spread the policy chose for the last
    // batch, read back after predictBatch. Kept under this name because it is
    // what the dashboard and the search harness report as the batch's noise,
    // and it means the same thing: how far the shots taken were from the
    // policy's mean action.
    this.exploreNoise = Math.exp(CONFIG.policy.logStdInit);
  }

  // Position all balls but don't launch (initial state / warm-up).
  spawnAll() {
    this.field.setVisible(true);
    for (const b of this.balls) b.spawn(this.spawnRadius);
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
    const actions = this.agent.predictBatch(batch, this.evalBalls);
    this.exploreNoise = this.agent.lastMeanStd;
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
      // Before the retire check below, so a made shot ends on the frame it
      // drops through rather than one frame later.
      trackHoopPass(b, this.court);
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
    const R = CONFIG.reward;
    if (b.scored) {
      const reward = R.score * (1.0 + shotDistance / R.scoreDistanceScale);
      return { reward, type: "score" };
    }
    const proximity = -b.minDist / R.missDistanceScale;
    // Through the hoop the wrong way. Its proximity term is the best in the
    // batch — the ball passed through the middle of the rim — so the penalty
    // has to be flat and large enough to swamp it, and it replaces the rim
    // bonus the same shot would otherwise collect on its way up.
    if (b.enteredFromBelow)
      return { reward: proximity + R.illegalEntry, type: "illegal" };
    if (b.hitRim) return { reward: proximity + R.rim, type: "rim" };
    if (b.hitBackboard) return { reward: proximity + R.backboard, type: "rim" };
    return { reward: proximity, type: "miss" };
  }

  // Open the court up once the policy is hitting at the current radius.
  //
  // Gated on the rolling accuracy rather than on a batch count, so a run that
  // is not learning stays on the near court instead of being marched out to
  // half court on a schedule. Growth is one-way: see CONFIG.curriculum.
  _advanceCurriculum(rollingAcc) {
    const c = CONFIG.curriculum;
    if (!c.enabled) return;
    if (this.spawnRadius >= c.maxRadius) return;
    if (this.accuracyHistory.n < c.minSamples) return;
    if (rollingAcc < c.expandAbove) return;
    this.spawnRadius = Math.min(c.maxRadius, this.spawnRadius * c.expandRate);
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
      if (b.enteredFromBelow) this.episodeStats.illegal++;
      this.trajectory.add(b.path, type, reward);
      actions.push(b.action);
      rewards.push(reward);
      this.episodeStats.shots++;
      this.episodeStats.count++;
      this.accuracyHistory.push(b.scored);
      if (b.isEval) this.evalHistory.push(b.scored);
    }
    // The states are already contiguous and in ball order in batchPixels, so
    // the agent takes the whole array by reference instead of rebuilding it.
    this.agent.store(this.batchPixels, actions, rewards);

    // Bound rather than inlined: _advanceCurriculum below gates on this same
    // number, and the two must not be able to drift apart.
    const rollingAcc = this.accuracyHistory.value;
    this.dashboard.setStats({
      accuracy: rollingAcc,
      // Null until the first eval ball has been graded.
      evalAccuracy: this.evalHistory.n ? this.evalHistory.value : null,
      spawnRadius: this.spawnRadius,
      policyStd: this.exploreNoise,
      baskets: this.episodeStats.baskets,
      episodes: this.episodeStats.count,
      illegalEntries: this.episodeStats.illegal
    });
    this.dashboard.setBatchProgress("Training GPU...");

    try {
      const loss = await this.agent.train();
      if (loss != null) this.dashboard.pushLoss(loss);
    } catch (e) {
      console.error("Train Error", e);
    }

    this._advanceCurriculum(rollingAcc);

    // No annealing step: the spread is a policy output now, so it moves when
    // the actor update moves it and resetBatch reads back what it became.

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
    this.enteredFromBelow = false;
    this.prevX = null;
    this.prevY = null;
    this.prevZ = null;
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
    if (other !== court.scoringSensor) return;
    if (isEntryFromBelow(this.body, court)) this.enteredFromBelow = true;
  }

  update(court) {
    this.mesh.position.copy(this.body.position);
    this.mesh.quaternion.copy(this.body.quaternion);
    if (this.inProgress) {
      // Only while a shot is live. Right-dragging the ball can carry it down
      // through the rim's plane, which is a reposition, not a basket.
      trackHoopPass(this, court);
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

    this.manual.update(this.court);
    this.sceneMgr.render();
  }
}

new App().start();
