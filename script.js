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
  // These balls still fit the critic and are still admitted to replay, but
  // they are held out of the actor's own update — under the likelihood
  // objective an action drawn at the mean cannot move the mean and can only
  // shrink the spread, which is a bias rather than a datum. See train().
  //
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
  // An advantage is a reading against a critic, though, and the critic moves.
  // Entries are therefore re-priced against the current one as they are drawn
  // (see ReplayBuffer.reprice) — without that the admission bar ratchets up
  // until nothing can clear it and the buffer freezes on whatever it happened
  // to be holding.
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
    side: 3,
    // Peak spin the shooter can put on the ball, in rad/s, about the axis
    // across the shot. +1 on the channel is backspin, -1 is topspin, and the
    // sign is chosen so the natural basketball shot is the positive one.
    //
    // 20rad/s is a bit over three revolutions a second, which is about what a
    // real jump shot carries, and it puts 8ft/s of surface speed at the
    // contact point against a rim the ball arrives at around 20ft/s.
    //
    // Worth knowing what this can and cannot do here: cannon has no
    // aerodynamics, so spin does not bend the flight — there is no Magnus
    // force — and a shot that touches nothing on the way in is unaffected by
    // it. It acts only through friction where the ball meets the rim, the
    // backboard or the floor.
    spin: 20
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
  // A simple air model: quadratic drag, and Magnus lift so that spin bends the
  // flight instead of only mattering when the ball touches something.
  //
  // Both are written as *accelerations*, and the force applied per step is
  // whatever produces them. That is deliberate. This world's ball has mass 2,
  // and with gravity at 32.2ft/s^2 the consistent unit is slugs — so it weighs
  // about 64lb, roughly 47 times a real basketball. Deriving forces from air
  // density and frontal area against that mass would give accelerations 47
  // times too small. Stating the accelerations directly sidesteps the ball's
  // mass entirely, and lets both constants be calibrated from a real
  // basketball's behaviour:
  //
  //  - drag is a = -k|v|v with k = 0.0066/ft, which puts terminal velocity at
  //    sqrt(g/k) = 70ft/s. A real basketball's is about 66ft/s.
  //  - magnus is a = C(w x v) with C dimensionless, calibrated so a shot
  //    carrying the full 20rad/s of backspin at 25ft/s gets about 2.2ft/s^2 of
  //    lift — near 7% of gravity, which over a second and a half of flight is
  //    a couple of feet of extra carry. That is the size of effect backspin
  //    actually has, and it is what makes the spin channel worth having.
  //
  // This replaces the ball's linearDamping, which was a velocity-proportional
  // stand-in for the same thing; leaving both on would count drag twice.
  //
  // One air for every shot, by default: wind and jitter are 0, so a given
  // action from a given spot always flies the same way.
  //
  // The alternative — rolling fresh air per shot, which these knobs turn on —
  // is more like a real gym, but it costs more than it looks. The agent sees
  // the court through two cameras; it cannot see the wind. Air drawn per shot
  // is therefore unobservable from the state, so it is not something the
  // policy can learn to compensate for, only something it must average over.
  // It lands entirely in the reward as noise the critic cannot explain, which
  // inflates the critic's loss and dilutes exactly the advantage signal the
  // actor is weighted by. A batch of 1024 balls already supplies plenty of
  // variety through spawn positions and the policy's own sampling, so the
  // randomness has somewhere better to come from.
  //
  // Left parameterised rather than deleted because "is a stochastic
  // environment worth it here" is a real question and tools/hpsearch can now
  // answer it: wind is a peak horizontal breeze in ft/s, entering as relative
  // airspeed so it feeds drag and Magnus the way moving air would rather than
  // as a bolted-on sideways nudge; jitter.drag stands in for inflation and
  // surface wear; jitter.magnus for where the seams happen to sit relative to
  // the spin axis, which genuinely swings how much lift a given spin makes.
  air: {
    enabled: true,
    drag: 0.0066,
    magnus: 0.0044,
    wind: 0,
    jitter: { drag: 0, magnus: 0 }
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
    // Made basket. Flat: every make is worth the same wherever it was taken
    // from.
    score: 25.0,
    // Extra reward per foot of shot distance, on top of it. Zero, and the
    // reason is that the agent never chooses where it shoots from.
    //
    // A three being worth more than a layup is a statement about shot
    // selection, and this agent has none: the spawn is drawn for it, it takes
    // the shot it is given, and no action it can pick changes how far out it
    // is standing. So a distance bonus cannot change a single decision the
    // policy makes. What it does instead is put the shot's *distance* into the
    // reward, where two things downstream read it as if it were the shot's
    // quality.
    //
    // The critic has to fit it. Value is what the reward will be, so a term
    // that swings the reward of a make from 27.5 to 85 across the disc is
    // variance the value head has to spend capacity explaining before it can
    // say anything about whether a state is a good one.
    //
    // And the actor is weighted by it. _advantageWeight is exp of the
    // advantage in units of the batch's own spread, and the advantage of a
    // make is dominated by the reward of the make: at 1.25/ft two makes 35ft
    // apart differ by 43.75 of reward against a batch advantage spread
    // measured around 12-22, so the far one pulls the policy several times
    // harder than the near one for no reason but where each was standing. The
    // shots that pull hardest are then the longest ones, which are the ones
    // whose makes carry the least information about what a good action is —
    // the achievable make rate out there is low, so a make from 45ft is mostly
    // a lucky draw from a wide sampling distribution, and the update is
    // weighted toward imitating exactly that.
    //
    // Flat makes every basket worth the same, which is what a make rate
    // measures, and leaves the advantage a reading of the shot rather than of
    // the spawn. Left parameterised — 1.25 restores the old scale — because
    // "does valuing the three help" is a real question tools/hpsearch can ask.
    scoreDistanceBonus: 0.0,
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

// The launch action, channel by channel. Ball.launch maps each onto its range
// in CONFIG.launch; the actor emits a mean and a log standard deviation for
// every one of them, so widening the action is a matter of adding a channel
// here, a range there, and the physics to apply it.
const ACTION = { fwd: 0, up: 1, side: 2, spin: 3 };
const ACTION_DIM = 4;

class Assets {
  constructor() {
    this.woodTexture = Assets.woodTexture();
    this.particleTexture = Assets.softParticleTexture();
    this.ballSkin = Assets.ballTextures();

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
    // Unlit: the backboard markings are paint on a bright white slab, and a
    // shaded black would go grey under the hoop light and lose the contrast
    // that is the whole point of them.
    this.boardMark = new THREE.MeshBasicMaterial({ color: 0x111111 });
    // Unlit, and the brightest thing on the court: the net is the agents'
    // marker for where the hole is, and a shaded white would go grey under the
    // basket exactly where it is needed most.
    this.net = new THREE.LineBasicMaterial({ color: 0xffffff });
    this.rim = new THREE.MeshStandardMaterial({
      color: 0xff4400,
      metalness: 0.6,
      roughness: 0.2,
      emissive: 0xff2200,
      emissiveIntensity: 1.0
    });
    this.highContrast = new THREE.MeshBasicMaterial({ color: 0xffffff });
    this.manualBall = new THREE.MeshStandardMaterial({
      map: this.ballSkin.map,
      bumpMap: this.ballSkin.bumpMap,
      bumpScale: 0.012,
      color: 0xffffff,
      // Leather, not plastic: rough enough to kill the specular hotspot the
      // old 0.4 left sitting on top of the ball, and not metallic at all.
      roughness: 0.78,
      metalness: 0.0
    });
    // The batch carried no texture at all — a flat orange sphere — so a
    // thousand balls on the floor read as gumballs. It gets the same skin as
    // the manual ball.
    //
    // color is white rather than orange because BallField tints each instance
    // through setColorAt, and instance colour multiplies both the material
    // colour and the map. Leaving it orange would multiply the orange in the
    // map by orange again and turn the batch to rust.
    this.batchBall = new THREE.MeshStandardMaterial({
      map: this.ballSkin.map,
      bumpMap: this.ballSkin.bumpMap,
      bumpScale: 0.012,
      color: 0xffffff,
      roughness: 0.78,
      metalness: 0.0
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

  // Deterministic generator for the procedural textures.
  //
  // Deliberately not Math.random(): tools/hpsearch replaces Math.random with a
  // seeded stream so two configs see identical spawns, and texture generation
  // runs during App construction — so every draw taken while painting a ball
  // shifts every spawn in the training run that follows. On its own generator,
  // what the balls look like cannot move what the agent sees, and the texture
  // comes out identical on every load instead of freshly random each time.
  static _texRandom(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // The ball skin: an equirectangular colour map and a matching bump map.
  //
  // The old texture was a flat orange field with 5000 dark dots scattered over
  // it. Two things were missing, and they are the two things that actually say
  // "basketball": the seams, and the fact that the surface is pebbled leather
  // rather than paint.
  //
  // 1024x512 rather than square because SphereGeometry's UVs are
  // equirectangular — u wraps a full 360 degrees of longitude, v covers 180 of
  // latitude — so a 2:1 map is the one that comes out with roughly square
  // texels at the equator.
  //
  // Seams: the 8-panel layout is one great circle around the equator plus two
  // more through the poles, 90 degrees apart, which in this projection is a
  // horizontal line at v = 0.5 and four vertical lines a quarter-turn apart.
  // Drawn straight, that reads as a beach ball. Real panels S-bend across the
  // equator, so each pole-to-pole seam is displaced by sin(2*pi*v): zero at
  // both poles and at the equator, maximum a quarter of the way to each. Every
  // seam takes the same displacement, so the eight panels stay congruent.
  //
  // The bump map carries the same geometry inverted — pebbles raised, seams cut
  // in as grooves — which is what makes the grain catch the light instead of
  // reading as printed-on speckle.
  static ballTextures() {
    const W = 1024;
    const H = 512;
    const rnd = Assets._texRandom(0x9e3779b9);

    const colour = document.createElement("canvas");
    colour.width = W;
    colour.height = H;
    const c = colour.getContext("2d");
    const bump = document.createElement("canvas");
    bump.width = W;
    bump.height = H;
    const b = bump.getContext("2d");

    c.fillStyle = "#c25f21";
    c.fillRect(0, 0, W, H);
    // Mid grey is "flat" for a bump map: the grain has to be able to sit both
    // above and below the surface.
    b.fillStyle = "#808080";
    b.fillRect(0, 0, W, H);

    // Low-frequency tone variation, so the leather is not one flat orange.
    for (let i = 0; i < 90; i++) {
      const x = rnd() * W;
      const y = rnd() * H;
      const r = 60 + rnd() * 190;
      const g = c.createRadialGradient(x, y, 0, x, y, r);
      const warm = rnd() > 0.5;
      g.addColorStop(0, warm ? "rgba(226,124,48,0.10)" : "rgba(150,60,14,0.10)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      c.fillStyle = g;
      c.beginPath();
      c.arc(x, y, r, 0, Math.PI * 2);
      c.fill();
    }

    // Pebble grain. Each pebble is a dot with a lit top-left and a shadowed
    // bottom-right, which is what the bump map turns into relief.
    //
    // Two corrections for the projection, both of which matter because the
    // grain is the finest detail on the ball and distortion in it is obvious:
    //
    //  - latitude is drawn as acos(1 - 2u)/pi rather than uniformly, because
    //    equal areas of this map are not equal areas of the sphere. Sampling
    //    the map uniformly piles pebbles up at the poles, where the map is
    //    squeezed into almost no sphere at all;
    //  - each pebble is stretched horizontally by 1/sin(latitude), the same
    //    factor the projection is about to squeeze it by, so it arrives on the
    //    ball round instead of as a thin vertical sliver.
    for (let i = 0; i < 42000; i++) {
      const v = Math.acos(1 - 2 * rnd()) / Math.PI;
      const y = v * H;
      const x = rnd() * W;
      const r = 0.7 + rnd() * 1.4;
      // Capped: at the pole itself the factor is unbounded.
      const rx = r / Math.max(0.12, Math.sin(Math.PI * v));
      const ellipse = (ctx, cx, cy, sx, sy) => {
        ctx.beginPath();
        ctx.ellipse(cx, cy, sx, sy, 0, 0, Math.PI * 2);
        ctx.fill();
      };

      c.fillStyle =
        rnd() > 0.5
          ? `rgba(255,186,126,${0.04 + rnd() * 0.08})`
          : `rgba(74,28,6,${0.05 + rnd() * 0.12})`;
      ellipse(c, x, y, rx, r);

      // Kept deliberately gentle. Leather grain is a shallow texture; at the
      // contrast this started out with, the bump map alone turned the ball
      // into orange peel.
      b.fillStyle = `rgba(255,255,255,${0.05 + rnd() * 0.10})`;
      ellipse(b, x - rx * 0.22, y - r * 0.22, rx, r);
      b.fillStyle = `rgba(0,0,0,${0.05 + rnd() * 0.10})`;
      ellipse(b, x + rx * 0.3, y + r * 0.3, rx * 0.72, r * 0.72);
    }

    // Seams.
    const bow = W * 0.045;
    const seam = (ctx, colourStop, width) => {
      ctx.strokeStyle = colourStop;
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      // Equator.
      ctx.beginPath();
      ctx.moveTo(0, H / 2);
      ctx.lineTo(W, H / 2);
      ctx.stroke();
      // Four pole-to-pole seams. Each is drawn three times, one period of the
      // map to either side, so a seam whose S-bend carries it over the u = 0
      // edge still meets itself where the texture wraps.
      for (const u of [0, 0.25, 0.5, 0.75]) {
        for (const shift of [-W, 0, W]) {
          ctx.beginPath();
          for (let y = 0; y <= H; y += 4) {
            const x =
              u * W + shift + bow * Math.sin((2 * Math.PI * y) / H);
            if (y === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      }
    };
    // A hint of raised welt either side of the groove, then the groove itself.
    seam(c, "rgba(70,26,4,0.55)", 13);
    seam(c, "#20120a", 7);
    seam(b, "rgba(255,255,255,0.35)", 13);
    seam(b, "#202020", 7);

    const map = new THREE.CanvasTexture(colour);
    const bumpMap = new THREE.CanvasTexture(bump);
    // u wraps all the way around the ball; v runs pole to pole and must not.
    for (const t of [map, bumpMap]) {
      t.wrapS = THREE.RepeatWrapping;
      t.wrapT = THREE.ClampToEdgeWrapping;
      t.anisotropy = 4;
    }
    return { map, bumpMap };
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
    this.actions = new Float32Array(capacity * ACTION_DIM);
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
    for (let k = 0; k < ACTION_DIM; k++)
      this.actions[slot * ACTION_DIM + k] = actions[i][k];
    this.rewards[slot] = reward;
    this.priority[slot] = priority;
    return true;
  }

  // Re-price entries against the critic that is running now.
  //
  // priority is an advantage, and an advantage is only meaningful next to the
  // value estimate it was measured against. Admission compares a fresh
  // advantage to one recorded batches or thousands of batches ago, so as the
  // critic calibrates and the same quality of shot scores a smaller advantage,
  // the bar to get in rises on its own — and it rises fastest against the
  // entries admitted when the critic was worst, which are exactly the ones
  // whose numbers were most inflated.
  //
  // Left alone that bar climbs to the largest advantage a batch can produce and
  // sticks there, and the actor spends the rest of the run replaying a set that
  // barely changes. Measured headlessly on this build: the weakest held entry's
  // priority goes 0.8 -> 7.0 -> 29.3 -> 31.0 over the first sixty batches and
  // then stops moving (31.0 -> 31.1 across the next fifteen), while admissions
  // fall from 64 a batch to between nought and three — against a batch whose
  // best shot is scoring around 32. Those entries were sampled when sigma was
  // still 0.2 wide, so what the actor keeps being shown is the scatter of a
  // policy it has already outgrown.
  //
  // So re-price what we draw. _replayPass already predicts a current value for
  // every sampled entry, which is the same subtraction admission does, so this
  // costs nothing but the write-back. Sampling is uniform, so at
  // samplesPerBatch/capacity a batch every entry is re-priced every few
  // batches, and one that only looked good because the critic was wrong sinks
  // to where a current batch can evict it.
  reprice(indices, advantages) {
    for (let k = 0; k < indices.length; k++) {
      const slot = indices[k];
      if (slot >= 0 && slot < this.size) this.priority[slot] = advantages[k];
    }
  }

  // `count` distinct entries at random, as flat arrays ready for tf.tensor.
  // Uniform rather than priority-weighted: the buffer is already the top of the
  // distribution, and weighting the draw too would train the actor almost
  // entirely on whichever handful of shots scored best.
  //
  // The draw's source slots come back with it so the caller can reprice them.
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
    const actions = new Float32Array(n * ACTION_DIM);
    const rewards = new Float32Array(n);
    const slots = new Int32Array(n);
    for (let k = 0; k < n; k++) {
      const src = idx[k];
      slots[k] = src;
      states.set(this.states.subarray(src * f, (src + 1) * f), k * f);
      for (let d = 0; d < ACTION_DIM; d++)
        actions[k * ACTION_DIM + d] = this.actions[src * ACTION_DIM + d];
      rewards[k] = this.rewards[src];
    }
    return { states, actions, rewards, slots, count: n };
  }
}

// A zip writer and reader, small enough to be worth having so the policy can
// leave and arrive as one file.
//
// Nothing here compresses. The payload is a float32 weight blob that deflate
// barely dents, and a stored archive is still an ordinary zip that any tool
// will open — which is the whole reason for using the format instead of
// inventing a container. Reading does handle deflated entries, because a
// policy that has been through some other zip tool on its way back will be
// compressed and refusing it would be a strange thing for the importer to do.
class Zip {
  // entries: [{ name, data: Uint8Array }] -> Uint8Array holding the archive.
  static write(entries) {
    const encoder = new TextEncoder();
    const parts = [];
    const directory = [];
    let offset = 0;

    for (const entry of entries) {
      const name = encoder.encode(entry.name);
      const crc = Zip._crc32(entry.data);
      const size = entry.data.length;

      const local = new DataView(new ArrayBuffer(30 + name.length));
      local.setUint32(0, 0x04034b50, true);
      local.setUint16(4, 20, true); // version needed: 2.0, stored
      local.setUint16(6, 0, true); // no flags
      local.setUint16(8, 0, true); // method: stored
      // Timestamps stay zero — 1980-01-01 to anything that reads them. They
      // are local-time DOS fields, so stamping them would make two exports of
      // identical weights differ by nothing but the clock.
      local.setUint16(10, 0, true);
      local.setUint16(12, 0, true);
      local.setUint32(14, crc, true);
      local.setUint32(18, size, true);
      local.setUint32(22, size, true);
      local.setUint16(26, name.length, true);
      local.setUint16(28, 0, true);
      const localBytes = new Uint8Array(local.buffer);
      localBytes.set(name, 30);
      parts.push(localBytes, entry.data);

      const central = new DataView(new ArrayBuffer(46 + name.length));
      central.setUint32(0, 0x02014b50, true);
      central.setUint16(4, 20, true);
      central.setUint16(6, 20, true);
      // 16, not 14: the central record carries two more bytes of version
      // before the timestamps than the local header does.
      central.setUint32(16, crc, true);
      central.setUint32(20, size, true);
      central.setUint32(24, size, true);
      central.setUint16(28, name.length, true);
      central.setUint32(42, offset, true);
      const centralBytes = new Uint8Array(central.buffer);
      centralBytes.set(name, 46);
      directory.push(centralBytes);

      offset += localBytes.length + size;
    }

    let directorySize = 0;
    for (const record of directory) directorySize += record.length;

    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, entries.length, true);
    end.setUint16(10, entries.length, true);
    end.setUint32(12, directorySize, true);
    end.setUint32(16, offset, true);

    parts.push(...directory, new Uint8Array(end.buffer));
    let total = 0;
    for (const part of parts) total += part.length;
    const archive = new Uint8Array(total);
    let at = 0;
    for (const part of parts) {
      archive.set(part, at);
      at += part.length;
    }
    return archive;
  }

  // -> [{ name, data: Uint8Array }], names exactly as stored.
  static async read(buffer) {
    const bytes = new Uint8Array(buffer);
    const view = new DataView(
      bytes.buffer,
      bytes.byteOffset,
      bytes.byteLength
    );

    // The end-of-central-directory record is last and 22 bytes with no
    // comment, but scan backwards anyway so an archive carrying one opens.
    let end = -1;
    const floor = Math.max(0, bytes.length - 22 - 0xffff);
    for (let i = bytes.length - 22; i >= floor; i--)
      if (view.getUint32(i, true) === 0x06054b50) {
        end = i;
        break;
      }
    if (end < 0) throw new Error("Not a zip archive");

    const count = view.getUint16(end + 10, true);
    let at = view.getUint32(end + 16, true);
    const entries = [];
    for (let i = 0; i < count; i++) {
      if (view.getUint32(at, true) !== 0x02014b50)
        throw new Error("The archive's directory is corrupt");
      const method = view.getUint16(at + 10, true);
      const crc = view.getUint32(at + 16, true);
      const stored = view.getUint32(at + 20, true);
      const nameLength = view.getUint16(at + 28, true);
      const extraLength = view.getUint16(at + 30, true);
      const commentLength = view.getUint16(at + 32, true);
      const header = view.getUint32(at + 42, true);
      const name = new TextDecoder().decode(
        bytes.subarray(at + 46, at + 46 + nameLength)
      );

      // The local header repeats the name and may carry different extra
      // fields, so where the bytes actually start comes from its own lengths.
      const from =
        header +
        30 +
        view.getUint16(header + 26, true) +
        view.getUint16(header + 28, true);
      const raw = bytes.subarray(from, from + stored);
      const data = method === 0 ? raw : await Zip._inflate(raw);
      // A truncated download would otherwise arrive as a model made of
      // plausible-looking garbage. The checksum is right there; check it.
      if (Zip._crc32(data) !== crc)
        throw new Error(`${name} is corrupt (checksum mismatch)`);
      entries.push({ name, data });

      at += 46 + nameLength + extraLength + commentLength;
    }
    return entries;
  }

  static async _inflate(data) {
    if (typeof DecompressionStream !== "function")
      throw new Error(
        "The archive is compressed and this browser cannot unpack it"
      );
    const stream = new Blob([data])
      .stream()
      .pipeThrough(new DecompressionStream("deflate-raw"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  static _crc32(data) {
    let table = Zip._crcTable;
    if (!table) {
      table = Zip._crcTable = new Uint32Array(256);
      for (let i = 0; i < 256; i++) {
        let c = i;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        table[i] = c >>> 0;
      }
    }
    let crc = 0xffffffff;
    for (let i = 0; i < data.length; i++)
      crc = table[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
    return (crc ^ 0xffffffff) >>> 0;
  }
}

class CNNAgent {
  constructor({ learningRate, batchSize, l2 }) {
    this.learningRate = learningRate;
    this.batchSize = batchSize;
    this.l2Reg = tf.regularizers.l2({ l2 });
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
    this.memory = null;
  }

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
    // Unfreeze conv layers: let the policy learn end-to-end visual representations
    m.layers[0].trainable = true;
    m.layers[1].trainable = true;

    m.add(tf.layers.dense({ units: 2 * ACTION_DIM }));
    m.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: "meanSquaredError"
    });

    tf.tidy(() => {
      const head = m.layers[m.layers.length - 1];
      const [kernel, bias] = head.getWeights();
      const b = bias.arraySync().slice();
      for (let i = ACTION_DIM; i < 2 * ACTION_DIM; i++)
        b[i] = CONFIG.policy.logStdInit;
      head.setWeights([kernel, tf.tensor1d(b)]);
    });

    this.actorOptimizer = tf.train.adam(this.learningRate);
    return m;
  }

  // Bound the spread the sampler sees without deleting the evidence that the
  // bound is in the wrong place.
  //
  // clipByValue's gradient is exactly zero outside its range, so logStdMin was
  // not a floor, it was an absorbing state: the first update that pushed a
  // state's log sigma under it removed the only term that could ever push it
  // back, and that state stopped exploring for the remainder of the run. The
  // objective's own equilibrium — log(sigma) pulling the spread down, z^2
  // pushing it up whenever the fitted actions scatter wider — exists only
  // where the gradient survives, and a one-sided ratchet to the floor is what
  // is left where it does not. That is the same failure the loss in
  // _fitActorWeighted has a long comment about; it was fixed there and left
  // standing here, where it is the sampler's bound rather than the loss's.
  //
  // Straight-through resolves it: the value handed downstream is clipped, so
  // nothing samples or reports a sigma outside CONFIG.policy's range, while
  // the gradient passes unchanged, so a policy whose good actions scatter
  // wider than the floor keeps being told so and climbs back off it. Inside
  // the range this is the identity — a policy that never reaches a bound
  // cannot tell the difference.
  _boundLogStd(x) {
    // Built on first use: tf is a global from a CDN script tag, and a class
    // field would evaluate before it exists in the harness's load order.
    if (!CNNAgent._clipST) {
      CNNAgent._clipST = tf.customGrad((t) => ({
        value: t.clipByValue(CONFIG.policy.logStdMin, CONFIG.policy.logStdMax),
        gradFunc: (dy) => [dy]
      }));
    }
    return CNNAgent._clipST(x);
  }

  _headSplit(raw) {
    const mean = tf.tanh(raw.slice([0, 0], [-1, ACTION_DIM]));
    const logStd = this._boundLogStd(
      raw.slice([0, ACTION_DIM], [-1, ACTION_DIM])
    );
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

  static _gauss() {
    const u = 1 - Math.random();
    const v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  predictBatch(pixelDataBatch, greedyCount = 0) {
    return tf.tidy(() => {
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

      const clamp = (v) => Math.max(-1, Math.min(1, v));
      const actions = [];
      for (let i = 0; i < count; i++) {
        const a = [];
        for (let k = 0; k < ACTION_DIM; k++) {
          const j = i * ACTION_DIM + k;
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
      const { mean } = this._headSplit(this.actor.layers[4].apply(dense64));
      return { dense: dense64.dataSync(), output: mean.dataSync() };
    });
  }

  // greedyCount is the batch's eval slice: the balls at the front that were
  // launched on the policy's mean rather than on a draw from it (see
  // CONFIG.evalBalls). They are stored like everything else — see train() for
  // what they do and do not take part in.
  store(states, actions, rewards, greedyCount = 0) {
    this.memory = {
      states,
      actions,
      rewards,
      greedyCount,
      count: actions.length
    };
  }

  // Weight only positive-advantage transitions; zero out misses and sub-baseline actions
  // How hard a sample pulls: the gate is whether the shot beat the critic, the
  // weight is how far it beat the batch.
  //
  // Centred on the batch's own mean advantage, which is what makes the weight
  // a comparison between shots rather than a reading of the critic's current
  // calibration. Uncentred, z carries advMean/advStd — measured wandering
  // between -0.34 and +0.37 batch to batch — so every weight in the batch is
  // multiplied by exp(advMean / (advStd * advantageTemp)) for no reason but
  // how far the value head happens to sit from the returns.
  //
  // That factor is uniform across the batch, and _fitActorWeighted normalizes
  // by the sum of the weights, so most of it divides straight back out. The
  // one place it survives is the clip: weights pinned at CONFIG.advantageClip
  // are equal weights, so a batch pushed into the clip together is a batch
  // that has stopped being weighted at all — and it is the lagging critic, the
  // case where advMean/advStd is largest, that pushes it there. The clip
  // rarely binds today (nought to one sample in most batches), so this is
  // insurance rather than a fix for something currently on fire.
  //
  // It also gives advMean a job again. It was still being passed in and
  // ignored, which is the shape of a term dropped by accident.
  _advantageWeight(advantage, advMean, advStd) {
    if (advantage <= 0) return 0;
    const z = (advantage - advMean) / advStd;
    return Math.min(CONFIG.advantageClip, Math.exp(z / CONFIG.advantageTemp));
  }

  async train() {
    if (!this.memory) return null;
    const {
      states,
      actions,
      rewards,
      greedyCount,
      count: batchSize
    } = this.memory;

    const stateTensor = tf.tensor(states, [
      batchSize,
      this.visionH,
      this.visionW,
      this.channels
    ]);
    const actionTensor = tf.tensor2d(actions, [batchSize, ACTION_DIM]);
    const rewardTensor = tf.tensor2d(rewards, [batchSize, 1]);

    const criticHistory = await this.critic.fit(stateTensor, rewardTensor, {
      epochs: CONFIG.criticEpochs,
      verbose: 0
    });
    const losses = criticHistory.history.loss;
    const loss = losses && losses.length ? losses[losses.length - 1] : 0;

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

    // The eval slice is held out of the actor's update, and it is the
    // objective rather than the shot that holds it out.
    //
    // Those balls were launched on the policy's mean, so the action being
    // fitted IS the mean it is being fitted to. Under a regression objective
    // that made them the batch's best imitation target, which is what the note
    // on CONFIG.evalBalls says. Under the likelihood objective that replaced
    // it, it makes them degenerate: -log N(a | mu, sigma) at a == mu has
    // gradient -z/sigma on the mean, which is zero, and +1 per dimension on
    // log sigma. They cannot move the mean at all, and the only thing they say
    // about the spread is "smaller" — with no evidence in them about how far
    // good actions actually scatter, because they were not allowed to scatter.
    //
    // The size of it: _fitActorWeighted normalizes by the sum of the weights,
    // so the fitted variance is the weight-average of (a - mu)^2, and a slice
    // holding a fraction f of the batch's weight scales it by (1 - f)
    // regardless of what the policy is doing. The sampler then draws from what
    // comes out, so "sigma equals (1 - f) times the spread of actions drawn
    // from sigma" has one fixed point, at zero. The slice was added to measure
    // the policy; left in the update it walks the policy's exploration to the
    // floor on a schedule that has nothing to do with what the policy has
    // learned — and the floor, before _boundLogStd above, was where
    // exploration ended for good.
    //
    // They still fit the critic, and they are still admitted to replay: by the
    // time a stored shot is drawn again the actor has moved, its action is no
    // longer the mean, and it carries an ordinary gradient on both.
    const weightData = new Float32Array(batchSize);
    for (let i = 0; i < batchSize; i++)
      weightData[i] =
        i < greedyCount
          ? 0
          : this._advantageWeight(advantageData[i], advMean, advStd);
    const weightTensor = tf.tensor1d(weightData);

    // Train the actor end-to-end across its conv backbone and dense head
    this._fitActorWeighted(stateTensor, actionTensor, weightTensor);

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
    const actions = tf.tensor2d(draw.actions, [draw.count, ACTION_DIM]);
    const values = this.critic.predict(states);
    const valueData = values.dataSync();

    const weightData = new Float32Array(draw.count);
    const repriced = new Float32Array(draw.count);
    for (let i = 0; i < draw.count; i++) {
      repriced[i] = draw.rewards[i] - valueData[i];
      weightData[i] = this._advantageWeight(repriced[i], advMean, advStd);
    }
    // The advantage just computed is the entry's price under the current
    // critic. See ReplayBuffer.reprice.
    this.replay.reprice(draw.slots, repriced);
    const weights = tf.tensor1d(weightData);

    this._fitActorWeighted(states, actions, weights);

    states.dispose();
    actions.dispose();
    values.dispose();
    weights.dispose();
  }

  _admitToReplay(states, actions, rewards, advantageData) {
    if (!this.replay) return;
    const n = rewards.length;
    const ranked = new Array(n);
    for (let i = 0; i < n; i++) ranked[i] = i;
    ranked.sort((a, b) => advantageData[b] - advantageData[a]);
    const take = Math.min(CONFIG.replay.admitPerBatch, n);
    for (let k = 0; k < take; k++) {
      const i = ranked[k];
      if (advantageData[i] > 0) {
        this.replay.offer(states, actions, rewards[i], advantageData[i], i);
      }
    }
  }

  // Minibatch update with bounded sigma precision and weight-normalized loss
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

        const weightSum = wb.sum().dataSync()[0];
        if (weightSum <= 1e-6) return;

        this.actorOptimizer.minimize(
          () => {
            const raw = this.actor.apply(xb, { training: true });
            const { mean, logStd, std } = this._headSplit(raw);

            // Both terms see the same sigma. This is -log N(a | mean, std)
            // dropping the constant 0.5*log(2*pi) per dimension, and the two
            // halves only balance because they are the same distribution's:
            // the log(sigma) term pushes the spread down, and the z^2 term
            // pushes it up whenever the actions being fitted scatter wider
            // than it, which is what gives sigma an equilibrium at the actual
            // spread of the good actions.
            //
            // Flooring sigma in the z^2 term alone breaks that. Below the
            // floor the quadratic has no gradient on sigma at all while
            // log(sigma) keeps pulling down, so there is no equilibrium left
            // under it — only a one-way ratchet to logStdMin. Measured: the
            // policy's mean sigma fell to 0.020 by the ninth batch and stayed
            // pinned there, against a run that dipped to 0.032 and recovered
            // to 0.17. A policy that has stopped exploring stops generating
            // the variety its own update is computed from.
            //
            // The precision term is bounded without the floor: _headSplit
            // clips logStd to CONFIG.policy.logStdMin, so 1/sigma^2 cannot
            // exceed 1/logStdMin^2. Raising that config value is the way to
            // bound it harder, because it bounds the sampler and the loss
            // together rather than putting them at odds.
            const z = yb.sub(mean).div(std);
            const perSample = z.square().mul(0.5).add(logStd).sum(1);

            // Normalize by sum of positive weights rather than batch size
            return perSample.mul(wb).sum().div(tf.scalar(weightSum));
          },
          false,
          vars
        );
      });
    }
  }

  // The policy leaves as one file. The archive holds exactly what TensorFlow's
  // own two-file save produces — the topology JSON and its weight blob — so
  // anything that unzips it gets an ordinary tfjs model back, while what the
  // user keeps, moves and hands to IMPORT POLICY is a single thing that cannot
  // arrive half missing.
  async saveActor() {
    const WEIGHTS = "model.weights.bin";
    const entries = [];

    await this.actor.save(
      tf.io.withSaveHandler(async (artifacts) => {
        entries.push({
          name: "model.json",
          data: new TextEncoder().encode(
            JSON.stringify({
              modelTopology: artifacts.modelTopology,
              format: artifacts.format,
              generatedBy: artifacts.generatedBy,
              convertedBy: artifacts.convertedBy,
              // One shard: the actor is a few hundred KB, and the multi-file
              // sharding tfjs does for large models is the thing being
              // collapsed here.
              weightsManifest: [
                { paths: [WEIGHTS], weights: artifacts.weightSpecs }
              ]
            })
          )
        });
        entries.push({
          name: WEIGHTS,
          data: new Uint8Array(artifacts.weightData)
        });
        return { modelArtifactsInfo: tf.io.getModelArtifactsInfoForJSON(artifacts) };
      })
    );

    const url = URL.createObjectURL(
      new Blob([Zip.write(entries)], { type: "application/zip" })
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "basketball-agent-actor.zip";
    anchor.click();
    // The object URL pins the archive in memory until it is revoked, and
    // revoking it in the same task can cancel the download that was just
    // started, so let the click get clear of it first.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // The mirror of saveActor: one archive in, the running policy replaced.
  //
  // The zip is unpacked in memory and handed to tf.io.browserFiles as the File
  // pair it already knows how to read, so the loading itself is still tfjs's —
  // the archive is a container, not a second format. Names are reduced to
  // their basename on the way through, so an archive somebody rebuilt with the
  // two files inside a folder still loads.
  //
  // The weights are copied into the existing actor rather than the loaded
  // model being swapped in for it: this.actor carries state a fresh
  // deserialization does not — its compile, and the layer indices
  // getActivations() reads — and setWeights assigns into the variables
  // actorOptimizer is already bound to, so training resumes on the imported
  // policy instead of on an optimizer that has never seen it.
  async loadActor(file) {
    if (!file) throw new Error("No archive selected");

    const entries = (await Zip.read(await file.arrayBuffer())).map((entry) => ({
      data: entry.data,
      name: entry.name.slice(entry.name.lastIndexOf("/") + 1)
    }));
    const topology = entries.find((entry) => entry.name.endsWith(".json"));
    const binaries = entries.filter((entry) => entry.name.endsWith(".bin"));
    if (!topology || !binaries.length)
      throw new Error(
        "The archive holds no policy: expected a model .json and its .bin"
      );

    const asFile = (entry, type) => new File([entry.data], entry.name, { type });
    const loaded = await tf.loadLayersModel(
      tf.io.browserFiles([
        asFile(topology, "application/json"),
        ...binaries.map((entry) => asFile(entry, "application/octet-stream"))
      ])
    );
    try {
      const source = loaded.getWeights();
      const target = this.actor.getWeights();
      if (source.length !== target.length)
        throw new Error(
          `Expected ${target.length} weight tensors, file has ${source.length}`
        );
      for (let i = 0; i < target.length; i++) {
        const want = target[i].shape;
        const got = source[i].shape;
        if (want.length !== got.length || want.some((d, k) => d !== got[k]))
          throw new Error(
            `Weight ${i} is ${got.join("x")} in the file, ` +
              `this build expects ${want.join("x")}`
          );
      }

      // Only the actor. The critic used to own the shared conv backbone and
      // copy it into the actor after every batch, so an import that did not
      // also seed the critic would have been undone by the first update — that
      // is what the copy here was for. The nets have had independent backbones
      // since the actor started training end-to-end, and copying now does the
      // opposite of repair: it throws away the critic's own learned features
      // and replaces them with the imported actor's, leaving the value
      // estimate that every advantage is measured against to relearn from a
      // stranger's encoder.
      this.actor.setWeights(source);
    } finally {
      loaded.dispose();
    }
  }
}

class SceneManager {
  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x222222);
    // No fog. It faded to the background colour from 50ft out, which on a
    // 94ft court meant the far half of the floor and the whole far hoop were
    // washed toward flat grey — in the player's view and, more to the point,
    // in the 96px one the agent is trained on. The court is an indoor space
    // the length of a room; nothing in it is far enough away for haze.

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

    // The spot over the target hoop, dimmer than it was. It used to be the
    // only real light in the building, so it had to carry the whole picture at
    // intensity 2 and did it by blowing out the paint under the basket. With
    // the banks below lighting the floor, it goes back to being what it looks
    // like — the basket picked out a little brighter than the rest.
    const hoop = new THREE.SpotLight(0xffffff, 0.8);
    hoop.position.set(25, 25, 0);
    hoop.target.position.set(CONFIG.rim.x, CONFIG.rim.y, CONFIG.rim.z);
    hoop.angle = Math.PI / 6;
    hoop.penumbra = 0.2;
    hoop.castShadow = true;

    this.scene.add(ambient, dir, hoop, hoop.target);
    for (const light of SceneManager._arenaBanks())
      this.scene.add(light, light.target);
  }

  // Six lights in two banks, hung outside the sidelines and angled in across
  // the court, which is how a gym is actually lit: nothing hangs over the
  // playing surface, and every point on the floor is covered from both sides
  // so the pools overlap instead of leaving the far end in the dark.
  //
  // The point is not atmosphere. The agents see the court through a 96x96
  // grayscale window, and under a single hoop spot the floor away from the
  // basket sat at nearly the value of the unlit background — a ball out there
  // was a dark blob against dark nothing, which is a bad state to have to
  // learn a shot from. Even light across the floor is what makes the ball, the
  // markings and the far half of the court legible at that resolution.
  //
  // None of the six casts a shadow, deliberately. VisionSystem renders the
  // scene twice per ball — 2048 renders a batch at batchSize 1024 — and every
  // shadow-casting light adds a full shadow pass to each of them. The
  // directional key and the hoop spot stay the only casters, so the ball keeps
  // the floor shadow that tells the agent how high it is; the banks are fill
  // and cost only their share of the fragment shader.
  static _arenaBanks() {
    const lights = [];
    // 32ft out puts the rig past the 25ft sideline, and 42ft up clears the
    // backboards and their poles by three times the height of a player.
    for (const z of [-32, 32])
      for (const x of [-30, 0, 30]) {
        // Slightly warm, the way a gym reads next to the daylight-white key.
        const light = new THREE.SpotLight(0xfff4e6, 0.5);
        light.position.set(x, 42, z);
        // Aimed past the middle line rather than straight down, so each bank
        // washes the far side of the court and the two overlap in the middle.
        light.target.position.set(x, 0, -z * 0.35);
        light.angle = 0.55;
        light.penumbra = 0.8;
        light.castShadow = false;
        lights.push(light);
      }
    return lights;
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

// Drag and Magnus for one step, accumulated straight into each body's force
// vector. See CONFIG.air for where the constants come from.
//
// Written against body.force directly rather than through applyForce(), which
// would allocate two Vec3s per body per step — 120k allocations a second at
// batchSize 1024, for arithmetic that is a dozen multiplies.
function applyAir(bodies) {
  const A = CONFIG.air;
  if (!A.enabled) return;
  for (const b of bodies) {
    // Retired balls are asleep and the integrator skips them anyway.
    if (b.sleepState === CANNON.Body.SLEEPING) continue;
    // Null on the singular-air default, and undefined before the first launch;
    // both mean "the gym's own air".
    const air = b.air || A;
    const v = b.velocity;
    // Relative airspeed. Wind belongs here rather than as a separate push:
    // what drags on the ball is its motion *through the air*, so a breeze
    // makes a shot into it slow faster and one with it carry, which is the
    // effect worth having.
    const rx = v.x - (air.windX || 0);
    const ry = v.y;
    const rz = v.z - (air.windZ || 0);
    const speed = Math.sqrt(rx * rx + ry * ry + rz * rz);
    if (speed < 1e-4) continue;
    const w = b.angularVelocity;
    const m = b.mass;
    // a = -k|vr|vr + C(w x vr), then F = ma.
    b.force.x +=
      m * (-air.drag * speed * rx + air.magnus * (w.y * rz - w.z * ry));
    b.force.y +=
      m * (-air.drag * speed * ry + air.magnus * (w.z * rx - w.x * rz));
    b.force.z +=
      m * (-air.drag * speed * rz + air.magnus * (w.x * ry - w.y * rx));
  }
}

// Roll the air for one shot and hang it on the body. See CONFIG.air.
function sampleShotAir(body) {
  const A = CONFIG.air;
  const j = A.jitter;
  // The default: one air for the whole gym. Returning before drawing anything
  // matters — it keeps the singular model free of RNG draws, so turning the
  // weather off does not shift the seeded spawn stream a run is compared on.
  if (!A.wind && !j.drag && !j.magnus) {
    body.air = null;
    return;
  }
  const spread = (f) => 1 + (Math.random() * 2 - 1) * f;
  const heading = Math.random() * Math.PI * 2;
  // sqrt keeps the draw uniform over the disc of possible breezes rather than
  // clustering it at dead calm.
  const speed = Math.sqrt(Math.random()) * A.wind;
  body.air = {
    drag: A.drag * spread(j.drag),
    magnus: A.magnus * spread(j.magnus),
    windX: Math.cos(heading) * speed,
    windZ: Math.sin(heading) * speed
  };
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

    this.aeroBodies = [];

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

  // Bodies the air acts on: the balls, and nothing else. The court does not
  // move and the forces would be discarded on a static body regardless.
  addAero(body) {
    this.aeroBodies.push(body);
  }

  step() {
    // Before the step, because cannon clears the force accumulator at the end
    // of every one.
    applyAir(this.aeroBodies);
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
    this._buildMarkings();

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

  // NBA regulation markings, all of them, as a single mesh.
  //
  // The court was already built to the rulebook — 94x50, and CONFIG.rim.x of
  // 41.75 is exactly the 4ft from the baseline to the backboard face plus the
  // 15in from the face to the middle of the ring — so every number below is
  // the real one rather than something eyeballed to fit.
  //
  // One mesh matters more here than it looks. VisionSystem renders the scene
  // twice per ball, which is 2048 renders a batch at batchSize 1024, so a draw
  // call added to the court is a draw call paid two thousand times before the
  // batch has been trained on once. Merged, the full set of markings costs one
  // — fewer than the five plain boundary lines it replaces.
  _buildMarkings() {
    const W = 0.17; // 2in stripe, the width the rulebook gives
    const g = [];
    const rect = (w, h, x, z) => {
      const geom = new THREE.PlaneGeometry(w, h);
      geom.rotateX(-Math.PI / 2);
      geom.translate(x, 0.02, z);
      g.push(geom);
    };
    // Angles are measured in the ring's own plane before it is laid flat, and
    // that rotation sends local +y to world -z — so an arc drawn from thetaStart
    // opens the way it reads here once it is on the floor.
    const arc = (cx, cz, r, from, len, seg = 96) => {
      const geom = new THREE.RingGeometry(r - W / 2, r + W / 2, seg, 1, from, len);
      geom.rotateX(-Math.PI / 2);
      geom.translate(cx, 0.02, cz);
      g.push(geom);
    };
    const dashedArc = (cx, cz, r, from, len, dashes) => {
      // Half stripe, half gap, which is what the far side of the free-throw
      // circle looks like.
      const step = len / (dashes * 2 - 1);
      for (let i = 0; i < dashes; i++)
        arc(cx, cz, r, from + i * 2 * step, step, 8);
    };

    // Boundary and half court.
    rect(94.17, W, 0, -25);
    rect(94.17, W, 0, 25);
    rect(W, 50, -47, 0);
    rect(W, 50, 47, 0);
    rect(W, 50, 0, 0);
    // Centre circle, and the 2ft one inside it.
    arc(0, 0, 6, 0, Math.PI * 2);
    arc(0, 0, 2, 0, Math.PI * 2);

    for (const isLeft of [true, false]) {
      const sign = isLeft ? -1 : 1;
      const rimX = sign * CONFIG.rim.x;
      const baseX = sign * 47;
      // The half's markings all open toward centre court.
      const facing = isLeft ? 0 : Math.PI;

      // Free-throw lane: 16ft wide, baseline to the free-throw line. The line
      // sits 15ft from the backboard face, which is 4ft in from the baseline.
      const ftX = sign * 28;
      rect(19, W, sign * 37.5, -8);
      rect(19, W, sign * 37.5, 8);
      rect(W, 16, ftX, 0);
      // Free-throw circle. The half away from the basket is solid; the half
      // inside the lane is dashed, which is the way round the rulebook has it.
      arc(ftX, 0, 6, facing - Math.PI / 2, Math.PI);
      dashedArc(ftX, 0, 6, facing + Math.PI / 2, Math.PI, 8);

      // Three-point line. The arc is 23.75ft from the middle of the ring; the
      // corners are straight, 22ft out — 3ft in from the sideline — and run
      // back to the baseline from where they meet the arc.
      const R3 = 23.75;
      const CORNER = 22;
      // Where arc meets corner, as a distance from the ring along the court.
      const meet = Math.sqrt(R3 * R3 - CORNER * CORNER);
      const meetX = rimX - sign * meet;
      const half = Math.acos(meet / R3);
      arc(rimX, 0, R3, facing - half, 2 * half);
      const cornerLen = Math.abs(baseX - meetX);
      for (const z of [-CORNER, CORNER])
        rect(cornerLen, W, (baseX + meetX) / 2, z);

      // Restricted area: 4ft from under the ring, closed off to the backboard.
      arc(rimX, 0, 4, facing - Math.PI / 2, Math.PI);
      const boardX = sign * 43;
      for (const z of [-4, 4])
        rect(Math.abs(boardX - rimX), W, (boardX + rimX) / 2, z);
    }

    this.group.add(new THREE.Mesh(Court._mergeFlat(g), this.assets.line));
  }

  // Concatenates flat geometries into one. Only positions are carried across:
  // the markings are painted with an unlit MeshBasicMaterial, so normals and
  // uvs would be bytes uploaded for nothing.
  //
  // Hand-rolled rather than pulled from examples/jsm/utils/BufferGeometryUtils
  // — the merge these need is a concatenation of one attribute, and the app
  // otherwise takes a single examples module.
  static _mergeFlat(geometries) {
    const flat = geometries.map((geom) => (geom.index ? geom.toNonIndexed() : geom));
    let total = 0;
    for (const geom of flat) total += geom.attributes.position.array.length;
    const positions = new Float32Array(total);
    let offset = 0;
    for (const geom of flat) {
      positions.set(geom.attributes.position.array, offset);
      offset += geom.attributes.position.array.length;
      geom.dispose();
    }
    const merged = new THREE.BufferGeometry();
    merged.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return merged;
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

    // The backboard, to the rulebook: 6ft wide by 3.5ft tall (72x42in), its
    // face 4ft in from the baseline, and its lower edge 9.5ft off the floor
    // (2.90m). Those three put the ring — 10ft up and 15in out from the face —
    // 6in above the bottom edge and centered on the board, and top the board
    // out at 13ft, which is the geometry every bank shot in the sport is
    // taken against.
    //
    // Only the height was ever wrong: the board hung with its bottom edge at
    // 9ft, a foot below the ring instead of six inches, so it presented half a
    // foot of extra board under the rim and half a foot too little above it.
    // The face was also 0.6in proud of the 4ft line, a leftover from when the
    // board was a 0.1ft sheet centered on boardX; with the face on the line,
    // the 1.25ft to the middle of the ring is exactly the rulebook's 6in of
    // gap plus the ring's 9in radius.
    const BOARD_WIDTH = 6;
    const BOARD_HEIGHT = 3.5;
    const BOARD_BOTTOM = 9.5;
    const boardCenterY = BOARD_BOTTOM + BOARD_HEIGHT / 2;

    // The board is a solid slab rather than the 0.1ft sheet of glass it used
    // to be, because at 0.1ft the balls went straight through it.
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
    // behind it. The depth is spent backwards, away from the court, so the
    // face the ball meets is the regulation plane and not 7in in front of it.
    const boardHalfDepth = 0.6;
    const boardFaceX = boardX;
    const boardCenterX = boardFaceX + sign * boardHalfDepth;

    const board = new THREE.Mesh(
      new THREE.BoxGeometry(boardHalfDepth * 2, BOARD_HEIGHT, BOARD_WIDTH),
      this.assets.glass
    );
    board.position.set(boardCenterX, boardCenterY, 0);
    group.add(board);
    this.boardMeshes.push(board);

    group.add(
      this._boardMarkings(boardFaceX, sign, boardCenterY, {
        width: BOARD_WIDTH,
        height: BOARD_HEIGHT,
        bottom: BOARD_BOTTOM
      })
    );

    const rim = new THREE.Mesh(
      new THREE.TorusGeometry(0.75, 0.05, 16, 100),
      this.assets.rim
    );
    rim.rotation.x = Math.PI / 2;
    rim.position.set(rimX, 10, 0);
    group.add(rim);
    if (!isLeft) this.rimMesh = rim;

    group.add(this._buildNet(rimX));

    const poleBody = this._staticBody(
      baseX + sign * 6,
      6,
      0,
      new CANNON.Box(new CANNON.Vec3(0.5, 6, 0.5))
    );
    this.physics.add(poleBody);

    const boardBody = this._staticBody(
      boardCenterX,
      boardCenterY,
      0,
      new CANNON.Box(
        new CANNON.Vec3(boardHalfDepth, BOARD_HEIGHT / 2, BOARD_WIDTH / 2)
      )
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

  // The rulebook's backboard markings: a 2in line around the border of the
  // board, and the 24x18in rectangle centered behind the ring with its base
  // level with the ring at 10ft.
  //
  // Black, not white. The rules give both — white lines on a transparent
  // board, black on a solid one — and this board renders as a solid white
  // slab, so white would be a marking nobody can see. Black on white is also
  // the strongest edge anywhere on the board at 96px, which matters more here
  // than it does in a gym: the rectangle is the only aiming feature the board
  // has, and a bank shot is the agent lining that rectangle up.
  //
  // Eight quads, merged into one mesh, half an inch off the face — far enough
  // out of the depth buffer's way at this near plane, near enough that a ball
  // meeting the glass still meets it flush. One mesh because the vision
  // capture renders the scene twice per ball, so a draw call added here is
  // paid two thousand times a batch (see _buildMarkings).
  _boardMarkings(faceX, sign, centerY, board) {
    const W = 0.1667; // 2in line, the width the rulebook gives
    const x = faceX - sign * 0.05;
    const g = [];
    // Built in the board's plane and turned to face the court: after the
    // rotation the quad's width runs along z and its height along y.
    const rect = (w, h, z, y) => {
      const geom = new THREE.PlaneGeometry(w, h);
      geom.rotateY((-sign * Math.PI) / 2);
      geom.translate(x, y, z);
      g.push(geom);
    };

    // Border, inset so the outer edge of the line is the edge of the board.
    const halfW = board.width / 2;
    const halfH = board.height / 2;
    rect(board.width, W, 0, centerY + halfH - W / 2);
    rect(board.width, W, 0, centerY - halfH + W / 2);
    for (const side of [-1, 1])
      rect(W, board.height - 2 * W, side * (halfW - W / 2), centerY);

    // Shooter's rectangle: 24in wide, 18in tall, outer edges, sitting on the
    // ring's plane rather than on the bottom of the board.
    const RECT_W = 2;
    const RECT_H = 1.5;
    const base = CONFIG.rim.y;
    rect(RECT_W, W, 0, base + W / 2);
    rect(RECT_W, W, 0, base + RECT_H - W / 2);
    for (const side of [-1, 1])
      rect(W, RECT_H - 2 * W, side * (RECT_W / 2 - W / 2), base + RECT_H / 2);

    return new THREE.Mesh(Court._mergeFlat(g), this.assets.boardMark);
  }

  // The net: twelve strands hung off the ring, tapering in over 15in, drawn as
  // one LineSegments.
  //
  // Lines rather than geometry, and no physics body. A net is a soft body, and
  // simulating one per hoop for a batch of a thousand balls is a different
  // project — cannon has no cloth, and faking it with linked bodies would put
  // hundreds of constraints under a rim that a ball passes through in three
  // steps. Nothing about scoring needs it either: the sensor under the ring
  // decides makes, and it is the ring that a shot bounces off.
  //
  // What the net is for is reading the basket. The ring is a 0.1ft torus seen
  // nearly edge-on from a ball's height, which at 96px is a few orange pixels;
  // the net hangs a bright, tapering cone below it that says where the hole is
  // and, because it is a fixed real size, roughly how far away it is. One
  // draw call per hoop, unlit, so it costs the vision capture nothing but its
  // own pixels and stays the same white in both render modes.
  _buildNet(rimX) {
    const STRANDS = 12;
    const ROWS = 4;
    const TOP_R = 0.75; // hung off the ring, so the ring's own radius
    const BOTTOM_R = 0.45; // the taper every net has under its own weight
    const LENGTH = 1.25; // 15in, the short end of the legal range

    // Half a step of twist per row is what turns pairs of strands into the
    // diamonds a net is actually knotted into.
    const node = (row, i) => {
      const t = row / ROWS;
      const r = TOP_R + (BOTTOM_R - TOP_R) * t;
      const a = ((i + row / 2) / STRANDS) * Math.PI * 2;
      return [
        rimX + Math.cos(a) * r,
        CONFIG.rim.y - LENGTH * t,
        Math.sin(a) * r
      ];
    };

    const pts = [];
    const seg = (a, b) => pts.push(...a, ...b);
    for (let row = 0; row < ROWS; row++)
      for (let i = 0; i < STRANDS; i++) {
        // Down-left and down-right from every node: between them the two
        // diagonals draw both sets of strands and close the diamonds.
        seg(node(row, i), node(row + 1, i));
        seg(node(row, i), node(row + 1, i - 1));
      }
    // The hem, which is what keeps the bottom row from reading as a fringe.
    for (let i = 0; i < STRANDS; i++)
      seg(node(ROWS, i), node(ROWS, i + 1));

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return new THREE.LineSegments(geom, this.assets.net);
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
      // No linearDamping: CONFIG.air's quadratic drag models the same thing
      // physically, and running both would count the air twice. angularDamping
      // stays as the stand-in for spin bleeding off over a flight.
      linearDamping: 0,
      angularDamping: 0.1
    });
    this.body.collisionFilterGroup = CONFIG.groups.ball;
    this.body.collisionFilterMask = CONFIG.groups.court;
    physics.add(this.body);
    physics.addAero(this.body);

    this._resetState();
  }

  _resetState() {
    this.active = false;
    this.action = new Array(ACTION_DIM).fill(0);
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
    const vFwd = lerp(L.fwdMin, L.fwdMax, action[ACTION.fwd]);
    const vUp = lerp(L.upMin, L.upMax, action[ACTION.up]);
    const vSide = action[ACTION.side] * L.side;
    const spin = action[ACTION.spin] * L.spin;

    const m = this.body.mass;
    const impulse = new THREE.Vector3()
      .add(dirToHoop.clone().multiplyScalar(vFwd * m))
      .add(new THREE.Vector3(0, 1, 0).multiplyScalar(vUp * m))
      .add(dirSide.clone().multiplyScalar(vSide * m));

    // This shot's air, drawn fresh (see CONFIG.air). Every ball in the batch
    // flies through slightly different weather, so two identical actions from
    // the same spot do not have to produce the same result.
    sampleShotAir(this.body);

    this.body.wakeUp();
    this.body.applyImpulse(
      new CANNON.Vec3(impulse.x, impulse.y, impulse.z),
      new CANNON.Vec3(0, 0, 0)
    );

    // Spin is set rather than applied: a torque impulse would have to be
    // divided through the sphere's inertia to get here anyway, and the shot
    // starts from rest, so the angular velocity IS the action.
    //
    // The axis is -dirSide. dirSide is UP x dirToHoop, so for a shot heading
    // down +x it is -z; backspin — the top of the ball travelling back toward
    // the shooter — is +z about that same shot, which is -dirSide. Setting it
    // after wakeUp() matters: Body.sleep() zeroes both velocities, so anything
    // written before the wake is discarded.
    const w = dirSide.clone().multiplyScalar(-spin);
    this.body.angularVelocity.set(w.x, w.y, w.z);
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
    // One node per action channel, centred on the panel however many there
    // are, so adding a channel does not push the column off the canvas.
    const outPositions = [];
    for (let i = 0; i < ACTION_DIM; i++)
      outPositions.push({
        x: outX,
        y: outYStart + outGap * (i - (ACTION_DIM - 1) / 2) + outGap
      });

    ctx.lineWidth = 0.5;
    for (let i = 0; i < 64; i++) {
      if (dense[i] > 0.1) {
        for (let j = 0; j < ACTION_DIM; j++) {
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
    // Vertical power, forward power, aim adjustment, spin.
    const labels = ["V", "F", "A", "S"];
    for (let i = 0; i < ACTION_DIM; i++) {
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
      const reward = R.score + R.scoreDistanceBonus * shotDistance;
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
    this.agent.store(this.batchPixels, actions, rewards, this.evalBalls);

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
      // No linearDamping: CONFIG.air's quadratic drag models the same thing
      // physically, and running both would count the air twice. angularDamping
      // stays as the stand-in for spin bleeding off over a flight.
      linearDamping: 0,
      angularDamping: 0.1
    });
    this.body.collisionFilterGroup = CONFIG.groups.ball;
    this.body.collisionFilterMask = CONFIG.groups.court;
    physics.add(this.body);
    physics.addAero(this.body);

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
        // The player's shot flies through its own air too, same as the batch's.
        sampleShotAir(this.body);
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
    const btnImport = document.getElementById("import-btn");
    const importInput = document.getElementById("import-input");

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

    btnImport.addEventListener("click", () => {
      if (!this.agent) return;
      // Refuse mid-training rather than importing over a batch already in
      // flight: those shots were taken by the outgoing policy, and the update
      // they trigger when the batch lands would fit the freshly imported actor
      // back toward them.
      if (this.trainingMode) {
        this.dashboard.setStatus("Stop training to import", "#ff0");
        return;
      }
      importInput.click();
    });

    importInput.addEventListener("change", async () => {
      // Take the file before clearing the picker: value = "" empties the live
      // FileList, and the clear has to happen either way so that re-selecting
      // the same archive still fires change — the obvious thing to do after a
      // failed import.
      const [archive] = Array.from(importInput.files || []);
      importInput.value = "";
      if (!this.agent || !archive) return;
      try {
        await this.agent.loadActor(archive);
        this.dashboard.setStatus("Policy Imported", "#0f0");
      } catch (e) {
        console.error("Policy import failed", e);
        this.dashboard.setStatus("Import Failed", "#ff4444");
      }
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
