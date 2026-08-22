// Runs in the page before any app code (Playwright addInitScript). Everything
// here is either measurement or headless plumbing — nothing in it changes what
// the agent learns:
//
//   * Math.random is replaced with a seeded PRNG so two configs see the same
//     spawn positions, the same exploration draws and the same weight init.
//     Config-to-config differences then come from the config, not from luck.
//   * requestAnimationFrame is unhooked from vsync. The physics step is a fixed
//     1/60 either way; on the display clock a 600-frame shot costs ten wall
//     seconds of waiting, which would dwarf everything the search is measuring.
//   * The main-canvas render, the trajectory trails and the dashboard canvases
//     are stubbed out. Under SwiftShader they are pure overhead, and none of
//     them feed the agent — the stereo capture is untouched and still renders
//     the real scene.
(() => {
  const P = window.__HP_PARAMS;
  window.__HP_CONFIG = P.config;

  // mulberry32 — small, fast, and identical across runs for a given seed.
  let s = P.seed >>> 0;
  Math.random = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  // Drain rAF through a message port instead of the frame clock.
  const queue = [];
  const chan = new MessageChannel();
  chan.port1.onmessage = () => {
    const cb = queue.shift();
    if (cb) cb(performance.now());
  };
  window.requestAnimationFrame = (cb) => {
    queue.push(cb);
    chan.port2.postMessage(0);
    return queue.length;
  };

  const H = {
    batches: [],
    maxBatches: P.batches,
    // Optional wall-clock cap. Configs that change how much compute a batch
    // costs (critic epochs, actor minibatch) can only be compared honestly on
    // a clock, not on a batch count.
    budgetMs: P.budgetMs || 0,
    cur: null,
    done: false,
    t0: 0,
    tPrev: 0,
    error: null,
    backend: null
  };
  window.__HP = H;

  const quantile = (sorted, q) =>
    sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] : null;

  window.__hpInstall = (TB) => {
    if (P.backend) {
      const setBackend = TB.tf.setBackend.bind(TB.tf);
      TB.tf.setBackend = () => setBackend(P.backend);
    }

    TB.SceneManager.prototype.render = function () {};
    TB.TrajectoryTrails.prototype.add = function () {};
    for (const m of ["drawLoss", "drawAgentView", "visualizeKernels", "visualizeActivations"])
      TB.Dashboard.prototype[m] = function () {};

    const reward = TB.TrainingArena.prototype._reward;
    TB.TrainingArena.prototype._reward = function (ball) {
      const r = reward.call(this, ball);
      const c = H.cur;
      if (c) {
        c.n++;
        c.rewardSum += r.reward;
        c.types[r.type] = (c.types[r.type] || 0) + 1;
        c.dists.push(ball.minDist);
      }
      return r;
    };

    const pushLoss = TB.Dashboard.prototype.pushLoss;
    TB.Dashboard.prototype.pushLoss = function (l) {
      if (H.cur) H.cur.loss = l;
      return pushLoss.call(this, l);
    };

    const finish = TB.TrainingArena.prototype.finishBatch;
    TB.TrainingArena.prototype.finishBatch = async function (mesh) {
      if (H.done) return;
      const tStart = performance.now();
      H.cur = { n: 0, rewardSum: 0, types: {}, dists: [], loss: null };
      const noise = this.exploreNoise;
      try {
        await finish.call(this, mesh);
      } catch (e) {
        H.error = String(e && e.stack ? e.stack : e);
        H.done = true;
        return;
      }
      const c = H.cur;
      H.cur = null;
      const now = performance.now();
      const dists = c.dists.sort((a, b) => a - b);
      H.batches.push({
        batch: H.batches.length + 1,
        n: c.n,
        // Fraction of the batch that dropped through the rim from above.
        acc: (c.types.score || 0) / c.n,
        illegal: (c.types.illegal || 0) / c.n,
        rim: (c.types.rim || 0) / c.n,
        meanReward: c.rewardSum / c.n,
        // Closest approach to the rim, the dense signal that moves long before
        // any ball goes in.
        distP10: quantile(dists, 0.1),
        distMed: quantile(dists, 0.5),
        loss: c.loss,
        noise,
        // How far out the batch was spawned. Accuracy is only comparable
        // between configs at the same radius, so a curriculum run has to
        // report it alongside.
        spawnRadius: this.spawnRadius,
        // Simulation of this batch's shots (between the previous finishBatch
        // and this one) vs. reward + train + next capture inside it.
        simMs: tStart - H.tPrev,
        stepMs: now - tStart,
        elapsedMs: now - H.t0
      });
      H.tPrev = now;
      if (
        H.batches.length >= H.maxBatches ||
        (H.budgetMs && now - H.t0 >= H.budgetMs)
      ) {
        H.done = true;
        window.__APP.trainingMode = false;
        this.halt();
      }
    };
  };

  window.__hpStart = async () => {
    const app = window.__APP;
    H.backend = TBBackend();
    H.t0 = performance.now();
    H.tPrev = H.t0;
    document.getElementById("toggle-train").click();
  };

  function TBBackend() {
    try {
      return window.tf ? window.tf.getBackend() : null;
    } catch (e) {
      return null;
    }
  }
})();
