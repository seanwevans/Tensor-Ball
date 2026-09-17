// Everything this folder claims, checked.
//
//   node selftest.mjs
//
// The claims are worth listing, because they are the reason to trust an answer
// that no training run produced:
//
//   1. script.js is still shaped the way appconfig.mjs reads it.
//   2. The forward stepper is not a model of the app's physics, it is the app's
//      physics — bit for bit against cannon-es over a whole flight.
//   3. The backward stepper is the forward stepper's exact inverse.
//   4. The action mapping round-trips.
//   5. Per-shot weather is refused rather than silently solved as if it were
//      not there.
import { fileURLToPath } from "node:url";
import { loadApp } from "./appconfig.mjs";
import { state, copyState, roll, stepForward } from "./physics.mjs";
import { actionToLaunch, launchToAction } from "./court.mjs";
import { makeSolver } from "./harness.mjs";
import { loadCannon } from "./cannoncheck.mjs";

let passed = 0;
let failed = 0;

function check(name, ok, detail = "") {
  if (ok) {
    passed++;
    process.stdout.write(`  ok   ${name}${detail ? `  (${detail})` : ""}\n`);
  } else {
    failed++;
    process.stdout.write(`  FAIL ${name}${detail ? `  (${detail})` : ""}\n`);
  }
}

async function main() {
  // cannon-es is optional, so the cross-checks that need it are asked for only
  // when it is there and skipped by name when it is not.
  const haveCannon = !!(await loadCannon());
  const S = await makeSolver({ seed: 11, withCannon: haveCannon });
  const { app, P, rand } = S;
  const rim = app.CONFIG.rim;

  process.stdout.write("\nconfig, read out of script.js\n");
  check("gravity", app.PHYS.gravity < 0 && Math.abs(app.PHYS.gravity) > 30, `${app.PHYS.gravity} ft/s^2`);
  check("timestep", Math.abs(app.PHYS.dt - 1 / 60) < 1e-12, `${app.PHYS.dt}`);
  check("ball", app.PHYS.mass > 0 && app.CONFIG.ballRadius > 0, `m=${app.PHYS.mass} r=${app.CONFIG.ballRadius}`);
  check(
    "ring is where CONFIG.rim says",
    Math.abs(app.GEOM.ring.x - rim.x) < 1e-12 && Math.abs(app.GEOM.ring.y - rim.y) < 1e-12
  );
  check(
    "score radius is the hole minus the ball",
    Math.abs(
      app.CONFIG.hoopEntry.scoreRadius - (app.GEOM.ring.radius - app.GEOM.ring.tube - app.CONFIG.ballRadius)
    ) < 1e-9,
    `${app.CONFIG.hoopEntry.scoreRadius} ft`
  );

  process.stdout.write("\nreversibility\n");
  {
    // Each direction is tested from the end it starts at. Forward-then-backward
    // begins at a launch; backward-then-forward begins at a ball dropping
    // through the ring, which is what a reverse pass is actually handed.
    const launch = state(18.5, 6.4, -7.25, 17.3, 26.9, 4.1, 1.7, 0, -14.2);
    const entry = state(41.75, 10, 0, -14.9, -21.4, 2.6, -1.1, 0, 13.8);
    let worstFB = 0;
    let worstBF = 0;
    // Horizons a shot actually spans: a flight from the launch to the ring is
    // sixty to a hundred and sixty steps.
    for (const n of [1, 10, 90, 160]) {
      const a = copyState(launch);
      roll(a, P, n, false);
      roll(a, P, n, true);
      const b = copyState(entry);
      roll(b, P, n, true);
      roll(b, P, n, false);
      for (const k of ["px", "py", "pz", "vx", "vy", "vz", "wx", "wy", "wz"]) {
        worstFB = Math.max(worstFB, Math.abs(a[k] - launch[k]));
        worstBF = Math.max(worstBF, Math.abs(b[k] - entry[k]));
      }
    }
    check("forward then backward returns the launch", worstFB < 1e-9, `worst ${worstFB.toExponential(2)}`);
    check("backward then forward returns the basket", worstBF < 1e-9, `worst ${worstBF.toExponential(2)}`);

    // And past that horizon it says so rather than inventing a number. Drag run
    // backwards is a growth term, so a reverse pass taken far past the launch
    // leaves the set of states any forward step can have produced, and the
    // implicit solve has a second root out there to land on.
    let named = false;
    try {
      roll(copyState(entry), P, 600, true);
    } catch (e) {
      named = /no physical preimage/.test(e.message);
    }
    check("a reverse pass run far past its launch fails loudly", named);
  }

  process.stdout.write("\nthe stepper is cannon-es, not a model of it\n");
  const CANNON = haveCannon ? await loadCannon() : null;
  if (!CANNON) {
    process.stdout.write("  skip cannon-es cross-checks — run `npm install` in tools/inverse\n");
  } else {
    const world = new CANNON.World();
    world.gravity.set(0, app.PHYS.gravity, 0);
    world.broadphase = new CANNON.SAPBroadphase(world);
    world.solver.iterations = app.PHYS.solverIterations;
    world.allowSleep = true;
    const body = new CANNON.Body({
      mass: app.PHYS.mass,
      shape: new CANNON.Sphere(app.CONFIG.ballRadius),
      linearDamping: app.PHYS.linearDamping,
      angularDamping: app.PHYS.angularDamping
    });
    world.addBody(body);
    const A = app.CONFIG.air;
    const s = state(9, 6.2, 11, 19, 30, -6, 0.5, 0, -17);
    body.position.set(s.px, s.py, s.pz);
    body.velocity.set(s.vx, s.vy, s.vz);
    body.angularVelocity.set(s.wx, s.wy, s.wz);
    let worst = 0;
    for (let i = 0; i < 200; i++) {
      if (A.enabled) {
        const v = body.velocity;
        const w = body.angularVelocity;
        const sp = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (sp >= 1e-4) {
          const m = body.mass;
          body.force.x += m * (-A.drag * sp * v.x + A.magnus * (w.y * v.z - w.z * v.y));
          body.force.y += m * (-A.drag * sp * v.y + A.magnus * (w.z * v.x - w.x * v.z));
          body.force.z += m * (-A.drag * sp * v.z + A.magnus * (w.x * v.y - w.y * v.x));
        }
      }
      world.step(app.PHYS.dt);
      stepForward(s, P);
      worst = Math.max(
        worst,
        Math.abs(body.position.x - s.px),
        Math.abs(body.position.y - s.py),
        Math.abs(body.position.z - s.pz),
        Math.abs(body.velocity.x - s.vx),
        Math.abs(body.velocity.y - s.vy),
        Math.abs(body.velocity.z - s.vz)
      );
    }
    check("200 steps of free flight, bit for bit", worst === 0, `worst ${worst}`);
  }

  process.stdout.write("\nthe action mapping round-trips\n");
  {
    let worst = 0;
    for (let i = 0; i < 500; i++) {
      const spawn = { x: rand() * 40, y: 5.4 + rand(), z: (rand() - 0.5) * 44 };
      if (Math.hypot(spawn.x - rim.x, spawn.z - rim.z) < app.CONFIG.minSpawnDistance) continue;
      const a = [rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1];
      const v = actionToLaunch(a, spawn.x, spawn.z, rim, app.CONFIG.launch);
      const back = launchToAction(v, spawn.x, spawn.z, rim, app.CONFIG.launch);
      for (let k = 0; k < 4; k++) worst = Math.max(worst, Math.abs(back.action[k] - a[k]));
      worst = Math.max(worst, back.spinResidual);
    }
    check("action -> velocity -> action", worst < 1e-12, `worst ${worst.toExponential(2)}`);
  }

  process.stdout.write("\nrefusals\n");
  {
    let refused = false;
    try {
      await makeSolver({ overlay: { air: { wind: 4 } } });
    } catch (e) {
      refused = /no exact inverse/.test(e.message);
    }
    check("per-shot weather is refused", refused);
    let anchored = false;
    try {
      await loadApp({ scriptPath: fileURLToPath(import.meta.url) });
    } catch (e) {
      anchored = /anchor not found/.test(e.message);
    }
    check("a script.js that is not script.js is refused", anchored);
  }

  process.stdout.write(`\n${passed} passed, ${failed} failed\n`);
  process.exitCode = failed ? 1 : 0;
}

await main();
