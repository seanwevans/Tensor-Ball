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
//   5. A reverse-pass solution, flown forward in cannon-es with the rings and
//      boards in the world, goes in — cleanly, through the middle, having
//      touched nothing.
//   6. A targeted solution does too, and its crossing is laterally dead centre
//      to the last bit, which is the claim that lets the solve be
//      one-dimensional.
//   7. The reported tolerance is the real edge of the make set: just inside it
//      the shot drops, just outside it does not.
//   8. Per-shot weather is refused rather than silently solved as if it were
//      not there.
import { fileURLToPath } from "node:url";
import { loadApp } from "./appconfig.mjs";
import { state, copyState, roll, stepForward } from "./physics.mjs";
import { actionToLaunch, launchToAction, launchFrame } from "./court.mjs";
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
  const { app, P, flight, reverse, target, rand, cannon } = S;
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

  process.stdout.write("\nreverse pass: made position in, shot out\n");
  {
    let worstClosure = 0;
    let n = 0;
    let scored = 0;
    let clean = 0;
    let tries = 0;
    while (n < 120 && tries < 6000) {
      tries++;
      const r = reverse.solve({ rand });
      if (!r) continue;
      n++;
      worstClosure = Math.max(worstClosure, r.verified.closure);
      if (cannon) {
        const v = cannon.fly(r.action, r.spawn);
        if (v.scored) scored++;
        if (v.clean) clean++;
      }
    }
    check("solutions were produced", n >= 100, `${n} of ${tries} made positions`);
    check(
      "the forward flight lands on the made position it came from",
      worstClosure < 1e-10,
      `worst ${worstClosure.toExponential(2)} ft`
    );
    if (cannon) {
      check("cannon-es scores every one", scored === n, `${scored}/${n}`);
      check("every one is a clean swish", clean === n, `${clean}/${n}`);
    }
  }

  process.stdout.write("\ntargeted solve: spawn in, exact action out\n");
  {
    let worstLateral = 0;
    let reachable = 0;
    let scored = 0;
    let clean = 0;
    let inEnvelope = 0;
    const spawns = [];
    for (let d = 3; d <= 42; d += 4.875)
      for (const side of [0, 0.6, -0.9])
        spawns.push({
          x: rim.x - d * Math.cos(side),
          y: app.CONFIG.spawnHeight.mean,
          z: rim.z + d * Math.sin(side)
        });
    const usable = spawns.filter(
      (s) => s.x >= 0 && s.x <= app.GEOM.spawnBounds.maxX && Math.abs(s.z) <= app.GEOM.spawnBounds.maxAbsZ
    );
    for (const spawn of usable) {
      const r = target.solve(spawn, { upSteps: 24, spinValues: [0, 1], refine: 3 });
      if (!r.reachable) continue;
      reachable++;
      worstLateral = Math.max(worstLateral, Math.abs(r.lateral));
      if (r.action.every((a) => Math.abs(a) <= 1)) inEnvelope++;
      if (cannon) {
        const v = cannon.fly(r.action, spawn);
        if (v.scored) scored++;
        if (v.clean) clean++;
      }
    }
    check("every spawn on the court has a make", reachable === usable.length, `${reachable}/${usable.length}`);
    check("the answer is inside the launch envelope", inEnvelope === reachable, `${inEnvelope}/${reachable}`);
    check(
      "side = 0 puts the crossing laterally dead centre",
      worstLateral < 1e-12,
      `worst ${worstLateral.toExponential(2)} ft`
    );
    if (cannon) {
      check("cannon-es scores every one", scored === reachable, `${scored}/${reachable}`);
      check("every one is a clean swish", clean === reachable, `${clean}/${reachable}`);
    }
  }

  process.stdout.write("\nthe reported tolerance is the edge of the make set\n");
  {
    const spawn = { x: rim.x - 17, y: app.CONFIG.spawnHeight.mean, z: 4 };
    const r = target.solve(spawn, { upSteps: 32, spinValues: [0] });
    const frame = launchFrame(spawn.x, spawn.z, rim);
    const fly = (action) => {
      const v = actionToLaunch(action, spawn.x, spawn.z, rim, app.CONFIG.launch, frame);
      const st = state(spawn.x, spawn.y, spawn.z, v.vx, v.vy, v.vz, v.wx, v.wy, v.wz);
      const out = flight.simulate(st);
      return out.scored && !out.contact;
    };
    let insideOk = true;
    let outsideOk = true;
    for (const [ch, side, sign] of [
      [0, "plus", 1],
      [0, "minus", -1],
      [1, "plus", 1],
      [1, "minus", -1],
      [2, "plus", 1],
      [2, "minus", -1]
    ]) {
      const t = r.tolerance[["fwd", "up", "side", "spin"][ch]][side];
      if (!(t > 0)) continue;
      const inside = r.action.slice();
      inside[ch] += sign * t * 0.98;
      const outside = r.action.slice();
      outside[ch] += sign * t * 1.05;
      if (!fly(inside)) insideOk = false;
      // A channel whose tolerance runs into the envelope rather than into a
      // miss has nothing outside it to test.
      if (Math.abs(outside[ch]) <= 1 && fly(outside)) outsideOk = false;
    }
    check("just inside the tolerance the shot still drops", insideOk);
    check("just outside it the shot does not", outsideOk);
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
