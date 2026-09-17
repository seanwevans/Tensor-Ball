// The same shot, re-flown in cannon-es with the court in the way.
//
// physics.mjs integrates free flight. That is exactly right for a swish and
// exactly nothing for anything else, and "the ball never touches the rim" is a
// claim about geometry that the geometry module is the wrong thing to check it
// with. So this builds the app's world for real — PhysicsWorld's gravity,
// broadphase, solver and contact material, Court's floor, both hoops' rings,
// boards and posts, the scoring sensor — launches a ball through Ball.launch's
// own impulse, and steps it with applyAir in front of each step the way
// PhysicsWorld.step does.
//
// Nothing in here is shared with the solver. If a solution's reverse pass, its
// clearance sweep and its forward replica were all wrong in the same direction,
// this is what would notice.
//
// cannon-es is the one dependency in this folder and it is optional: without it
// `npm install` has not been run and every other tool here still works. The
// version is pinned to the app's own (see package.json).
import { actionToLaunch } from "./court.mjs";

export async function loadCannon() {
  try {
    return await import("cannon-es");
  } catch {
    return null;
  }
}

export async function makeCannonCourt(app) {
  const CANNON = await loadCannon();
  if (!CANNON) return null;
  const { CONFIG, PHYS, GEOM, STOP } = app;

  // PhysicsWorld.
  const world = new CANNON.World();
  world.gravity.set(0, PHYS.gravity, 0);
  world.broadphase = new CANNON.SAPBroadphase(world);
  world.solver.iterations = PHYS.solverIterations;
  world.allowSleep = true;
  const concrete = new CANNON.Material("concrete");
  const plastic = new CANNON.Material("plastic");
  world.addContactMaterial(
    new CANNON.ContactMaterial(concrete, plastic, {
      friction: PHYS.friction,
      restitution: PHYS.restitution
    })
  );

  const staticBody = (x, y, z, shape) => {
    const b = new CANNON.Body({ mass: 0, material: concrete });
    if (shape) b.addShape(shape);
    b.position.set(x, y, z);
    b.collisionFilterGroup = CONFIG.groups.court;
    b.collisionFilterMask = CONFIG.groups.ball | CONFIG.groups.court;
    return b;
  };

  // Court._buildFloor.
  const floor = new CANNON.Body({ mass: 0, material: concrete });
  floor.addShape(new CANNON.Plane());
  floor.quaternion.setFromAxisAngle(new CANNON.Vec3(1, 0, 0), -Math.PI / 2);
  floor.collisionFilterGroup = CONFIG.groups.court;
  floor.collisionFilterMask = CONFIG.groups.ball | CONFIG.groups.court;
  world.addBody(floor);

  // Court._buildHoop, both ends. Only the right-hand one is scored against —
  // the left is there because it is there in the app.
  const parts = {};
  for (const sign of [1, -1]) {
    const board = GEOM.board;
    const pole = GEOM.pole;
    const ring = GEOM.ring;

    const boardBody = staticBody(
      sign * board.centerX,
      board.centerY,
      board.centerZ,
      new CANNON.Box(new CANNON.Vec3(board.halfDepth, board.halfHeight, board.halfWidth))
    );
    world.addBody(boardBody);

    const poleBody = staticBody(
      sign * pole.centerX,
      pole.centerY,
      pole.centerZ,
      new CANNON.Box(new CANNON.Vec3(pole.halfDepth, pole.halfHeight, pole.halfWidth))
    );
    world.addBody(poleBody);

    const rimBody = staticBody(sign * ring.x, ring.y, ring.z, null);
    for (let i = 0; i < ring.segments; i++) {
      const a = (i / ring.segments) * Math.PI * 2;
      rimBody.addShape(
        new CANNON.Sphere(ring.tube),
        new CANNON.Vec3(Math.cos(a) * ring.radius, 0, Math.sin(a) * ring.radius)
      );
    }
    world.addBody(rimBody);

    const sensor = new CANNON.Body({
      mass: 0,
      isTrigger: true,
      shape: new CANNON.Cylinder(
        GEOM.sensor.radius,
        GEOM.sensor.radius,
        GEOM.sensor.halfHeight * 2,
        8
      )
    });
    sensor.position.set(sign * GEOM.sensor.x, GEOM.sensor.y, GEOM.sensor.z);
    sensor.collisionFilterGroup = CONFIG.groups.court;
    sensor.collisionFilterMask = CONFIG.groups.ball | CONFIG.groups.court;
    world.addBody(sensor);

    if (sign === 1) Object.assign(parts, { rimBody, boardBody, poleBody, sensor });
  }

  // The ball, built the way Ball's constructor builds one.
  const ball = new CANNON.Body({
    mass: PHYS.mass,
    shape: new CANNON.Sphere(CONFIG.ballRadius),
    material: plastic,
    linearDamping: PHYS.linearDamping,
    angularDamping: PHYS.angularDamping
  });
  ball.collisionFilterGroup = CONFIG.groups.ball;
  ball.collisionFilterMask = CONFIG.groups.court;
  world.addBody(ball);

  const rimVec = new CANNON.Vec3(CONFIG.rim.x, CONFIG.rim.y, CONFIG.rim.z);
  const touched = { rim: false, board: false, pole: false, floor: false, fromBelow: false };
  world.addEventListener("beginContact", ({ bodyA, bodyB }) => {
    const other = bodyA === ball ? bodyB : bodyB === ball ? bodyA : null;
    if (!other) return;
    if (other === parts.sensor) {
      // isEntryFromBelow, verbatim.
      if (ball.velocity.y <= CONFIG.hoopEntry.minAscentSpeed) return;
      const dx = ball.position.x - CONFIG.rim.x;
      const dz = ball.position.z - CONFIG.rim.z;
      const r = CONFIG.hoopEntry.columnRadius;
      if (dx * dx + dz * dz <= r * r) touched.fromBelow = true;
    } else if (other === parts.rimBody) touched.rim = true;
    else if (other === parts.boardBody) touched.board = true;
    else if (other === parts.poleBody) touched.pole = true;
    else if (other === floor) touched.floor = true;
  });

  const A = CONFIG.air;
  // applyAir, for the one body it acts on.
  function applyAir() {
    if (!A.enabled) return;
    const v = ball.velocity;
    // Math.sqrt of the sum of squares, not Math.hypot: applyAir writes it this
    // way and the two do not round identically.
    const speed = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (speed < 1e-4) return;
    const w = ball.angularVelocity;
    const m = ball.mass;
    ball.force.x += m * (-A.drag * speed * v.x + A.magnus * (w.y * v.z - w.z * v.y));
    ball.force.y += m * (-A.drag * speed * v.y + A.magnus * (w.z * v.x - w.x * v.z));
    ball.force.z += m * (-A.drag * speed * v.z + A.magnus * (w.x * v.y - w.y * v.x));
  }

  // Fly one shot. Takes the action rather than the velocity, so the launch
  // mapping is on trial here too.
  function fly(action, spawn) {
    for (const k of Object.keys(touched)) touched[k] = false;

    // Ball.spawn, then Ball.launch.
    ball.position.set(spawn.x, spawn.y, spawn.z);
    ball.velocity.set(0, 0, 0);
    ball.angularVelocity.set(0, 0, 0);
    ball.sleep();

    const L = actionToLaunch(action, spawn.x, spawn.z, CONFIG.rim, CONFIG.launch);
    ball.wakeUp();
    ball.applyImpulse(
      new CANNON.Vec3(L.vx * PHYS.mass, L.vy * PHYS.mass, L.vz * PHYS.mass),
      new CANNON.Vec3(0, 0, 0)
    );
    ball.angularVelocity.set(L.wx, L.wy, L.wz);

    // TrainingArena.update's state, per ball.
    let scored = false;
    let prevX = null;
    let prevY = 0;
    let prevZ = 0;
    let entryX = 0;
    let entryZ = 0;
    let entryOffset = Infinity;
    let minDist = Infinity;
    let frames = 0;
    let stop = "maxsteps";

    for (let n = 1; n <= STOP.timeout + 1; n++) {
      applyAir();
      world.step(PHYS.dt);
      frames = n;
      const p = ball.position;

      // trackHoopPass.
      if (prevX !== null && !scored && !touched.fromBelow) {
        if (prevY >= CONFIG.rim.y && p.y < CONFIG.rim.y) {
          const t = (prevY - CONFIG.rim.y) / (prevY - p.y);
          entryX = prevX + (p.x - prevX) * t - CONFIG.rim.x;
          entryZ = prevZ + (p.z - prevZ) * t - CONFIG.rim.z;
          entryOffset = Math.sqrt(entryX * entryX + entryZ * entryZ);
          if (entryX * entryX + entryZ * entryZ <= CONFIG.hoopEntry.scoreRadius ** 2)
            scored = true;
        }
      }
      prevX = p.x;
      prevY = p.y;
      prevZ = p.z;

      minDist = Math.min(minDist, p.distanceTo(rimVec));

      const stopped = ball.velocity.length() < STOP.stoppedSpeed && n > STOP.stoppedAfter;
      if (scored) { stop = "scored"; break; }
      if (p.y < STOP.floorY) { stop = "floor"; break; }
      if (Math.abs(p.x) > STOP.oobX || Math.abs(p.z) > STOP.oobZ) { stop = "oob"; break; }
      if (stopped) { stop = "stopped"; break; }
      if (n > STOP.timeout) { stop = "timeout"; break; }
    }

    return {
      scored,
      entryX,
      entryZ,
      entryOffset,
      minDist,
      frames,
      stop,
      hitRim: touched.rim,
      hitBackboard: touched.board,
      hitPole: touched.pole,
      enteredFromBelow: touched.fromBelow,
      // A clean swish: through the ring having touched nothing at all.
      clean: scored && !touched.rim && !touched.board && !touched.pole
    };
  }

  return { fly, world, ball, parts };
}
