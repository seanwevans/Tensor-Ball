// The court as the solver has to see it: the map between an action and a launch
// velocity, and how close a trajectory comes to the bodies that would deflect
// it.
//
// Both are the app's own. The action mapping is Ball.launch read backwards; the
// bodies are the ones Court._buildHoop adds to the world, at the radii
// appconfig.mjs pulled out of it — the ring as cannon sees it (sixteen spheres
// on a circle, not a torus), the backboard slab, the post, the floor.

// Ball.launch's frame: forward is the horizontal direction from the shot to the
// ring, side is UP x forward. Returned as two unit vectors in the ground plane.
export function launchFrame(sx, sz, rim) {
  let fx = rim.x - sx;
  let fz = rim.z - sz;
  const len = Math.hypot(fx, fz);
  if (len < 1e-9)
    throw new Error("launchFrame: the shot is on the ring's axis, so there is no forward");
  fx /= len;
  fz /= len;
  // UP x (fx, 0, fz) = (fz, 0, -fx).
  return { fx, fz, sxx: fz, szz: -fx, distance: len };
}

// action -> launch velocity and angular velocity, exactly as Ball.launch builds
// them. The impulse is applied at the body's centre with the ball at rest, so
// impulse/mass is the launch velocity outright and the mass cancels.
export function actionToLaunch(action, sx, sz, rim, L, frame = null) {
  const f = frame || launchFrame(sx, sz, rim);
  const lerp = (lo, hi, a) => lo + ((a + 1) / 2) * (hi - lo);
  const vFwd = lerp(L.fwdMin, L.fwdMax, action[0]);
  const vUp = lerp(L.upMin, L.upMax, action[1]);
  const vSide = action[2] * L.side;
  const spin = action[3] * L.spin;
  return {
    vx: f.fx * vFwd + f.sxx * vSide,
    vy: vUp,
    vz: f.fz * vFwd + f.szz * vSide,
    // Ball.launch sets the angular velocity to -spin about dirSide.
    wx: -f.sxx * spin,
    wy: 0,
    wz: -f.szz * spin,
    vFwd,
    vUp,
    vSide,
    spin,
    frame: f
  };
}

// Launch velocity -> the action that produces it. The inverse of the above:
// project onto the frame, then undo the lerp.
//
// `inEnvelope` is the question the reverse pass exists to ask. Every velocity
// is representable in the frame — forward, up and side span the space — so the
// only thing that can make a shot unavailable to the agent is a channel outside
// [-1, 1]. tanh never reaches the ends, so `margin` (how much room the tightest
// channel has left) is the honest reading of how comfortably the policy could
// have produced this.
export function launchToAction(v, sx, sz, rim, L, frame = null) {
  const f = frame || launchFrame(sx, sz, rim);
  const unlerp = (lo, hi, x) => (2 * (x - lo)) / (hi - lo) - 1;
  const vFwd = v.vx * f.fx + v.vz * f.fz;
  const vSide = v.vx * f.sxx + v.vz * f.szz;
  // w = -spin * dirSide, so spin = -(w . dirSide).
  const spin = -(v.wx * f.sxx + v.wz * f.szz);
  // Anything out of that plane is not a spin Ball.launch can produce.
  const spinResidual = Math.hypot(v.wy, v.wx + spin * f.sxx, v.wz + spin * f.szz);
  const action = [
    unlerp(L.fwdMin, L.fwdMax, vFwd),
    unlerp(L.upMin, L.upMax, v.vy),
    L.side === 0 ? 0 : vSide / L.side,
    L.spin === 0 ? 0 : spin / L.spin
  ];
  let worst = 0;
  for (const a of action) worst = Math.max(worst, Math.abs(a));
  return {
    action,
    vFwd,
    vUp: v.vy,
    vSide,
    spin,
    spinResidual,
    inEnvelope: worst <= 1,
    margin: 1 - worst,
    frame: f
  };
}

// The launch spots Ball.spawn will actually produce: inside the court box it
// rejects against, and outside the column under the ring where every upward
// shot is an illegal entry.
export function spawnLegal(x, z, { CONFIG, GEOM }) {
  const b = GEOM.spawnBounds;
  if (!(x >= 0 && x <= b.maxX && Math.abs(z) <= b.maxAbsZ)) return false;
  const d = Math.hypot(x - CONFIG.rim.x, z - CONFIG.rim.z);
  return d >= CONFIG.minSpawnDistance;
}

// Distance from a point to the union of the bodies a shot can hit on the way
// in, minus the ball's radius: the gap between the ball's surface and the
// nearest thing that would deflect it. Negative means cannon would have found
// an overlap.
export function makeClearance({ CONFIG, GEOM }) {
  const R = CONFIG.ballRadius;
  const ring = GEOM.ring;
  const boxes = [GEOM.board, GEOM.pole];

  // Sphere centres of the ring body, precomputed.
  const rcx = new Float64Array(ring.segments);
  const rcz = new Float64Array(ring.segments);
  for (let i = 0; i < ring.segments; i++) {
    const a = (i / ring.segments) * Math.PI * 2;
    rcx[i] = ring.x + Math.cos(a) * ring.radius;
    rcz[i] = ring.z + Math.sin(a) * ring.radius;
  }

  // One bound covering ring, board and post, so a ball out over the court pays
  // for a box test rather than sixteen sphere tests. The distance to this box
  // is a lower bound on the distance to anything inside it, which is all the
  // sweep below needs.
  let lo = [ring.x - ring.radius - ring.tube, ring.y - ring.tube, ring.z - ring.radius - ring.tube];
  let hi = [ring.x + ring.radius + ring.tube, ring.y + ring.tube, ring.z + ring.radius + ring.tube];
  for (const b of boxes) {
    lo = [
      Math.min(lo[0], b.centerX - b.halfDepth),
      Math.min(lo[1], b.centerY - b.halfHeight),
      Math.min(lo[2], b.centerZ - b.halfWidth)
    ];
    hi = [
      Math.max(hi[0], b.centerX + b.halfDepth),
      Math.max(hi[1], b.centerY + b.halfHeight),
      Math.max(hi[2], b.centerZ + b.halfWidth)
    ];
  }

  const boxDist = (px, py, pz, cx, cy, cz, hx, hy, hz) => {
    const qx = Math.abs(px - cx) - hx;
    const qy = Math.abs(py - cy) - hy;
    const qz = Math.abs(pz - cz) - hz;
    const ox = Math.max(qx, 0);
    const oy = Math.max(qy, 0);
    const oz = Math.max(qz, 0);
    return (
      Math.hypot(ox, oy, oz) + Math.min(Math.max(qx, Math.max(qy, qz)), 0)
    );
  };

  const regionCx = (lo[0] + hi[0]) / 2;
  const regionCy = (lo[1] + hi[1]) / 2;
  const regionCz = (lo[2] + hi[2]) / 2;
  const regionHx = (hi[0] - lo[0]) / 2;
  const regionHy = (hi[1] - lo[1]) / 2;
  const regionHz = (hi[2] - lo[2]) / 2;

  // The gap at a point. `floor` is the infinite ground plane the court sits on.
  function gap(px, py, pz) {
    let d = py - GEOM.floorY - R;
    const near = boxDist(px, py, pz, regionCx, regionCy, regionCz, regionHx, regionHy, regionHz);
    if (near - R >= d) return d;
    for (const b of boxes) {
      const bd = boxDist(px, py, pz, b.centerX, b.centerY, b.centerZ, b.halfDepth, b.halfHeight, b.halfWidth) - R;
      if (bd < d) d = bd;
    }
    const dy = py - ring.y;
    for (let i = 0; i < rcx.length; i++) {
      const dx = px - rcx[i];
      const dz = pz - rcz[i];
      const sd = Math.sqrt(dx * dx + dy * dy + dz * dz) - ring.tube - R;
      if (sd < d) d = sd;
    }
    return d;
  }

  // The smallest gap anywhere along the segment between two frames, found by
  // conservative advancement: the gap is 1-Lipschitz in position, so from a
  // point with gap g nothing within g of it can be touching, and the walk can
  // skip that far. When it terminates without a sample at or below `graze`, no
  // contact anywhere on the segment has been *proved*, not sampled for.
  //
  // This is strictly stronger than what the app runs into. cannon only tests
  // the frame positions themselves, so a trajectory that clears continuously
  // clears cannon's discrete test as well, whatever the ball's speed.
  function sweep(ax, ay, az, bx, by, bz, graze = 1e-3) {
    const dx = bx - ax;
    const dy = by - ay;
    const dz = bz - az;
    const len = Math.hypot(dx, dy, dz);
    let worst = Infinity;
    if (len < 1e-12) {
      worst = gap(ax, ay, az);
      return { gap: worst, contact: worst <= graze };
    }
    let t = 0;
    for (let guard = 0; guard < 4096; guard++) {
      const g = gap(ax + dx * t, ay + dy * t, az + dz * t);
      if (g < worst) worst = g;
      if (g <= graze) return { gap: worst, contact: true };
      if (t >= 1) break;
      t = Math.min(1, t + g / len);
    }
    return { gap: worst, contact: false };
  }

  return { gap, sweep, ballRadius: R };
}
