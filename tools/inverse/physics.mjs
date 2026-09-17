// One flight step of the app's physics, and its exact inverse.
//
// Between the launch and the rim a shot touches nothing, so everything that
// happens to it is three lines of arithmetic. Written out in the order
// cannon-es and PhysicsWorld.step actually apply them:
//
//   applyAir(bodies)                      <- PhysicsWorld.step, before the step
//   force += m*g                          <- World.internalStep, "Add gravity"
//   v *= (1 - linearDamping)^dt           <- World.internalStep, "Apply damping"
//   w *= (1 - angularDamping)^dt
//   v += force/m * dt                     <- Body.integrate, "leap frog"
//   p += v * dt
//   force = 0                             <- World.clearForces
//
// which collapses to
//
//   a = airAccel(v_n, w_n)                (the air reads the velocity the body
//   v_{n+1} = Ld*v_n + (g + a)*dt          had *before* damping, because
//   w_{n+1} = Ad*w_n                       applyAir runs before world.step)
//   p_{n+1} = p_n + v_{n+1}*dt            (semi-implicit: the NEW velocity)
//
// Reversing it is not "run it with -dt". Drag is dissipative, so the flow is
// not symmetric in time; what is true is that the discrete map above is a
// bijection, and it can be inverted exactly:
//
//   p_n = p_{n+1} - v_{n+1}*dt            (trivial, the position is explicit)
//   w_n = w_{n+1} / Ad                    (trivial, the spin decay is a scale)
//   v_n : solve Ld*v + (g + airAccel(v, w_n))*dt = v_{n+1}
//
// The last line is implicit in v_n — airAccel depends on the velocity we are
// solving for — so it is a 3x3 root find, done here by Newton against the
// analytic Jacobian. It converges in two or three iterations because the air
// contributes about 0.1ft/s of the ~30ft/s being solved for, and it converges
// to machine precision, which is what makes a reverse pass *exact* rather than
// approximate: step backward n times and forward n times and the state comes
// back to ~1e-13 of where it started. selftest.mjs asserts exactly that.
//
// Nothing in here allocates. A ceiling sweep runs tens of millions of these.

// Everything the stepper needs, precomputed once from the app's config.
export function makeStepper({ CONFIG, PHYS }) {
  const A = CONFIG.air;
  const dt = PHYS.dt;
  return {
    // Where the forward step stops being injective, and so where a reverse pass
    // stops having an answer to give.
    //
    // Reversing a dissipative system is stable in one direction only. Drag takes
    // energy out going forwards, so it puts energy in going backwards, and a
    // reverse pass run past the launch keeps accelerating: the speed roughly
    // doubles every twenty steps once it is out past 100ft/s. It is not merely
    // inaccurate out there — it is answering a different question, because
    // v -> v - k|v|v*dt folds back on itself at |v| = 1/(2*k*dt) and every
    // target velocity past its peak has either two preimages or none. The
    // physical one is always the slower.
    //
    // A real shot is sixty to a hundred and sixty steps and spends all of them
    // three orders of magnitude inside this, so it never comes up. It is here so
    // that a caller who walks off the end gets an error naming the reason rather
    // than a plausible number off the wrong branch. See the README.
    branchSpeed: A.enabled && A.drag > 0 ? 1 / (2 * A.drag * dt) : Infinity,
    dt,
    gy: PHYS.gravity,
    // (1 - damping)^dt, exactly as World.internalStep computes it. Ld is 1 at
    // the app's linearDamping of 0 and is carried anyway so that a config
    // overlay turning it back on does not silently go unmodelled.
    Ld: Math.pow(1 - PHYS.linearDamping, dt),
    Ad: Math.pow(1 - PHYS.angularDamping, dt),
    air: {
      enabled: !!A.enabled,
      drag: A.drag,
      magnus: A.magnus,
      // The gym's own air. Per-shot weather (CONFIG.air.wind / jitter) is
      // refused up front by the CLIs: a shot whose air is drawn after the
      // action is chosen has no exact inverse, because the thing being
      // inverted is not a function of the action alone.
      windX: 0,
      windZ: 0
    },
    mass: PHYS.mass
  };
}

export function state(px, py, pz, vx, vy, vz, wx = 0, wy = 0, wz = 0) {
  return { px, py, pz, vx, vy, vz, wx, wy, wz };
}

export function copyState(s, into = {}) {
  into.px = s.px; into.py = s.py; into.pz = s.pz;
  into.vx = s.vx; into.vy = s.vy; into.vz = s.vz;
  into.wx = s.wx; into.wy = s.wy; into.wz = s.wz;
  return into;
}

// applyAir, line for line: a = -k|vr|vr + C(w x vr) against the relative
// airspeed. Written into `out` so the inner loop never allocates.
const _a = { x: 0, y: 0, z: 0 };
export function airAccel(vx, vy, vz, wx, wy, wz, air, out = _a) {
  out.x = out.y = out.z = 0;
  if (!air.enabled) return out;
  const rx = vx - air.windX;
  const ry = vy;
  const rz = vz - air.windZ;
  const speed = Math.sqrt(rx * rx + ry * ry + rz * rz);
  // applyAir's own guard. Below it the body gets no air force at all.
  if (speed < 1e-4) return out;
  const k = air.drag;
  const c = air.magnus;
  out.x = -k * speed * rx + c * (wy * rz - wz * ry);
  out.y = -k * speed * ry + c * (wz * rx - wx * rz);
  out.z = -k * speed * rz + c * (wx * ry - wy * rx);
  return out;
}

// One step of free flight, in place.
export function stepForward(s, P) {
  const a = airAccel(s.vx, s.vy, s.vz, s.wx, s.wy, s.wz, P.air);
  const dt = P.dt;
  s.vx = P.Ld * s.vx + a.x * dt;
  s.vy = P.Ld * s.vy + (P.gy + a.y) * dt;
  s.vz = P.Ld * s.vz + a.z * dt;
  s.wx *= P.Ad;
  s.wy *= P.Ad;
  s.wz *= P.Ad;
  s.px += s.vx * dt;
  s.py += s.vy * dt;
  s.pz += s.vz * dt;
  return s;
}

// Solve a 3x3 system by Cramer's rule, into [out0, out1, out2]. Returns false
// on a singular matrix, which for this Jacobian would mean the air model had
// been configured into something degenerate.
const _sol = new Float64Array(3);
function solve3(m, b, out = _sol) {
  const [a11, a12, a13, a21, a22, a23, a31, a32, a33] = m;
  const c11 = a22 * a33 - a23 * a32;
  const c12 = a23 * a31 - a21 * a33;
  const c13 = a21 * a32 - a22 * a31;
  const det = a11 * c11 + a12 * c12 + a13 * c13;
  if (!det) return null;
  const inv = 1 / det;
  out[0] = inv * (b[0] * c11 + b[1] * (a13 * a32 - a12 * a33) + b[2] * (a12 * a23 - a13 * a22));
  out[1] = inv * (b[0] * c12 + b[1] * (a11 * a33 - a13 * a31) + b[2] * (a13 * a21 - a11 * a23));
  out[2] = inv * (b[0] * c13 + b[1] * (a12 * a31 - a11 * a32) + b[2] * (a11 * a22 - a12 * a21));
  return out;
}

const _J = new Float64Array(9);
const _F = new Float64Array(3);

// The velocity a step must have started from to arrive at (vx, vy, vz) with
// angular velocity (wx, wy, wz) in hand. Newton on
//   F(v) = Ld*v + (g + a(v, w))*dt - v_target,
//   J    = Ld*I + dt * da/dv,
//   da/dv = -k(|r| I + r r^T/|r|) + C [w]_x
// Writes the answer back over the velocity fields of `s`.
export function solvePrevVelocity(s, P, maxIter = 24) {
  const dt = P.dt;
  const air = P.air;
  const tx = s.vx;
  const ty = s.vy;
  const tz = s.vz;
  // Seed with the air-free inverse, which is already within ~0.1ft/s.
  let vx = (tx) / P.Ld;
  let vy = (ty - P.gy * dt) / P.Ld;
  let vz = (tz) / P.Ld;

  for (let iter = 0; iter < maxIter; iter++) {
    const a = airAccel(vx, vy, vz, s.wx, s.wy, s.wz, air);
    _F[0] = P.Ld * vx + a.x * dt - tx;
    _F[1] = P.Ld * vy + (P.gy + a.y) * dt - ty;
    _F[2] = P.Ld * vz + a.z * dt - tz;
    if (Math.abs(_F[0]) + Math.abs(_F[1]) + Math.abs(_F[2]) < 1e-15) break;

    const rx = vx - air.windX;
    const ry = vy;
    const rz = vz - air.windZ;
    const speed = Math.sqrt(rx * rx + ry * ry + rz * rz);
    const live = air.enabled && speed >= 1e-4;
    const k = live ? air.drag : 0;
    const c = live ? air.magnus : 0;
    const invS = live ? 1 / speed : 0;

    _J[0] = P.Ld + dt * (-k * (speed + rx * rx * invS));
    _J[1] = dt * (-k * rx * ry * invS - c * s.wz);
    _J[2] = dt * (-k * rx * rz * invS + c * s.wy);
    _J[3] = dt * (-k * ry * rx * invS + c * s.wz);
    _J[4] = P.Ld + dt * (-k * (speed + ry * ry * invS));
    _J[5] = dt * (-k * ry * rz * invS - c * s.wx);
    _J[6] = dt * (-k * rz * rx * invS - c * s.wy);
    _J[7] = dt * (-k * rz * ry * invS + c * s.wx);
    _J[8] = P.Ld + dt * (-k * (speed + rz * rz * invS));

    const d = solve3(_J, _F);
    if (!d) throw new Error("solvePrevVelocity: singular Jacobian");
    vx -= d[0];
    vy -= d[1];
    vz -= d[2];
  }
  const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
  if (!Number.isFinite(speed) || speed > P.branchSpeed)
    throw new Error(
      `solvePrevVelocity: no physical preimage — the reverse pass reached ${speed.toFixed(0)}ft/s, ` +
        `past the ${P.branchSpeed.toFixed(0)}ft/s fold in the forward step. A reverse pass is ` +
        `stable over a shot and divergent well past one; this one was run past its launch.`
    );
  s.vx = vx;
  s.vy = vy;
  s.vz = vz;
  return s;
}

// One step of free flight, backwards: the exact inverse of stepForward.
export function stepBackward(s, P) {
  const dt = P.dt;
  // p_{n+1} = p_n + v_{n+1}*dt, and v_{n+1} is what we are holding.
  s.px -= s.vx * dt;
  s.py -= s.vy * dt;
  s.pz -= s.vz * dt;
  // The spin decay is a pure scale, so undoing it is a division. Do it before
  // the velocity solve, which needs w_n.
  s.wx /= P.Ad;
  s.wy /= P.Ad;
  s.wz /= P.Ad;
  solvePrevVelocity(s, P);
  return s;
}

export function roll(s, P, steps, backward = false) {
  const step = backward ? stepBackward : stepForward;
  for (let i = 0; i < steps; i++) step(s, P);
  return s;
}
