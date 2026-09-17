// A shot, flown forward, graded by the app's own rules.
//
// This mirrors TrainingArena.update step for step, including the order things
// happen in — App._loop steps the world first and updates the balls second, so
// the first position any ball is ever graded at is one frame of flight in, and
// trackHoopPass's `prev` starts there rather than at the launch. That is why
// `scored` below is not "did the arc pass through the ring" but "would the app
// have credited it", which is the only question worth answering.
//
// Contacts are the boundary of what this can model: it integrates free flight,
// so the moment the ball would touch the ring, the board, the post or the floor
// its answer stops being the app's. Rather than integrate on through a rim it
// cannot bounce off, the flight ends and says what it hit. Every solution this
// folder emits is contact-free by construction, and cannoncheck.mjs re-flies it
// in cannon-es with the bodies present to confirm it.
import { copyState, stepForward } from "./physics.mjs";

export function makeFlight(app, P, clearance) {
  const { CONFIG, STOP } = app;
  const rim = CONFIG.rim;
  const entry = CONFIG.hoopEntry;
  const R = CONFIG.ballRadius;
  // Court._buildHoop's scoring sensor, as a cylinder rather than cannon's
  // eight-sided prism: slightly fatter than the real trigger, so a shot this
  // calls an illegal entry might not be one, and never the other way round. It
  // exists only so a pathological arc cannot be scored by accident; no solution
  // this folder produces comes near it.
  const sensorY = app.GEOM.sensor.y;
  const sensorHalf = app.GEOM.sensor.halfHeight;
  const sensorR = app.GEOM.sensor.radius;

  const s = {};
  const result = {
    scored: false,
    entryX: 0,
    entryZ: 0,
    entryOffset: Infinity,
    entryVy: 0,
    entrySpeed: 0,
    frames: 0,
    minDist: Infinity,
    minGap: Infinity,
    contact: null,
    enteredFromBelow: false,
    stop: null
  };

  // maxSteps defaults to the app's own flight timeout.
  // stopOnContact is the switch between the two questions this can answer. On,
  // the flight ends where the app's would stop being free flight. Off, it
  // integrates the free-flight arc through the bodies and only records that it
  // would have touched one — which is what a root find wants, because "did the
  // arc cross the ring's plane" has to stay a smooth function of the launch for
  // a bracket to mean anything. A shot that clips the front of the rim and one
  // that clips the back of it would otherwise both report "never crossed", with
  // the answer sitting between them.
  function simulate(
    s0,
    { maxSteps = STOP.timeout + 1, path = null, stopAtScore = true, stopOnContact = true } = {}
  ) {
    copyState(s0, s);
    result.scored = false;
    result.entryOffset = Infinity;
    result.entryX = result.entryZ = result.entryVy = result.entrySpeed = 0;
    result.frames = 0;
    result.minDist = Infinity;
    result.minGap = Infinity;
    result.contact = null;
    result.enteredFromBelow = false;
    result.stop = null;
    if (path) path.length = 0;

    let prevX = null;
    let prevY = 0;
    let prevZ = 0;
    let lastX = s.px;
    let lastY = s.py;
    let lastZ = s.pz;
    let lastGap = clearance.gap(s.px, s.py, s.pz);
    if (lastGap <= 0) {
      result.contact = "launch";
      if (stopOnContact) {
        result.stop = "contact";
        return result;
      }
    }

    for (let n = 1; n <= maxSteps; n++) {
      stepForward(s, P);
      result.frames = n;
      if (path) path.push(s.px, s.py, s.pz);

      // Did the ball pass through anything on the way here? The app's cannon
      // world only samples the frame positions; this sweeps the segment between
      // them, which is strictly stricter.
      const g = clearance.gap(s.px, s.py, s.pz);
      const len = Math.hypot(s.px - lastX, s.py - lastY, s.pz - lastZ);
      // (gap(a) + gap(b) - len)/2 is a lower bound on the gap anywhere between
      // them, because the gap is 1-Lipschitz. Above it, the sweep is skippable.
      const bound = (lastGap + g - len) / 2;
      if (bound <= 1e-3) {
        const sw = clearance.sweep(lastX, lastY, lastZ, s.px, s.py, s.pz);
        if (sw.gap < result.minGap) result.minGap = sw.gap;
        if (sw.contact) {
          result.contact = result.contact || "body";
          if (stopOnContact) {
            result.stop = "contact";
            return result;
          }
        }
      } else if (bound < result.minGap) {
        result.minGap = bound;
      }
      lastGap = g;
      lastX = s.px;
      lastY = s.py;
      lastZ = s.pz;

      // isEntryFromBelow, on the sensor's column.
      if (
        !result.enteredFromBelow &&
        s.vy > entry.minAscentSpeed &&
        Math.abs(s.py - sensorY) <= sensorHalf + R
      ) {
        const dx = s.px - rim.x;
        const dz = s.pz - rim.z;
        const d2 = dx * dx + dz * dz;
        const reach = sensorR + R;
        if (d2 <= reach * reach && d2 <= entry.columnRadius * entry.columnRadius)
          result.enteredFromBelow = true;
      }

      // trackHoopPass: a descending crossing of the ring's plane, interpolated.
      if (prevX !== null && !result.scored && !result.enteredFromBelow) {
        if (prevY >= rim.y && s.py < rim.y) {
          const t = (prevY - rim.y) / (prevY - s.py);
          const cx = prevX + (s.px - prevX) * t - rim.x;
          const cz = prevZ + (s.pz - prevZ) * t - rim.z;
          result.entryX = cx;
          result.entryZ = cz;
          result.entryOffset = Math.sqrt(cx * cx + cz * cz);
          result.entryVy = s.vy;
          result.entrySpeed = Math.sqrt(s.vx * s.vx + s.vy * s.vy + s.vz * s.vz);
          // Squared, the way trackHoopPass writes it: hypot and sqrt-of-sums
          // are not the same function, and a shot exactly on the edge of the
          // hole should be graded here the way the app grades it.
          if (cx * cx + cz * cz <= entry.scoreRadius * entry.scoreRadius)
            result.scored = true;
        }
      }
      prevX = s.px;
      prevY = s.py;
      prevZ = s.pz;

      // Vec3.distanceTo, which is what TrainingArena.update measures with.
      const ddx = s.px - rim.x;
      const ddy = s.py - rim.y;
      const ddz = s.pz - rim.z;
      const dist = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
      if (dist < result.minDist) result.minDist = dist;

      if (result.scored && stopAtScore) {
        result.stop = "scored";
        return result;
      }
      if (s.py < STOP.floorY) {
        result.stop = "floor";
        return result;
      }
      if (Math.abs(s.px) > STOP.oobX || Math.abs(s.pz) > STOP.oobZ) {
        result.stop = "oob";
        return result;
      }
      // Vec3.length().
      if (
        Math.sqrt(s.vx * s.vx + s.vy * s.vy + s.vz * s.vz) < STOP.stoppedSpeed &&
        n > STOP.stoppedAfter
      ) {
        result.stop = "stopped";
        return result;
      }
      if (n > STOP.timeout) {
        result.stop = "timeout";
        return result;
      }
    }
    result.stop = "maxsteps";
    return result;
  }

  return { simulate, result };
}
