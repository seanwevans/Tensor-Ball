// The reverse pass: start the ball in a made position and run the physics
// backwards until it is back in a shooter's hands.
//
// This is the part of the problem that has an exact answer. Forward, "what
// action makes this shot" is a search — the app's agent runs a thousand of them
// a batch and grades what comes back. Backwards it is an evaluation: a ball
// dropping through the middle of the ring at 20ft/s came from exactly one place
// at exactly one velocity, and physics.mjs's stepBackward computes it to machine
// precision. No optimiser, no residual, no tolerance.
//
// What the sampling has to get right is only which made positions are *legal*:
// a ball through the ring at 3ft/s straight down has an exact history too, and
// that history is a shooter standing on a ladder holding it. So every sample is
// run through the app's own gates on the way back — the shot has to leave from
// a height Ball.spawn would draw (CONFIG.spawnHeight), from a spot Ball.spawn
// would pick, on an action inside CONFIG.launch's envelope — and the ones that
// come back clean are, exactly, shots the agent could have taken and made.
//
// Two small fixed points make the launch land on the numbers a caller asked
// for rather than on whatever fell out:
//
//   - Spin decays forwards (angularDamping), so it *grows* backwards. Starting
//     the reverse pass with the spin the launch should carry would arrive at
//     the ring with too much of it. The entry spin is instead pre-divided by
//     the decay over the flight, which takes one pass to measure.
//   - Ball.launch puts the spin axis across the line from the shot to the ring,
//     and that line is not known until the shot has been traced back to a
//     shooter. One pass locates the shooter, the next re-aims the axis.
//
// Both converge in two or three passes because neither materially changes the
// flight's length.
import { copyState, state, stepBackward } from "./physics.mjs";
import { launchToAction, launchFrame, spawnLegal } from "./court.mjs";

// The density Ball.spawn draws eye heights from: a normal truncated to the ends
// of an NBA roster. Used to pick between the frames of a reverse pass that land
// in the band, so a generated set of shots has the app's own height mix rather
// than whichever frame happened to be closest.
function heightWeight(y, S) {
  if (y < S.min || y > S.max) return 0;
  const z = (y - S.mean) / S.stdDev;
  return Math.exp(-0.5 * z * z);
}

export function makeReverse(app, P, clearance, flight) {
  const { CONFIG } = app;
  const rim = CONFIG.rim;
  const S = CONFIG.spawnHeight;
  const L = CONFIG.launch;
  const maxSteps = app.STOP.timeout;

  // One reverse pass from a fully specified entry state. Returns the frames
  // that landed in the launch-height band, newest first, or null if the arc
  // never came back down into it.
  function traceBack(entry, spinAxis, spinAtEntry) {
    const s = copyState(entry);
    s.wx = spinAxis.x * spinAtEntry;
    s.wy = 0;
    s.wz = spinAxis.z * spinAtEntry;
    const candidates = [];
    for (let n = 1; n <= maxSteps; n++) {
      stepBackward(s, P);
      // The launch leg: in forward time the ball is still rising, so anywhere
      // the reverse pass is past the apex and vy is positive is a moment the
      // shot could have left from.
      if (s.vy > 0) {
        if (s.py > S.max) continue;
        if (s.py < S.min) break;
        candidates.push({ n, s: copyState(s) });
      }
    }
    return candidates.length ? candidates : null;
  }

  // A made position and the shot that produced it.
  //
  //   entryOffset  how far off the ring's axis the ball crosses, as a fraction
  //                of CONFIG.hoopEntry.scoreRadius. 1 is the edge of what counts
  //                as a basket; the default leaves a little room so the answer
  //                is a make rather than a coin flip on the last decimal.
  //   entrySpeed   ft/s through the ring.
  //   entryAngle   degrees below horizontal on the way through.
  //   spinAction   the fourth action channel the launch should carry, in
  //                [-1, 1]; the reverse pass arranges the entry spin to produce
  //                exactly it.
  function solve({
    rand,
    entryOffset = null,
    entrySpeed = null,
    entryAngle = null,
    entryAzimuth = null,
    spinAction = null,
    offsetLimit = 0.8,
    // Wide enough to reach both ends of the court. A layup comes through the
    // ring slowly and steeply and a heave from 40ft comes through fast and
    // flatter; anything outside these is a shot no legal launch produces, and
    // the gates below reject it anyway — this only decides how many draws are
    // spent finding out.
    speedRange = [8, 34],
    angleRange = [25, 80],
    spinRange = [-1, 1],
    minMargin = 0,
    maxSpawnRadius = CONFIG.curriculum.maxRadius,
    requireClean = true
  } = {}) {
    const u = rand || Math.random;
    // The entry point, area-uniform over the disc it is allowed to cross in.
    const offR =
      (entryOffset === null ? Math.sqrt(u()) * offsetLimit : entryOffset) *
      CONFIG.hoopEntry.scoreRadius;
    const offA = u() * Math.PI * 2;
    const ex = rim.x + Math.cos(offA) * offR;
    const ez = rim.z + Math.sin(offA) * offR;

    const speed = entrySpeed === null ? speedRange[0] + u() * (speedRange[1] - speedRange[0]) : entrySpeed;
    const angle =
      ((entryAngle === null ? angleRange[0] + u() * (angleRange[1] - angleRange[0]) : entryAngle) *
        Math.PI) /
      180;
    // Which way the ball is travelling as it goes through. The shooter ends up
    // roughly opposite; the ones that end up off the court are rejected below.
    const az = entryAzimuth === null ? u() * Math.PI * 2 : entryAzimuth;
    const horiz = speed * Math.cos(angle);
    const target = spinAction === null ? spinRange[0] + u() * (spinRange[1] - spinRange[0]) : spinAction;
    const spin = target * L.spin;

    const entry = state(
      ex,
      rim.y,
      ez,
      Math.cos(az) * horiz,
      -speed * Math.sin(angle),
      Math.sin(az) * horiz
    );

    // The spin axis Ball.launch would have used, guessed from the entry heading
    // and then re-aimed once the shooter is located. w = -spin * dirSide.
    //
    // The height is picked from one draw taken before the loop rather than a
    // fresh one per pass: re-rolling which frame is the launch every time would
    // keep moving the target the fixed point is chasing.
    const heightPick = u();
    let axis = { x: -Math.sin(az), z: Math.cos(az) };
    let decay = 1;
    let chosen = null;
    for (let pass = 0; pass < 8; pass++) {
      const candidates = traceBack(entry, axis, -spin * decay);
      if (!candidates) return null;
      let total = 0;
      for (const c of candidates) total += (c.weight = heightWeight(c.s.py, S));
      if (total <= 0) return null;
      let pick = heightPick * total;
      chosen = candidates[candidates.length - 1];
      for (const c of candidates) {
        pick -= c.weight;
        if (pick <= 0) {
          chosen = c;
          break;
        }
      }
      const f = launchFrame(chosen.s.px, chosen.s.pz, rim);
      const nextDecay = Math.pow(P.Ad, chosen.n);
      const settled =
        Math.abs(f.sxx - axis.x) < 1e-12 &&
        Math.abs(f.szz - axis.z) < 1e-12 &&
        Math.abs(nextDecay - decay) < 1e-15;
      axis = { x: f.sxx, z: f.szz };
      decay = nextDecay;
      if (settled) break;
    }

    const launch = chosen.s;
    if (!spawnLegal(launch.px, launch.pz, app)) return null;
    const dist = Math.hypot(launch.px - rim.x, launch.pz - rim.z);
    if (dist > maxSpawnRadius) return null;

    const act = launchToAction(launch, launch.px, launch.pz, rim, L);
    if (!act.inEnvelope || act.margin < minMargin) return null;

    // The shot as the app would fly it. Everything above is exact arithmetic;
    // this is the measurement that says the exact arithmetic was about the
    // right thing.
    const check = flight.simulate(launch);
    if (requireClean && (check.contact || !check.scored)) return null;

    return {
      spawn: { x: launch.px, y: launch.py, z: launch.pz },
      action: act.action,
      launch: { vx: launch.vx, vy: launch.vy, vz: launch.vz, wx: launch.wx, wy: launch.wy, wz: launch.wz },
      launchSpeed: Math.hypot(launch.vx, launch.vy, launch.vz),
      vFwd: act.vFwd,
      vUp: act.vUp,
      vSide: act.vSide,
      spin: act.spin,
      margin: act.margin,
      shotDistance: dist,
      steps: chosen.n,
      // Offsets from the ring's axis, so they line up with `verified` below
      // rather than being the same point in different coordinates.
      entry: {
        offsetX: ex - rim.x,
        offsetZ: ez - rim.z,
        offset: offR,
        speed,
        angleDeg: (angle * 180) / Math.PI,
        azimuth: az
      },
      // What the forward flight actually did, so a caller never has to take the
      // reverse pass's word for it.
      verified: {
        scored: check.scored,
        entryX: check.entryX,
        entryZ: check.entryZ,
        entryOffset: check.entryOffset,
        entrySpeed: check.entrySpeed,
        frames: check.frames,
        minGap: check.minGap,
        contact: check.contact,
        // How far the forward flight's crossing landed from the one the reverse
        // pass started at: the round-trip error of the whole exercise, in feet.
        closure: Math.hypot(check.entryX - (ex - rim.x), check.entryZ - (ez - rim.z))
      }
    };
  }

  return { solve, traceBack };
}
