// The same problem again, in closed form.
//
// target.mjs finds the shot by root-finding against the integrator. That is
// exact and it works with the air on, but it is a *search*: it asks the physics
// a few dozen questions per answer. With the air off there is no need to ask
// anything, because the app's integrator has a closed form, and once it is
// written down the inverse falls out as algebra.
//
// ## The integrator is a parabola, exactly
//
// Semi-implicit Euler at constant acceleration,
//
//     v_{n+1} = v_n + g*dt          p_{n+1} = p_n + v_{n+1}*dt
//
// sums in closed form: v_n = v_0 + n*g*dt, and
//
//     p_n = p_0 + n*dt*v_0 + g*dt^2 * n(n+1)/2
//         = p_0 + (v_0 + g*dt/2)*t + (g/2)*t^2        with t = n*dt.
//
// So the app's trajectory is not *approximately* a parabola — it is one, with
// the launch velocity shifted by half a step of gravity. Every classical
// projectile formula therefore applies to this app verbatim, provided the
// shift is carried. That shift is 32.2/120 = 0.268ft/s of vertical, which is
// not a rounding error: it is two thirds of the whole side channel.
//
// ## The reverse map, with no search in it
//
// Put the origin of time at the moment the ball drops through the ring, let
// `drop` be the (positive) height from the launch to the ring, and let `w` be
// the ball's vertical speed as it goes through. Then the launch's vertical
// speed is conservation of energy and nothing else,
//
//     vUp~ = sqrt(w^2 + 2*g*drop)          flightTime = (vUp~ + w)/g
//
// and the distance is `vFwd * flightTime`. One square root, no iteration: a
// made basket names its own shooter.
//
// ## The forward map, also with no search in it
//
// Going the other way — given how far out the shot is and how steeply it is to
// come through the ring — squaring away the radical leaves
//
//     vFwd^2 = g*D^2 / (2*drop + 2*D*tan(entry))
//
// which is exact and unconditional: *every* entry angle has a shot, and the
// only question is whether its two speeds are inside CONFIG.launch's envelope.
// This is the piece target.mjs has to bisect for.
//
// ## And the tolerance
//
// Holding the launch's vertical speed fixed, the flight time is fixed too, so
// the distance is just `vFwd * flightTime` and the window of forward speeds
// that still finds the hole is
//
//     half-window = scoreRadius / flightTime
//
// exactly. Which says something the numerical sweep could only show: the
// forward channel's tolerance is *worse* on a higher arc, because the ball is
// in the air longer and a given speed error has longer to accumulate. Steep
// arcs win on the vertical channel and lose on this one, and that trade is
// what target.mjs's canonical choice is resolving.
//
// ## What it is for
//
// Two things. It is an orthogonal check on the numerical solver — same answer,
// no shared code, no shared method — so where the two agree to machine
// precision the answer is not resting on either one being right. And with the
// air on, the gap between them is a measurement of exactly what drag and
// Magnus are worth, which is otherwise not separable from everything else.

export function makeAnalytic(app) {
  const { CONFIG, PHYS } = app;
  const g = -PHYS.gravity;
  const dt = PHYS.dt;
  // The half-step of gravity that turns the discrete trajectory into a
  // continuous parabola. Add it to a launch velocity to get the parabola's,
  // subtract it to get back.
  const shift = (g * dt) / 2;
  const rimY = CONFIG.rim.y;
  const L = CONFIG.launch;
  const scoreRadius = CONFIG.hoopEntry.scoreRadius;
  const perFwd = 2 / (L.fwdMax - L.fwdMin);
  const perUp = 2 / (L.upMax - L.upMin);

  // A shot from `distance` feet out, launched from `height`, coming down
  // through the ring at `entryDeg` degrees below horizontal. Closed form.
  function fromEntryAngle(distance, height, entryDeg) {
    const drop = rimY - height;
    const tanE = Math.tan((entryDeg * Math.PI) / 180);
    const denom = 2 * drop + 2 * distance * tanE;
    if (!(denom > 0)) return null;
    const vFwd = Math.sqrt((g * distance * distance) / denom);
    // The parabola's launch slope, then the app's actual launch velocity.
    const upParabola = Math.sqrt(vFwd * vFwd * tanE * tanE + 2 * g * drop);
    const vUp = upParabola + shift;
    const flightTime = (upParabola + vFwd * tanE) / g;
    return {
      vFwd,
      vUp,
      flightTime,
      entryDeg,
      // Vertical speed through the ring, as the app's discrete state carries
      // it: the parabola's slope there, minus the same half-step.
      entryVy: -(vFwd * tanE) - shift,
      entrySpeed: Math.hypot(vFwd, vFwd * tanE + shift),
      // scoreRadius / flightTime, in the units the actor emits.
      fwdTolerance: (scoreRadius / flightTime) * perFwd,
      action: [
        (2 * (vFwd - L.fwdMin)) / (L.fwdMax - L.fwdMin) - 1,
        (2 * (vUp - L.upMin)) / (L.upMax - L.upMin) - 1,
        0,
        0
      ],
      inEnvelope:
        vFwd >= L.fwdMin && vFwd <= L.fwdMax && vUp >= L.upMin && vUp <= L.upMax
    };
  }

  // The same shot named by its launch angle instead — the classical form, and
  // the one that shows the constraint directly: a shot has to be aimed above
  // the line from the release to the ring or it cannot get there.
  function fromLaunchAngle(distance, height, launchDeg) {
    const drop = rimY - height;
    const tanL = Math.tan((launchDeg * Math.PI) / 180);
    const denom = 2 * (distance * tanL - drop);
    if (!(denom > 0)) return null;
    const vFwd = Math.sqrt((g * distance * distance) / denom);
    const upParabola = vFwd * tanL;
    const flightTime = distance / vFwd;
    const slopeAtRing = upParabola - g * flightTime;
    if (slopeAtRing >= 0) return null; // still climbing at the ring
    return {
      vFwd,
      vUp: upParabola + shift,
      flightTime,
      launchDeg,
      entryDeg: (Math.atan2(-slopeAtRing, vFwd) * 180) / Math.PI,
      entryVy: slopeAtRing - shift,
      fwdTolerance: (scoreRadius / flightTime) * perFwd,
      inEnvelope:
        vFwd >= L.fwdMin &&
        vFwd <= L.fwdMax &&
        upParabola + shift >= L.upMin &&
        upParabola + shift <= L.upMax
    };
  }

  // The reverse map: a ball dropping through the ring at this speed and angle
  // came from exactly here. No search, one square root — the closed-form twin
  // of reverse.mjs's stepping pass.
  function shooterOf(entrySpeed, entryDeg, height) {
    const drop = rimY - height;
    const e = (entryDeg * Math.PI) / 180;
    const vFwd = entrySpeed * Math.cos(e);
    // The parabola's slope through the ring, from the app's discrete velocity.
    const w = entrySpeed * Math.sin(e) + shift;
    const upParabola = Math.sqrt(w * w + 2 * g * drop);
    const flightTime = (upParabola + w) / g;
    return {
      distance: vFwd * flightTime,
      vFwd,
      vUp: upParabola + shift,
      flightTime,
      fwdTolerance: (scoreRadius / flightTime) * perFwd
    };
  }

  // The whole family from one spot, by entry angle — target.mjs's `family`
  // with the searching taken out. Returns only the arcs the agent could
  // actually launch.
  function family(distance, height, { from = 20, to = 85, step = 0.25 } = {}) {
    const arcs = [];
    for (let e = from; e <= to; e += step) {
      const a = fromEntryAngle(distance, height, e);
      if (a && a.inEnvelope) arcs.push(a);
    }
    return arcs;
  }

  // The flattest arc that still fits the envelope is the one with the widest
  // forward window, because the window is scoreRadius/flightTime and a flatter
  // arc is a shorter flight. So the best shot on that criterion is the extreme
  // of the feasible family, not an interior optimum — which is worth knowing
  // before running a sweep to look for one.
  function widestForward(distance, height, opts) {
    const arcs = family(distance, height, opts);
    if (!arcs.length) return null;
    let best = arcs[0];
    for (const a of arcs) if (a.fwdTolerance > best.fwdTolerance) best = a;
    return best;
  }

  return {
    g,
    dt,
    shift,
    fromEntryAngle,
    fromLaunchAngle,
    shooterOf,
    family,
    widestForward,
    perFwd,
    perUp
  };
}
