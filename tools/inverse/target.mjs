// The other direction: name a spot on the floor, get the shot.
//
// The reverse pass in reverse.mjs answers "what made this basket" in closed
// form, but it does not get to choose where the shooter is standing — that
// falls out of the arc. For cross-validation the question usually runs the
// other way: here is a state the agent was in, what was the right action?
//
// That is a boundary-value problem, and it collapses to a one-dimensional root
// find once two things are noticed:
//
//   - Ball.launch aims the shot down the line to the ring, so with the side
//     channel at zero the whole flight lies in the vertical plane through the
//     ring's axis. Drag is along the velocity and Magnus is w x v about an axis
//     Ball.launch puts square across that plane, so neither can push the ball
//     out of it. The crossing is therefore dead centre laterally, exactly, and
//     side = 0 is not a good guess — it is the answer. (`lateral` on every
//     result is the measured confirmation, and it reads 0.)
//   - What is left is one equation: the arc has to come down through the ring's
//     plane at the distance the ring actually is. For a fixed launch height,
//     that is one condition on two free numbers (up and forward), so the makes
//     form a curve, not a point — the same family a shooter picks an arc from.
//
// So: sweep the up channel and the spin channel, solve the forward channel
// exactly on each, and the answer is the whole family of makes from that spot.
// Which one to call *the* answer is then a real choice. The one made here is
// the shot with the most room for error — the widest box in action space that
// still drops the ball through the hole — because that is the shot a policy
// with any spread at all wants to be aiming at, and because the width is the
// number worth reporting on its own: it says how precisely the agent has to
// act, in the units the actor emits.
import { state } from "./physics.mjs";
import { launchFrame, launchToAction } from "./court.mjs";

export function makeTarget(app, P, clearance, flight) {
  const { CONFIG } = app;
  const rim = CONFIG.rim;
  const L = CONFIG.launch;
  const scoreRadius = CONFIG.hoopEntry.scoreRadius;
  // Action units per ft/s, per channel. The fwd and up channels each stretch
  // [-1, 1] over their whole range, so a foot per second is 2/(hi - lo) of
  // action and not 1/(hi - lo).
  const perFwd = 2 / (L.fwdMax - L.fwdMin);
  const perUp = 2 / (L.upMax - L.upMin);
  const perSide = L.side ? 1 / L.side : 0;
  const perSpin = L.spin ? 1 / L.spin : 0;

  const s = state(0, 0, 0, 0, 0, 0);

  // How steeply the ball came down through the ring, in degrees below
  // horizontal — the number a shooting coach would call the arc.
  const entryAngle = (hit) =>
    (Math.atan2(-hit.entryVy, Math.sqrt(Math.max(0, hit.entrySpeed ** 2 - hit.entryVy ** 2))) * 180) /
    Math.PI;

  // Fly one shot from a spawn on explicit launch numbers, and report where it
  // crossed the ring's plane relative to the ring — signed along the line to
  // the hoop, so negative is short and positive is long.
  //
  // Free flight all the way through (see flight.mjs's stopOnContact): a shot
  // that clips the front of the ring and one that clips the back of it would
  // otherwise both report "never crossed", with the answer sitting between them
  // and no bracket able to find it.
  function shoot(spawn, frame, vFwd, vUp, vSide, spin) {
    s.px = spawn.x;
    s.py = spawn.y;
    s.pz = spawn.z;
    s.vx = frame.fx * vFwd + frame.sxx * vSide;
    s.vy = vUp;
    s.vz = frame.fz * vFwd + frame.szz * vSide;
    s.wx = -frame.sxx * spin;
    s.wy = 0;
    s.wz = -frame.szz * spin;
    const r = flight.simulate(s, { stopOnContact: false });
    const crossed = Number.isFinite(r.entryOffset);
    return {
      crossed,
      radial: crossed ? r.entryX * frame.fx + r.entryZ * frame.fz : NaN,
      lateral: crossed ? r.entryX * frame.sxx + r.entryZ * frame.szz : NaN,
      scored: r.scored,
      contact: r.contact,
      minGap: r.minGap,
      entrySpeed: r.entrySpeed,
      entryVy: r.entryVy,
      frames: r.frames,
      clean: r.scored && !r.contact
    };
  }

  // The largest forward speed from which the arc still reaches the ring's
  // height at all. Past it the extra drag holds the apex under 10ft, or the
  // shot leaves the world before it comes down; either way there is no crossing
  // to measure, and both are things only more forward speed can cause, so the
  // boundary is a single one.
  //
  // `vSide` is carried rather than assumed zero because this is also what
  // diagnose.mjs asks about a policy's own action, which has a side channel of
  // its own and is entitled to keep it.
  function reachLimit(spawn, frame, vUp, spin, vSide = 0) {
    if (shoot(spawn, frame, L.fwdMax, vUp, vSide, spin).crossed) return L.fwdMax;
    if (!shoot(spawn, frame, L.fwdMin, vUp, vSide, spin).crossed) return null;
    let lo = L.fwdMin;
    let hi = L.fwdMax;
    for (let i = 0; i < 24; i++) {
      const mid = (lo + hi) / 2;
      if (shoot(spawn, frame, mid, vUp, vSide, spin).crossed) lo = mid;
      else hi = mid;
    }
    return lo;
  }

  // Solve `radial(vFwd) = want` on [lo, hi]. The residual is the horizontal
  // distance the arc covers before it comes back down through the ring's plane,
  // which grows with forward speed, so the endpoints bracket it; a short scan
  // catches the case where they do not rather than assuming they must.
  //
  // Illinois-modified false position: the residual is close to linear in vFwd —
  // its slope is roughly the flight time — so a secant step lands near the root
  // immediately, and the bracket keeps it honest when it does not.
  function solveRadial(spawn, frame, vUp, spin, want, lo, hi, scan = 16, vSide = 0) {
    const f = (x) => shoot(spawn, frame, x, vUp, vSide, spin).radial - want;
    let a = lo;
    let fa = f(a);
    let b = hi;
    let fb = f(b);
    if (!Number.isFinite(fa)) return null;
    if (!Number.isFinite(fb) || fa * fb > 0) {
      // Endpoints do not bracket: walk in and look for a sign change.
      let px = a;
      let pf = fa;
      let found = false;
      for (let i = 1; i <= scan; i++) {
        const x = lo + ((hi - lo) * i) / scan;
        const fx = f(x);
        if (!Number.isFinite(fx)) break;
        if (pf * fx <= 0) {
          a = px;
          fa = pf;
          b = x;
          fb = fx;
          found = true;
          break;
        }
        px = x;
        pf = fx;
      }
      if (!found) return null;
    }
    if (fa === 0) return a;
    if (fb === 0) return b;
    let side = 0;
    for (let i = 0; i < 40; i++) {
      const c = (fa * b - fb * a) / (fa - fb);
      if (!Number.isFinite(c) || c <= Math.min(a, b) || c >= Math.max(a, b)) break;
      const fc = f(c);
      if (fc === 0 || Math.abs(b - a) < 1e-12) return c;
      if (fa * fc < 0) {
        b = c;
        fb = fc;
        if (side === -1) fa /= 2;
        side = -1;
      } else {
        a = c;
        fa = fc;
        if (side === 1) fb /= 2;
        side = 1;
      }
    }
    return (a + b) / 2;
  }

  // Walk out from a value known to make the shot until it stops making it. The
  // makes along any one axis form a single interval, so bisecting the predicate
  // from inside it finds the edge.
  function edge(good, bad, ok) {
    if (ok(bad)) return bad;
    for (let i = 0; i < 30; i++) {
      const mid = (good + bad) / 2;
      if (mid === good || mid === bad) break;
      if (ok(mid)) good = mid;
      else bad = mid;
    }
    return good;
  }

  // Every arc that makes the shot from this spot: one per sampled launch angle
  // and spin, with the interval of forward speeds each of them tolerates.
  function family(spawn, { upSteps = 48, spinValues = [0], minGap = 0 } = {}) {
    const frame = launchFrame(spawn.x, spawn.z, rim);
    const arcs = [];
    for (const spinAction of spinValues) {
      const spin = spinAction * L.spin;
      for (let i = 0; i <= upSteps; i++) {
        const vUp = L.upMin + ((L.upMax - L.upMin) * i) / upSteps;
        const limit = reachLimit(spawn, frame, vUp, spin);
        if (limit === null) continue;
        const centre = solveRadial(spawn, frame, vUp, spin, 0, L.fwdMin, limit);
        if (centre === null) continue;
        const hit = shoot(spawn, frame, centre, vUp, 0, spin);
        if (!hit.clean || hit.minGap < minGap) continue;

        const ok = (x) =>
          x >= L.fwdMin &&
          x <= L.fwdMax &&
          (() => {
            const r = shoot(spawn, frame, x, vUp, 0, spin);
            return r.clean && r.minGap >= minGap;
          })();
        // Bounded by the score radius or by the envelope, whichever bites
        // first, then trimmed to where the arc still clears the ring.
        const shortOf = solveRadial(spawn, frame, vUp, spin, -scoreRadius, L.fwdMin, limit);
        const longOf = solveRadial(spawn, frame, vUp, spin, scoreRadius, L.fwdMin, limit);
        const lo = edge(centre, Math.max(L.fwdMin, shortOf ?? L.fwdMin), ok);
        const hi = edge(centre, Math.min(L.fwdMax, longOf ?? L.fwdMax), ok);

        arcs.push({
          vUp,
          vFwd: centre,
          spin,
          spinAction,
          window: { lo, hi },
          // In action units, which is what the policy's sigma is in.
          fwdHalf: ((hi - lo) / 2) * perFwd,
          entrySpeed: hit.entrySpeed,
          entryVy: hit.entryVy,
          entryAngleDeg: entryAngle(hit),
          minGap: hit.minGap,
          frames: hit.frames
        });
      }
    }
    return { frame, arcs };
  }

  // The half-width of the make set along one channel, in action units, with
  // every other channel held at the arc's own value. These are the sides of the
  // box inscribed in the make set, not its diagonal: the set is a thin curved
  // sliver in (fwd, up), and moving along it is far cheaper than moving across.
  function channelTolerance(spawn, frame, arc, minGap) {
    const clean = (vFwd, vUp, vSide, spin) => {
      if (vFwd < L.fwdMin || vFwd > L.fwdMax) return false;
      if (vUp < L.upMin || vUp > L.upMax) return false;
      if (Math.abs(vSide) > L.side) return false;
      if (Math.abs(spin) > L.spin) return false;
      const r = shoot(spawn, frame, vFwd, vUp, vSide, spin);
      return r.clean && r.minGap >= minGap;
    };
    const span = (probe, lo, hi, per) => ({
      plus: (edge(0, hi, probe) - 0) * per,
      minus: (0 - edge(0, lo, probe)) * per
    });
    return {
      fwd: {
        plus: (arc.window.hi - arc.vFwd) * perFwd,
        minus: (arc.vFwd - arc.window.lo) * perFwd
      },
      up: span(
        (d) => clean(arc.vFwd, arc.vUp + d, 0, arc.spin),
        L.upMin - arc.vUp,
        L.upMax - arc.vUp,
        perUp
      ),
      side: span((d) => clean(arc.vFwd, arc.vUp, d, arc.spin), -L.side, L.side, perSide),
      spin: span(
        (d) => clean(arc.vFwd, arc.vUp, 0, arc.spin + d),
        -L.spin - arc.spin,
        L.spin - arc.spin,
        perSpin
      )
    };
  }

  // The exact answer for one spawn.
  //
  // Ranking the whole family by its up-channel tolerance as well would cost a
  // pair of bisections per arc, so it is done in two stages: rank on the
  // forward window, which is the tightest channel and the cheapest to have
  // already, then measure the rest for the best few and keep whichever has the
  // largest (fwd x up) box. `refine` is how many get the second look.
  function solve(spawn, opts = {}) {
    const { refine = 6, minGap = 0 } = opts;
    const { frame, arcs } = family(spawn, opts);
    if (!arcs.length)
      return { spawn, reachable: false, arcs: 0, distance: frame.distance };

    const ranked = arcs.slice().sort((a, b) => b.fwdHalf - a.fwdHalf);
    let best = null;
    let bestTol = null;
    let bestScore = -1;
    for (const arc of ranked.slice(0, Math.max(1, refine))) {
      const tol = channelTolerance(spawn, frame, arc, minGap);
      const score =
        Math.min(tol.fwd.plus, tol.fwd.minus) * Math.min(tol.up.plus, tol.up.minus);
      if (score > bestScore) {
        bestScore = score;
        best = arc;
        bestTol = tol;
      }
    }

    const action = launchToAction(
      {
        vx: frame.fx * best.vFwd,
        vy: best.vUp,
        vz: frame.fz * best.vFwd,
        wx: -frame.sxx * best.spin,
        wy: 0,
        wz: -frame.szz * best.spin
      },
      spawn.x,
      spawn.z,
      rim,
      L,
      frame
    );
    const hit = shoot(spawn, frame, best.vFwd, best.vUp, 0, best.spin);

    return {
      spawn,
      reachable: true,
      distance: frame.distance,
      action: action.action,
      arcs: arcs.length,
      arc: {
        vFwd: best.vFwd,
        vUp: best.vUp,
        vSide: 0,
        spin: best.spin,
        frames: best.frames,
        entrySpeed: hit.entrySpeed,
        entryVy: hit.entryVy,
        entryAngleDeg: entryAngle(hit),
        minGap: hit.minGap
      },
      tolerance: bestTol,
      // The family this was picked out of: how wide a band of launch angles
      // makes the shot at all, and the best forward window anywhere in it.
      arcSpan: {
        vUpMin: Math.min(...arcs.map((a) => a.vUp)),
        vUpMax: Math.max(...arcs.map((a) => a.vUp)),
        entryAngleMin: Math.min(...arcs.map((a) => a.entryAngleDeg)),
        entryAngleMax: Math.max(...arcs.map((a) => a.entryAngleDeg))
      },
      // Measured, not assumed: the crossing's offset across the line of the
      // shot, which the argument at the top of this file says is exactly zero.
      lateral: hit.lateral
    };
  }

  return { shoot, family, solve, reachLimit, solveRadial, channelTolerance };
}
