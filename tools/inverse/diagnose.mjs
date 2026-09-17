// What was wrong with the shot the policy actually took.
//
// The obvious cross-validation — compare the agent's action to the exact one —
// measures the wrong thing. There are about 165 arcs from a mid-range spot that
// all go in (see target.mjs), so a policy aiming at a different one than this
// folder happens to call canonical is not making a mistake, and the distance
// between the two actions is mostly a statement about which arc each preferred.
//
// The question worth asking holds the policy's own arc and asks what it got
// wrong about it:
//
//     the agent chose an up channel, a side channel and a spin. Fixing those,
//     what forward speed would have dropped the ball through the middle, and
//     how far off was it?
//
// That is one bracketed root find against the exact physics, and what comes
// back is a signed number in action units — the same units the actor emits and
// CONFIG.policy measures sigma in. It has three properties nothing already in
// the app has:
//
//   - It is dense. Accuracy is a Bernoulli sample that reads zero for the first
//     several batches of a run and tells you nothing about whether anything is
//     being learned. This reads a real number on every shot from the first one.
//   - It is signed, so a policy that is systematically shooting long and one
//     that is systematically short look different rather than both looking like
//     "missed".
//   - It is comparable to the spread. Next to the shot's own tolerance it says
//     whether the mean action is inside the set of makes at all, and next to
//     Policy Sigma it says whether the misses are the mean being wrong or the
//     exploration being wide. Those are completely different problems and the
//     accuracy trace shows the same picture for both.
//
// None of it is learned from, and that is the point: the answer comes from the
// physics, so it cannot be fooled by the same thing that fooled the critic.
import { launchFrame } from "./court.mjs";

export function makeDiagnose(app, P, clearance, flight, target) {
  const { CONFIG } = app;
  const rim = CONFIG.rim;
  const L = CONFIG.launch;
  const scoreRadius = CONFIG.hoopEntry.scoreRadius;
  const perFwd = 2 / (L.fwdMax - L.fwdMin);
  const lerp = (lo, hi, a) => lo + ((a + 1) / 2) * (hi - lo);

  // One greedy shot, taken apart. `action` is the actor's mean for this state —
  // the four numbers Ball.launch would be handed.
  function score(spawn, action, { minGap = 0 } = {}) {
    const frame = launchFrame(spawn.x, spawn.z, rim);
    const vFwd = lerp(L.fwdMin, L.fwdMax, action[0]);
    const vUp = lerp(L.upMin, L.upMax, action[1]);
    const vSide = action[2] * L.side;
    const spin = action[3] * L.spin;

    const took = target.shoot(spawn, frame, vFwd, vUp, vSide, spin);
    const out = {
      distance: frame.distance,
      // Where the shot the policy took actually crossed the ring's plane,
      // signed along the line to the hoop: negative is short, positive is long.
      radial: took.crossed ? took.radial : null,
      lateral: took.crossed ? took.lateral : null,
      scored: took.scored,
      contact: took.contact,
      // Why no forward speed could have saved this arc, when none could.
      blocked: null,
      fwdStar: null,
      fwdError: null,
      fwdTolerance: null,
      inMakeSet: false
    };

    // The side channel spends part of the hole before the forward channel gets
    // a say: a shot pushed half a foot wide cannot be rescued by throwing it
    // harder or softer, and the radius left for the forward channel to land in
    // is what is left of the hole after the miss across it.
    const lateral = took.crossed ? Math.abs(took.lateral) : 0;
    const usable = Math.sqrt(Math.max(0, scoreRadius * scoreRadius - lateral * lateral));
    if (took.crossed && lateral >= scoreRadius) {
      out.blocked = "side";
      return out;
    }

    // Can this arc reach the ring at all? A launch angle too flat to get the
    // ball to 10ft from here is a different mistake from one aimed correctly
    // and thrown too hard, and it is the up channel's, not the forward
    // channel's.
    const limit = target.reachLimit(spawn, frame, vUp, spin, vSide);
    if (limit === null) {
      out.blocked = "arc";
      return out;
    }

    const star = target.solveRadial(spawn, frame, vUp, spin, 0, L.fwdMin, limit, 16, vSide);
    if (star === null) {
      // The arc reaches, but not as far as the ring is — the shot is beyond
      // what this launch angle can carry inside the envelope.
      out.blocked = "range";
      return out;
    }

    const lo = target.solveRadial(spawn, frame, vUp, spin, -usable, L.fwdMin, limit, 16, vSide);
    const hi = target.solveRadial(spawn, frame, vUp, spin, usable, L.fwdMin, limit, 16, vSide);
    out.fwdStar = star;
    // Signed in action units: positive is the policy shooting too hard.
    out.fwdError = (vFwd - star) * perFwd;
    out.fwdTolerance =
      lo !== null && hi !== null ? ((hi - lo) / 2) * perFwd : null;
    out.inMakeSet = took.scored && !took.contact && took.minGap >= minGap;
    return out;
  }

  // A batch of them, pooled. Rates and means only — the per-shot records are
  // the caller's to keep or throw away.
  //
  // `meanAbs` is the headline: the average size of the mistake, in action
  // units, over shots where there was a forward speed that would have worked.
  // `bias` is the same average with the signs left in, so a policy pulling
  // consistently long shows up as long.
  function pool(scores) {
    const acc = {
      n: 0,
      solvable: 0,
      meanAbs: null,
      medianAbs: null,
      bias: null,
      // Share whose mean action is itself a make. At the greedy balls this is
      // the accuracy the policy would have with the exploration turned off,
      // measured rather than inferred.
      inMakeSet: 0,
      // Share within one tolerance of a make, which is the same thing said
      // about the arc the policy chose rather than about the shot it took.
      withinTolerance: 0,
      meanTolerance: null,
      blocked: { side: 0, arc: 0, range: 0 }
    };
    const abs = [];
    let sum = 0;
    let signed = 0;
    let tolSum = 0;
    let tolN = 0;
    for (const s of scores) {
      acc.n++;
      if (s.inMakeSet) acc.inMakeSet++;
      if (s.blocked) {
        acc.blocked[s.blocked]++;
        continue;
      }
      if (s.fwdError === null) continue;
      acc.solvable++;
      const a = Math.abs(s.fwdError);
      abs.push(a);
      sum += a;
      signed += s.fwdError;
      if (s.fwdTolerance !== null) {
        tolSum += s.fwdTolerance;
        tolN++;
        if (a <= s.fwdTolerance) acc.withinTolerance++;
      }
    }
    if (acc.solvable) {
      abs.sort((x, y) => x - y);
      acc.meanAbs = sum / acc.solvable;
      acc.medianAbs = abs[abs.length >> 1];
      acc.bias = signed / acc.solvable;
    }
    if (tolN) acc.meanTolerance = tolSum / tolN;
    for (const k of Object.keys(acc.blocked)) acc.blocked[k] /= acc.n || 1;
    acc.inMakeSet /= acc.n || 1;
    acc.withinTolerance /= acc.n || 1;
    return acc;
  }

  return { score, pool };
}
