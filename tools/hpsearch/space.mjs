// The candidates each stage runs.
//
// Stage 1 (`screen`) moves one knob at a time off the shipped CONFIG. With a
// ~20s batch there is no budget for a dense grid, and one-at-a-time answers the
// question that actually matters here — which knobs move accuracy at all —
// instead of handing back a winning tuple with no explanation. Stage 2
// (`combo`) crosses the directions that paid off, stage 3 (`final`) re-runs the
// finalists against the shipped defaults on held-out seeds and a longer horizon.
//
// `set` is a CONFIG overlay: nested keys are merged, everything else is left at
// the app's default.
export const stages = {
  screen: [
    { name: "baseline", set: {} },

    // Adam step for both critic and actor.
    { name: "lr-0003", set: { learningRate: 0.0003 } },
    { name: "lr-003", set: { learningRate: 0.003 } },

    // Width of the action jitter that generates the batch's variety. The actor
    // only ever imitates actions it already took, so this is the only source of
    // anything better than what it currently does.
    //
    // This is no longer a schedule: the policy is Gaussian and learns its own
    // spread, so what is screenable is where it starts and how far it is
    // allowed to travel (CONFIG.policy). The exploreNoise* entries that used to
    // sit here are gone with the knobs; results/screen still holds their runs.
    { name: "std-init-lo", set: { policy: { logStdInit: Math.log(0.1) } } },
    { name: "std-init-hi", set: { policy: { logStdInit: Math.log(0.5) } } },
    // Can the policy commit? A higher floor keeps it exploring whether or not
    // it has earned the right to stop.
    { name: "std-floor-hi", set: { policy: { logStdMin: Math.log(0.08) } } },
    { name: "std-ceil-lo", set: { policy: { logStdMax: Math.log(0.4) } } },

    // How sharply the advantage weighting favours the batch's best shots.
    { name: "temp-sharp", set: { advantageTemp: 0.5 } },
    { name: "temp-soft", set: { advantageTemp: 2.0 } },
    { name: "clip-hi", set: { advantageClip: 100 } },

    // How hard the critic is fit. The shipped single pass leaves it close to
    // predicting the batch mean (loss ~50-100 against a reward spread of the
    // same order), which makes every advantage roughly "reward minus average"
    // and throws away the state-dependence the actor is supposed to exploit.
    { name: "critic-2ep", set: { criticEpochs: 2 } },
    { name: "critic-3ep", set: { criticEpochs: 3 } },
    { name: "l2-lo", set: { l2: 0.0001 } },

    // Adam steps per batch on the actor's head.
    { name: "mb-16", set: { actorMinibatch: 16 } },
    { name: "mb-64", set: { actorMinibatch: 64 } },

    // Reward shape. Advantages are z-scored before weighting, so only the
    // relative geometry of the reward matters, not its scale.
    { name: "shape-tight", set: { reward: { missDistanceScale: 5 } } },
    { name: "shape-loose", set: { reward: { missDistanceScale: 20 } } },
    { name: "bonus-hi", set: { reward: { rim: 6, backboard: 2 } } },
    { name: "score-hi", set: { reward: { score: 100 } } }
  ],

  // NOTE: this stage was derived against the annealed-noise policy and is
  // written entirely in terms of exploreNoiseDecay, a knob that no longer
  // exists — the policy learns its own spread now. It needs re-deriving from a
  // fresh stage 1 before it is run again; it is left here as the record of what
  // was asked last time.
  //
  // Stage 1 said the exploration schedule dominates everything else, and that
  // the shipped one is far too hot for far too long: exploreNoise 0.4 decaying
  // at 0.999/batch barely moves inside a run, and every config that cut the
  // noise beat every config that didn't. Decaying it at 0.93 instead — 0.4 down
  // to the 0.15 floor over ~14 batches — went from 1.1% to 9.6% accuracy while
  // the shipped defaults reached 2.5%.
  //
  // So stage 2 re-centres on that schedule (D = exploreNoiseDecay 0.93) and asks
  // the remaining questions from there. Testing the reward shape or the critic
  // on top of the shipped noise would have been measuring them through a policy
  // that exploration was drowning anyway.
  combo: [
    { name: "d-base", set: { exploreNoiseDecay: 0.93 } },

    // Anneal harder, and further.
    { name: "d-vfast", set: { exploreNoiseDecay: 0.85 } },
    { name: "d-floor05", set: { exploreNoiseDecay: 0.93, exploreNoiseMin: 0.05 } },
    // Stage 1's other winner was simply starting cold (0.2). Does starting cold
    // still help once the schedule anneals, or was it the same finding twice?
    { name: "d-lo", set: { exploreNoise: 0.2, exploreNoiseMin: 0.05, exploreNoiseDecay: 0.93 } },

    // Advantage weighting: both directions beat baseline in stage 1.
    { name: "d-clip", set: { exploreNoiseDecay: 0.93, advantageClip: 100 } },
    { name: "d-temp-clip", set: { exploreNoiseDecay: 0.93, advantageTemp: 0.5, advantageClip: 100 } },

    // A better-fit critic, which costs real time per batch — see score.mjs on
    // why that makes the batch-indexed half of the score generous to it.
    { name: "d-critic2", set: { exploreNoiseDecay: 0.93, criticEpochs: 2 } },

    // Reward shape, asked on top of a policy that is actually learning.
    { name: "d-tight", set: { exploreNoiseDecay: 0.93, reward: { missDistanceScale: 5 } } },
    { name: "d-bonus", set: { exploreNoiseDecay: 0.93, reward: { rim: 6, backboard: 2 } } }
  ],

  final: []
};
