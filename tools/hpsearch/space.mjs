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
    { name: "noise-lo", set: { exploreNoise: 0.2, exploreNoiseMin: 0.08 } },
    { name: "noise-hi", set: { exploreNoise: 0.8, exploreNoiseMin: 0.3 } },
    { name: "noise-vhi", set: { exploreNoise: 1.4, exploreNoiseMin: 0.4 } },
    // Default decay is 0.999/batch — over a run this short it may as well be 1.
    { name: "decay-fast", set: { exploreNoiseDecay: 0.93 } },

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

  combo: [],
  final: []
};
