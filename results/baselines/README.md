# Reference baselines

These references use existing model records, task densities and simulator initial states. No new model inference is required. All intervals are pointwise 95% whole-layout bootstrap intervals.

## Uniform random placement

For goal g, let p_g = K_g / N_g, the fraction of saved valid grid placements that solve it. The expected success probability within eight independent uniform placements is 1 - (1 - p_g)^8. We average this probability across tasks, rather than applying the formula to the average density.

On the **1,509 density-supported tasks within the 1,641-task main population**, expected random-search success is **16.59 [14.23, 19.07]%**. Eight distinct placements without replacement give **16.60 [14.24, 19.08]%**.

| Method | Eight-attempt full-feedback success (%) | Retry-only success (%) |
|---|---:|---:|
| Uniform random placement | 16.59 [14.23, 19.07] | same policy |
| GPT-5 | 39.10 [35.71, 42.42] | 34.13 [30.61, 37.74] |
| Gemini 3.1 Pro | 43.47 [40.18, 46.79] | 40.76 [37.60, 43.82] |
| Qwen 3.6 Plus | 38.90 [35.62, 42.11] | 37.31 [34.24, 40.31] |

Across all 1,560 noncanonical benchmark goals, the eight-attempt reference is 16.33 [13.98, 18.80]%. The 132 canonical goals have an older three-second solution-density proxy, while current success uses a two-second dwell. We therefore do not substitute that proxy into a supposedly matched full-population baseline.
Allowing the unknown canonical random-search success rates to range anywhere from zero to one bounds the 1,641-task mean at 15.26%–23.30%. These are bounds, not confidence intervals.
This baseline is uniform over the saved valid-placement grids, with each attempted action starting from the same initial world. The bootstrap describes variation across layouts conditional on the saved densities; it does not estimate continuous-placement discretization error. Model retry-only sampling is a separate, informed policy and outperforms this uniform reference. Beating random search does not by itself establish feedback learning; that question uses paired full-feedback versus retry controls.

## No-motion prediction

Predict that each scored dynamic object stays at its initial simulator center and that the tool stays at the supplied placement. Predict no exits. Use the same objects, actions, novel probes and Gaussian PQ rule (sigma = 150 pixels) as each model. This reference uses exact initial geometry and has no perceptual-localization error. It is a heuristic, not a universal chance level.

### Successful chosen actions

These are the Figure 4A successful-attempt populations, with inverse episode-length weights. Differences and intervals pair each model prediction with no-motion on the identical action.

| Model | Successful tasks | Model PQ | No-motion PQ | Model minus no-motion |
|---|---:|---:|---:|---:|
| GPT-5 | 686 | 0.500 [0.458, 0.541] | 0.312 [0.269, 0.355] | 0.188 [0.160, 0.214] |
| Gemini 3.1 Pro | 763 | 0.449 [0.395, 0.505] | 0.328 [0.284, 0.373] | 0.121 [0.075, 0.165] |
| Qwen 3.6 Plus | 683 | 0.454 [0.413, 0.495] | 0.316 [0.272, 0.360] | 0.138 [0.118, 0.161] |
| Gemini Robotics-ER 2 | 100 | 0.585 [0.529, 0.638] | 0.272 [0.220, 0.325] | 0.312 [0.261, 0.366] |

### Terminal novel-action predictions

| Model | Tasks | Model PQ after interaction | No-motion PQ | Model minus no-motion |
|---|---:|---:|---:|---:|
| GPT-5 | 116 | 0.594 [0.538, 0.649] | 0.571 [0.510, 0.631] | 0.023 [-0.039, 0.090] |
| Gemini 3.1 Pro | 116 | 0.497 [0.433, 0.556] | 0.562 [0.500, 0.623] | -0.065 [-0.154, 0.021] |
| Qwen 3.6 Plus | 116 | 0.618 [0.566, 0.668] | 0.547 [0.487, 0.607] | 0.071 [0.016, 0.126] |
| Gemini Robotics-ER 2 | 127 | 0.582 [0.529, 0.633] | 0.559 [0.498, 0.619] | 0.023 [-0.049, 0.096] |

The three main models use the same 116 task IDs but can receive different probe placements, so their no-motion values differ. Robotics-ER 2 uses its separate 127-task probe population. For a fixed probe, no-motion is identical before and after feedback.
Successful chosen-action predictions exceed no-motion in all four models. For terminal novel probes, Qwen has a positive paired interval; GPT-5, Gemini 3.1 Pro and Robotics-ER 2 have intervals spanning zero. This supports an above-reference chosen-prediction result and a model-dependent novel-prediction result, rather than equating a PQ near 0.5 with chance.

## Scope of qualitative prediction

The released schema and primary PQ evaluate terminal object states, spatial endpoints and exit sides. Optional orientation and coarse terminal-location diagnostics exist. The movement and collision events used for goal generation and trajectory characterization are not separately scored prospective event predictions. Appropriate manuscript wording is “terminal object states and spatial endpoints,” without claiming a direct test of which objects will move or collide.

## Files and reproduction

- `summary.json`: exact estimates, intervals, paired contrasts, curves and methods.
- `random_search.csv`: matched model success and model-minus-random contrasts.
- `random_search_task_densities.csv`: the exact task densities, supporting counts and membership.
- `no_motion.csv`: initial/final chosen, successful/failed chosen, pre/post novel and one-rollout novel comparisons.
- `random_search.tex` and `no_motion.tex`: standalone `booktabs` table fragments.

From the repository root:

```bash
python scripts/evaluation/baselines.py --results results --output outputs/baselines
```

The main `python scripts/reproduce.py --output outputs/reproduction` command also regenerates these references.
