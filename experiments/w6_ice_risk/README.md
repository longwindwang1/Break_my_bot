# W6 terrain-risk model, phase 2: visible and hidden ice

The question is how the work should be split for ice that the robot steps on:
- **Visible ice** is drawn in the level. The terrain-risk ("vision") model gets it as an extra input channel and can warn before the robot gets there.
- **Hidden ice** looks like ground. Only the proprioceptive world model's surprisal (W5) can notice it, after a foot is on it.

Data: `experiments/w6_ice_data/`, with ice realized as per-foot friction × 0.1 from an ice map.

**Model.** This is the phase-1 network plus an `ice` input channel.
- The channel is 1 where a grid point lies on visible ice. It gets its own first-layer kernel, with no zero-sum constraint and no bias.
- The height kernel is still zero-sum, so a constant height offset still leaves the output unchanged.
- With the ice input all zero, the model is a height-only model.
- Training data: the 3 geo splits (no ice) plus the 3 ice splits (all ice visible). That is 32.2M frames: 110k positives at 0.5 s and 220k at 1 s.
- The same recipe as phase 1 took 5.2 minutes. The best mean validation AUROC was 0.906.
- After Platt scaling, the predicted probabilities match the validation base rate: 0.33 % at 0.5 s and 0.68 % at 1 s.

```
PY=~/miniforge3/envs/env_isaaclab/bin/python; D=logs/wm_data
$PY -m models.terrain_risk.train --data $D/w6_geo_train $D/w6_geo_train2 $D/w6_geo_train3 $D/w6_ice_train $D/w6_ice_train2 $D/w6_ice_train3 --name w6_ice_risk
$PY -m models.terrain_risk.evaluate_ice --run logs/terrain_risk/2026-11-08_17-11-33_w6_ice_risk \
    --phase1_run logs/terrain_risk/2026-11-08_01-11-04_w6_risk3 --test $D/w6_ice_test --geo_test $D/w6_geo_test \
    --train $D/w6_geo_train $D/w6_geo_train2 $D/w6_geo_train3 $D/w6_ice_train $D/w6_ice_train2 $D/w6_ice_train3 \
    --wm_run logs/wm/2026-11-07_22-45-27_w4_wm_nominal --out experiments/w6_ice_risk
# closed loop: see below
```

## Held-out terrain with half of the ice hidden (`ice_test`, seed 900)
**Fall classes.** A fall is classed by the ice under the feet in the second before it:
- on visible ice: 145 (v1 76, v2 69);
- on hidden ice: 115 (v1 62, v2 53);
- no ice (geometry): 268.

A positive frame takes the class of the fall it precedes.

### Per frame
The table gives the AUROC at 0.5 s. Each row compares that class's positives with all negatives. 95 % CIs are from a bootstrap over envs.

| falls on | ice model, visible ice shown (game) | ice model, all ice hidden | ice model, all ice shown | phase-1 model | world-model fall head |
|---|---:|---:|---:|---:|---:|
| visible ice | **0.909** (0.890–0.927) | 0.790 | 0.911 | 0.750 (0.706–0.795) | 0.978 |
| hidden ice | 0.784 (0.745–0.820) | 0.803 | *0.919* | 0.747 | 0.984 |
| no ice | 0.899 (0.881–0.912) | 0.913 | 0.889 | 0.907 (0.890–0.922) | 0.956 |
| all | 0.877 (0.863–0.888) | 0.855 | 0.901 | 0.829 (0.811–0.846) | 0.968 |

**Visible ice.** The ice channel does the job: visible-ice falls go from 0.75–0.79 (blind) to 0.91. The gain is largest for v1, which slips more on ice.

| | ice shown | ice hidden | phase 1 |
|---|---:|---:|---:|
| v1, visible-ice falls | 0.904 | 0.708 | 0.674 |
| v2, visible-ice falls | 0.906 | 0.855 | 0.826 |

**On ice cells the model raises the predicted risk** by these factors at v = 0.3–1.0 m/s:
- v1: 7.5–8.8 ×. Its mean risk on ice is 0.013–0.026.
- v2: 2.3–4.5 ×. Its mean risk on ice is only 0.003–0.005, against about 0.05 at tall stair edges.

So for v2, ice is a much smaller hazard than geometry. This fits its friction training, where it survives μ 0.1 90 % of the time (`experiments/v2_eval/`).

**Hidden ice.**
- The model is blind to hidden ice, as intended: 0.78 for hidden-ice falls, against 0.92 if the same patches were shown.
- The world-model fall head, which sees the robot's state, scores 0.98 on both kinds of ice.

**No regression on geometry.**
- On `ice_test`, geometry falls score 0.899 against 0.907 for phase 1, and the CIs overlap.
- On `geo_test` (no ice): 0.907 (0.895–0.920) against phase 1's 0.906 (0.892–0.921) at 0.5 s, and 0.906 against 0.903 at 1 s.

### Lead time
These numbers use the phase-1 definition: a 5 % false-positive rate on frames with no fall within 3 s, both policies pooled, and the warning run must last until the fall.

| score | visible-ice falls: warned at fall / ≥ 0.5 s / ≥ 1 s | hidden-ice falls | no-ice falls |
|---|---|---|---|
| risk ahead, ice model (visible shown) | 21 % / 7 % / 5 % | 11 % / 1 % / 1 % | 33 % / 16 % / 14 % |
| risk ahead, same model, ice hidden | 21 % / 8 % / 6 % | 11 % / 1 % / 1 % | 35 % / 18 % / 14 % |
| risk ahead, phase-1 model | 19 % / 5 % / 3 % | 12 % / 3 % / 2 % | 32 % / 18 % / 12 % |
| world-model fall head | 94 % / 73 % / 33 % | 95 % / 76 % / 30 % | 88 % / 58 % / 28 % |

Showing the ice does **not** produce earlier warnings at this operating point:
- The pooled 5 % threshold (risk ahead 0.087) is set by the geometric hot spots.
- The risk ice adds, 0.005–0.05, mostly stays below it.

The closed-loop trigger below uses per-policy thresholds that are 2–7 times lower.

### World-model surprisal (W5 parameters from robot_interface.yaml, unchanged)
**Falls with a CUSUM alarm before them:**

| falls on | within 0.5 s | within 1 s | within 2 s |
|---|---:|---:|---:|
| visible ice | 21 % | 46 % | 61 % |
| hidden ice | 27 % | 45 % | 57 % |
| no ice | 4 % | 11 % | 15 % |

**Ice contacts.** A contact is the first foot on a patch after at least 1 s off ice.
- An alarm follows within 1 s for 33 % of visible-ice contacts (1 923) and 31 % of hidden-ice contacts (1 738).
- The world model does not see the ice channel, so the two kinds of ice behave the same for it.
- Most contacts end without a slip that matters.

**False alarms.** Off ice and with no fall within 3 s, there are 2.9 alarms per 10 min. The W5 target on clean terrain was 1, and this terrain is rougher.

**Who covers what.**
- On hidden ice, surprisal fires before about half of the falls. That is 4 times its rate before geometry falls, and the vision model cannot help there.
- On visible ice, both models carry information. The vision model's information is available before the robot arrives.

`heatmap_ice_test.png`: a 24 × 16 m region of `ice_test` with the most ice falls. There is one row per policy: height with the ice, risk with visible ice shown, and risk with all ice hidden, with falls by class. For v1, the visible patches light up; for v2 they barely do.

## Closed loop: slow down where the model sees risk ahead
**Setup.** `isaac/eval_risk_slowdown.py --terrain ice_test`:
- **Terrain and ice.** The `ice_test` terrain with its ice physics. Half of the patches are hidden in the "map" view.
- **Runs.** 1 023 envs (341 v1, 682 v2), 20 s each, with rollout seeds 0 and 1.
- **Commands and gait.** The same mostly-forward commands and cautious gait (|v| ≤ 0.3 m/s) as phase 1, with a 0.2 s update and a 1 s hold.
- **Thresholds.** Set on the training splits' held-out tiles (`results.json` → `trigger_thresholds`, `trigger_thresholds_ice_excess`). They are the values exceeded on 10 % or 20 % of frames.
- **Ice view** (what the model is shown):
  - `all`: every patch;
  - `none`: no patch;
  - `map`: the visible half, as in the game.
- **Trigger** (what is compared with the threshold):
  - `total`: the risk ahead;
  - `ice excess`: the risk ahead with the ice shown minus the risk with no ice shown, i.e. the risk the ice adds.
- **Fall classes** are counted as in the per-frame evaluation: by the ice under the feet in the last second.

```
PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python isaac/eval_risk_slowdown.py --terrain ice_test --mode none|always|risk|random \
    [--trigger 10pct|20pct --ice_view all|none|map --risk_kind total|ice_excess | --cautious_frac 0.4391] --seed 0|1 \
    --risk_run logs/terrain_risk/2026-11-08_17-11-33_w6_ice_risk --risk_eval experiments/w6_ice_risk/results.json \
    --v2 <student ckpt> --v1 <v1 ckpt> --out experiments/w6_ice_risk/slowdown.jsonl
```

Falls per 100 m, both seeds pooled. "Ice falls" are falls on visible plus hidden ice.

**v2**

| arm | cautious time | falls / 100 m | vs none | geometry falls | ice falls | mean speed |
|---|---:|---:|---:|---:|---:|---:|
| none | 0 % | 1.69 (216 falls) | | 1.15 | 0.54 | 0.47 m/s |
| always cautious | 100 % | 1.98 | +17 % | 1.14 | 0.83 | 0.22 |
| random, 44 % of the time | 43 % | 1.66 | −2 % | 1.11 | 0.55 | 0.36 |
| total, 20 %, ice shown (all) | 44 % | **1.14** | **−32 %** | 0.66 | 0.48 | 0.38 |
| total, 20 %, ice hidden (none) | 39 % | 1.28 | −24 % | 0.76 | 0.52 | 0.39 |
| total, 20 %, map (game) | 41 % | **1.13** | **−33 %** | 0.72 | 0.41 | 0.38 |
| total, 10 %, ice shown | 26 % | 1.33 | −21 % | 0.91 | 0.42 | 0.42 |
| total, 10 %, ice hidden | 23 % | 1.07 | −37 % | 0.66 | 0.40 | 0.43 |
| ice excess, 20 %, ice shown | 35 % | 1.63 | −3 % | 1.12 | 0.51 | 0.38 |
| ice excess, 10 %, ice shown | 19 % | 1.61 | −4 % | 1.12 | 0.49 | 0.43 |

**v1**

| arm | cautious time | falls / 100 m | vs none | geometry falls | ice falls | mean speed |
|---|---:|---:|---:|---:|---:|---:|
| none | 0 % | 3.04 (186 falls) | | 1.52 | 1.52 | 0.45 m/s |
| always cautious | 100 % | 4.86 | +60 % | 3.47 | 1.39 | 0.20 |
| random, 44 % of the time | 43 % | 3.88 | +28 % | 2.08 | 1.79 | 0.36 |
| total, 20 %, ice shown | 47 % | 2.52 | −17 % | 1.61 | 0.91 | 0.36 |
| total, 20 %, ice hidden | 30 % | 2.94 | −3 % | 1.54 | 1.40 | 0.39 |
| total, 20 %, map | 39 % | 2.89 | −5 % | 1.68 | 1.21 | 0.38 |
| total, 10 %, ice shown | 30 % | 2.56 | −16 % | 1.35 | 1.21 | 0.39 |
| total, 10 %, ice hidden | 18 % | 2.78 | −9 % | 1.25 | 1.53 | 0.43 |
| **ice excess, 20 %, ice shown** | 34 % | **2.54** | **−16 %** | 1.50 | **1.04** | 0.37 |
| ice excess, 10 %, ice shown | 23 % | 2.90 | −5 % | 1.70 | 1.20 | 0.40 |
| ice excess, 20 %, map | 20 % | 3.01 | −1 % | 1.51 | 1.49 | 0.40 |

**Noise.** With 120–220 falls per arm, the Poisson uncertainty of a rate ratio is about ±15–20 %. The two seeds agree for the main arms:
- v2, total 20 %, all: 1.15 and 1.14, against 1.74 and 1.64 for none;
- v1, ice excess 20 %, all: 2.63 and 2.45, against 2.72 and 3.36.

**v2: slowing before risk helps; slowing on ice does not.**
- Total-risk triggers cut falls per 100 m by 21–37 % at 14–19 % less speed. Random slowdown (−2 %) and always cautious (+17 %) do not help.
- The gain comes from geometry: geometry falls drop from 1.15 to 0.66–0.91 per 100 m.
- Ice falls barely move: 0.40–0.52 against 0.54. Whether the ice is shown makes no consistent difference (all vs none: −32 % vs −24 % at 20 %, −21 % vs −37 % at 10 %).
- The ice-excess trigger does nothing for v2 (−3 % to −4 %).
- This matches phase 1 and `experiments/v2_eval/`, where the cautious gait did not raise the student's survival at μ 0.1.

**v1: slowing before visible ice helps; slowing near geometry does not.**
- With all ice shown, the total trigger at 20 % cuts ice falls per 100 m from 1.52 to 0.91 (−40 %). With the same trigger and the ice hidden, they stay at 1.40.
- The total trigger also slows v1 near geometry, where it does not help (geometry falls 1.61 against 1.52).
- The ice-excess trigger slows v1 only where visible ice adds risk.
  - At 20 % it gives the same −16 % overall with less cautious time (34 % against 47 %). Ice falls drop by 32 % and geometry falls are unchanged.
  - On terrain without ice it never fires, so it cannot cause the phase-1 problem, where slowed v1 falls more.
- **Caveat.** In the game-like map view, only half of the ice is shown, and the v1 gains shrink to −1 % to −5 %, which is within noise. Visible-ice falls were not reduced there (0.75 against 0.78 per 100 m). The v1 benefit is real but modest and needs most of the ice to be visible.

**No regression on terrain without ice.** On `geo_test` (`slowdown_geo_test.jsonl`):
- The none arm reproduces phase 1's numbers exactly.
- The new model's v2 total trigger at its 20 % threshold (0.0125) gives 0.82 against 1.44 falls per 100 m (−43 %), with 38 % cautious time and 0.41 m/s.
- The phase-1 model gave −46 % with 46 % cautious time.

**Deployed rule** (`robot_interface.yaml terrain_risk.slowdown`):
- `trigger: {v1: ice_excess, v2: total}`;
- `threshold: {v1: 0.00799, v2: 0.0125}`, both at the 20 % operating point.

## Conclusions
1. **Visible ice is a job for the vision model; hidden ice is a job for the world model.**
   - The ice channel lifts visible-ice fall prediction from 0.75–0.79 to 0.91 AUROC.
   - The model stays blind to hidden ice, as intended.
   - The world model's surprisal fires before about half of the falls on either kind of ice, against 11 % before geometry falls.
   - Geometry prediction is unchanged.
2. **The response depends on the policy.**
   - v2 is robust on ice, so its slowdown keeps targeting geometry: −32 % to −43 % falls per 100 m.
   - v1 is fragile on ice and clumsy when slow. Slowing it only where visible ice adds risk cuts its ice falls by a third, without the geometry penalty.
3. **A per-frame threshold at a fixed false-positive rate is not the right yardstick.**
   - The pooled 5 % lead-time metric shows no earlier warnings.
   - The closed loop with per-policy thresholds shows the practical effect.

## Open points
- **Unity:** ice colliders (visible / hidden), the ice flag in the ray grid, and the second ONNX run for v1's ice excess (handoff §12.5).
- **Hidden ice in the closed loop:** a surprisal-triggered cautious gait for v2. This is experiment 1, the next milestone. The cautious gait did not help the v2 student on ice, so that experiment should also look at clearing the history.
- **Other ice friction:** ice friction is fixed at 0.1. Other values would need new data or a friction channel.
