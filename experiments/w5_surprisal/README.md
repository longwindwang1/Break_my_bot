# W5 world-model evaluation: prediction, falls, surprisal and change detection

Model: `logs/wm/2026-11-07_22-45-27_w4_wm_nominal/model_fall.pt`, deployed as `export/models/go2_wm.onnx`. The dynamics are trained on normal physics and the fall head is refit on the full randomization (`experiments/w4_wm_ensemble/`). Data: `experiments/w4_wm_data/`.

```
PY=~/miniforge3/envs/env_isaaclab/bin/python
$PY -m models.world_model.evaluate --run logs/wm/2026-11-07_22-45-27_w4_wm_nominal --checkpoint model_fall.pt \
    --data logs/wm_data --out experiments/w5_surprisal
$PY -m models.world_model.evaluate --run logs/wm/2026-11-07_21-56-08_w4_wm \
    --data logs/wm_data --out experiments/w5_surprisal/ablation_full_data_model
```
Outputs:
- `results.json`: all numbers below and more, including the full calibration grid.
- `calibration.yaml`: the `world_model.surprisal` block, copied into `configs/robot_interface.yaml`.
- `surprisal_events.png`, `multi_step.png`.

The run takes about 12 minutes.

## 1. One-step prediction
Held-out envs of the dynamics training split (normal physics), transitions without a push and not at an episode start:

| state term | RMSE | RMSE if s stays | z std | \|z\| < 1 | \|z\| < 2 |
|---|---:|---:|---:|---:|---:|
| base_lin_vel (m/s) | 0.0143 | 0.0320 | 0.88 | 78.8 % | 97.3 % |
| base_ang_vel (rad/s) | 0.144 | 0.351 | 0.89 | 79.8 % | 97.1 % |
| projected_gravity | 0.0017 | 0.0080 | 0.86 | 80.2 % | 97.2 % |
| joint_pos (rad) | 0.0099 | 0.0628 | 0.89 | 83.8 % | 97.8 % |
| joint_vel (rad/s) | 0.731 | 2.105 | 0.92 | 84.7 % | 97.5 % |

- **Calibration.** The residuals are peaked with heavy tails: the 1σ coverage of 79–85 % is above the Gaussian 68.3 %, and z std is about 0.9. So the bulk is slightly under-confident and the tails are slightly heavier than Gaussian.
- **Consistency.** The clean split agrees (z std 0.86, 97.7 % within 2σ). It holds for both policies and every terrain type; random_rough is the least calibrated (z std 0.95, 96.7 %).
- **On ice the model is miscalibrated, by design** (full-randomization val envs):
  - μ < 0.3: z std 1.59, 92.0 % within 2σ;
  - μ ≥ 0.6: z std 0.93, 97.3 % within 2σ.

## 2. Open-loop rollouts (ghost path)
4 096 windows of 50 valid steps, held-out envs. The model's own states are fed back. Actions and height scans are the recorded ones, so this is optimistic: the policy's real reactions are replayed.

| steps ahead | base_lin_vel RMSE | joint_pos RMSE | base xy error (mean / p90) | distance travelled | yaw error |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.012 | 0.009 | 0.000 m | 0.012 m | 0.000 |
| 10 | 0.052 | 0.044 | 0.004 / 0.008 m | 0.12 m | 0.010 |
| 25 | 0.099 | 0.066 | 0.024 / 0.047 m | 0.31 m | 0.040 |
| 50 | 0.207 | 0.104 | 0.107 / 0.208 m | 0.61 m | 0.111 |

- Joint predictions stay well below the "state stays" baseline throughout.
- Base velocity and attitude errors exceed that baseline after about 40 steps (`multi_step.png`).
- A 1 s ghost path is off by about 0.1 m (18 % of the distance). Beyond 1 s it needs care.

## 3. Fall prediction
AUROC / average precision of the fall heads, against a baseline that only uses the base tilt |g_xy|. Steps with censored labels and noisy-action envs are excluded.

| split | h = 25 steps (0.5 s): AUROC model / tilt | AP model / tilt | h = 50: AUROC model / tilt |
|---|---|---|---|
| full-randomization val envs | 0.997 / 0.947 | 0.68 / 0.25 | 0.972 / 0.900 |
| clean | 0.995 / 0.940 | 0.60 / 0.19 | 0.978 / 0.912 |
| events | 0.997 / 0.955 | 0.79 / 0.22 | 0.973 / 0.901 |
| pits 15–30 cm, half deceived (out of distribution) | 0.888 / 0.770 | 0.27 / 0.07 | 0.827 / 0.717 |

- **Lead time.** We used a threshold at 1 % false positives on the h = 50 head.
  - Clean, events and val: 99–100 % of falls are flagged beforehand, 97–99 % of them at least 10 steps (0.2 s) ahead. The median lead is 43–50 steps; 50 is the horizon cap.
  - Deep hidden pits: 61 % flagged and 53 % at least 0.2 s ahead.
- **By policy.** v1 is slightly harder to predict than v2 (clean h25 AUROC: 0.984 vs 0.998).

## 4. Surprisal and change detection
Two detectors with a shared false-alarm budget of 1 per 30 000 scored steps (10 min of walking), measured on the clean split.
- **Spike (pushes).** Fires when max |z| of base_lin_vel x, y exceeds `spike_z`. `spike_z` is the clean 1 − 1/60 000 quantile: 16.1 for v1 and 16.3 for v2.
- **CUSUM (persistent changes).** For each configuration we took the smallest h such that both detectors together meet the budget on clean half A (env i % 2 == 0). We checked it on half B.
- **Grid.** 64 configurations: statistic {norm, nll} × weights {all, group, base, joints} × clip {9, 25} × k {0.1, 0.25, 0.5, 1}.
- **Selection rule.** Highest mean detection within 2 s over μ → 0.1, μ → 0.05, one foot μ → 0.05, and pushes of 0.5 and 1.0 m/s, among configurations whose half-B false-alarm rate is at most 2× the target.
- **Chosen.** `norm`, joints weights (joint_pos and joint_vel), clip 25, k = 1. h = 33.3 for v1 and 68.2 for v2. False alarms on half B are 1.28× the target for v1 and 0.61× for v2.
- Scoring skips the first 25 steps of every episode. An alarm within 50 steps before a fall is counted as a hit, not a false alarm.

Detection in the events split. At step 200 (4 s) every env got one event; there are about 195 (v1) and 390 (v2) envs per cell.

| event | v1 ≤ 1 s | v1 ≤ 2 s | v1 median delay | v2 ≤ 1 s | v2 ≤ 2 s | v2 median delay | x 1–3 s after (v1 / v2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| none (control) | 0.0 % | 0.0 % | – | 0.0 % | 0.0 % | – | −0.03 / −0.06 |
| all feet μ → 0.3 | 13.4 % | 22.7 % | 54 | 1.8 % | 4.4 % | 86 | +0.73 / +0.33 |
| all feet μ → 0.1 | 87.7 % | 90.3 % | 14 | 61.2 % | 80.7 % | 34 | +2.20 / +2.24 |
| all feet μ → 0.05 | 95.9 % | 96.4 % | 9 | 93.6 % | 95.9 % | 20 | +2.52 / +3.66 |
| one foot μ → 0.05 | 21.0 % | 48.2 % | 62 | 4.9 % | 20.9 % | 85 | +1.08 / +1.14 |
| push 0.5 m/s | 97.4 % | 97.4 % | 1 | 99.2 % | 99.2 % | 1 | −0.00 / −0.07 |
| push 1.0 m/s | 100 % | 100 % | 1 | 100 % | 100 % | 1 | −0.02 / −0.07 |

Delays are in policy steps (50 = 1 s).
- **Pushes** are caught by the spike detector in the step they happen.
- **Ice is persistent.** Surprisal stays 2–4 σ above the clean level for as long as the robot is on ice (`surprisal_events.png`). After the first alarm the CUSUM fires again 1–2 times per second. So a "cautious while alarmed" rule keeps the robot cautious on ice and releases it afterwards.
- **v1 vs v2.** v1's ice alarms come earlier: it slips more, and its threshold is lower. v2 needs more evidence (h 68 vs 33) because its clean baseline has longer surprisal excursions.
- **Weak cases.** μ → 0.3 and a single icy foot are weak changes for these policies and are detected less often.

**Hidden pits** (pit terrain with 15–30 cm blocks, after step 200). Alarm rate while the height scan reports flat ground, compared with envs whose scan stays truthful:

| | deceived | truthful scan |
|---|---:|---:|
| v1 | 0.24 alarms/s | 0.05 alarms/s |
| v2 | 0.17 alarms/s | 0.02 alarms/s |

## 5. Ablation: dynamics trained on the full randomization
`ablation_full_data_model/` applies the same evaluation and selection to the model trained on `w4_train`, which includes ice, friction changes and pushes.
- **Best configuration:** norm, joints, clip 9, k = 1.
- **Detection within 2 s:**
  - pushes 98.5–100 %;
  - μ → 0.1: v1 3.6 %, v2 0.5 %;
  - μ → 0.05: v1 16.5 %, v2 0.5 %;
  - one foot μ → 0.05: 0 / 0.5 %.
- **Key-event mean:** 0.42, against 0.83 for the deployed model.
- **Hidden pits:** 0.12–0.13 alarms/s deceived vs 0.02–0.04 truthful.
- **Why.** This model has learned ice dynamics, so ice does not surprise it: the score after an ice event rises only +0.5 to +0.8 σ.

## 6. Unity reference traces
`export/export_wm_onnx.py --reference ...` wrote two traces in `experiments/w3_reference/`. Each holds the full `wm_in`, all outputs, z, S, x, spike, g and alarm per step.
- `go2_wm.json` (clean v2 rollout, μ 1.0): no alarm; g peaks at 9.9 against h 68.
- `go2_wm_mudrop.json`: same rollout setup (`go2_v2_mudrop.npz`, seed 0, no resets), with foot μ set to 0.05 at step 250. The first alarm comes at step 262, 0.24 s after the drop. Then 11 more alarms follow in the remaining 4.8 s on ice.

## Open points
- **Isaac-only calibration.** Everything above was calibrated in Isaac Lab. In Unity, the per-dim z bias/scale and h must be re-estimated on clean walking (`docs/unity_handoff.md` §11.4). That gap is part of experiment 3.
- **Closed loop not tested yet.** Whether an alarm actually helps (cautious gait for v1, history refresh for v2) is the W6–8 closed-loop test of experiment 1. Here the evaluation is offline only.
- **Weaker detections.** Single-foot ice and μ 0.3 are weak signals. Mixing base and joint terms, or a longer CUSUM memory (smaller k), trades detection of these against delay on strong ice; see the grid in `results.json`.
