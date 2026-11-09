# W6 terrain-risk ("vision") model: phase 1, terrain height only

The question the model answers is: if the robot's base were at this cell, facing along the map's x axis, under this velocity command and policy, how likely is a fall within 0.5 s (25 steps) or 1 s (50 steps)?

**Input**
- Terrain height from vertical ray casts on a 10 cm grid. There is no RGB, so Isaac Lab and Unity see identical values.
- A condition vector: the command and the policy one-hot.

**Network.** Fully convolutional, using unpadded dilated 3 × 3 convolutions with a 17 × 17 receptive field (1.6 m). About 30k parameters.
- A 17 × 17 patch gives one cell.
- The 47 × 37 egocentric window gives a 31 × 21 risk map over x ∈ [−0.5, 2.5] m and y ∈ [−1, 1] m.
- A whole level gives a whole-level map.
- The first layer's height kernels sum to zero, so the output does not change under a constant height offset.
- The spec is in `configs/robot_interface.yaml` → `terrain_risk`. The model is `export/models/go2_terrain_risk.onnx`.

```
PY=~/miniforge3/envs/env_isaaclab/bin/python
$PY -m models.terrain_risk.train --data logs/wm_data/w6_geo_train logs/wm_data/w6_geo_train2 logs/wm_data/w6_geo_train3 --name w6_risk3
$PY -m models.terrain_risk.evaluate --run logs/terrain_risk/2026-11-08_01-11-04_w6_risk3 \
    --train logs/wm_data/w6_geo_train logs/wm_data/w6_geo_train2 logs/wm_data/w6_geo_train3 \
    --test logs/wm_data/w6_geo_test --wm_run logs/wm/2026-11-07_22-45-27_w4_wm_nominal --out experiments/w6_terrain_risk
# closed loop (one process per arm; rollout seeds 0 and 1)
PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python isaac/eval_risk_slowdown.py --mode none|always|risk --trigger 10pct|20pct|random --cautious_frac 0.4033 \
    --seed 0 --risk_run logs/terrain_risk/2026-11-08_01-11-04_w6_risk3 --risk_eval experiments/w6_terrain_risk/results.json \
    --v2 <student ckpt> --v1 <v1 ckpt> --out experiments/w6_terrain_risk/slowdown.jsonl
```

## Training
**Data** (`experiments/w6_geo_data/`):
- Three terrain seeds (600 tiles in total, 1 940 falls).
- One sample per recorded frame: the 17 × 17 patch around the base, rotated to its yaw, plus the command and policy.
- Labels: the world model's censored "fall within h steps" labels.
- Excluded: noisy-action envs, first frames and deceived frames.
- Validation: 10 % of the tiles of every split, held out by tile, not by env.
- 16.1M training frames: 44 524 positives at 0.5 s, 89 056 at 1 s.

**Optimization:** BCE; positives oversampled to 10 % of every batch; left-right mirror augmentation (patch flipped, vy and wz negated); AdamW with lr 2e-3 and weight decay 1e-3; 8 000 iterations of 4 096 samples. It takes 4.3 minutes.
- Best validation AUROC (0.910 at 0.5 s, 0.911 at 1 s) came at iteration 2 750.
- Platt scaling on the validation tiles maps the oversampled logits back to probabilities. On held-out terrain the mean predicted probability equals the base rate: 0.27 % vs 0.27 % at 0.5 s, and 0.54 % vs 0.56 % at 1 s.

**Data ablation.** With one terrain seed (639 falls) and 20 % oversampling, the model memorizes:
- validation AUROC peaks at 0.87 after 1 000 iterations;
- it then falls to 0.78 while the training loss keeps dropping.

Three seeds and 10 % oversampling fixed it.

## Results on held-out terrain (geo_test, seed 700)
### Per-frame fall prediction
Every 5th frame; 594k frames, 1 615 positives at 0.5 s.

| score | AUROC 0.5 s (95 % CI) | AP 0.5 s | AUROC 1 s |
|---|---:|---:|---:|
| **terrain-risk model** | **0.906** (0.891–0.918) | 0.035 | **0.903** |
| steepest step in the patch | 0.835 (0.816–0.852) | 0.011 | 0.835 |
| height range in the patch | 0.793 | 0.010 | 0.797 |
| tile type × difficulty fall rate (from the training splits) | 0.808 (0.785–0.830) | 0.011 | 0.811 |
| proprioceptive world model, fall head (knows the robot's state) | 0.983 | 0.558 | 0.957 |

- The CIs are from a bootstrap over envs.
- By policy, at 0.5 s: v1 0.896 and v2 0.910. The rule baselines score 0.77–0.85.
- **Comparison with the rule baselines.** Seeing the terrain shape beats simple geometric rules and the terrain-type table, and the confidence intervals do not overlap.
- **Comparison with the world model.** Per frame, the terrain model is far weaker. It gives a prior for "this place is dangerous". The world model sees the robot actually stumbling.

### Lead time before falls
Both scores use the same 5 % false-positive rate, measured on frames with no fall within 3 s. Lead time is the length of the above-threshold run that ends at the fall.
- **Risk ahead** (the max over the 1.2 m corridor along the command, from the egocentric window): 27 % of falls are flagged at the moment of the fall, 13 % at least 0.5 s ahead and 7 % at least 1 s ahead.
- **World-model fall head:** 91 %, 75 % and 42 %.

So the terrain model did **not** warn earlier at a fixed false-positive rate. Dangerous terrain (stairs, deep pits) is crossed safely most of the time, so a sharp threshold on terrain alone flags far more places than falls happen.

### `heatmap_test.png`
Risk at 0.5 s over a 24 × 16 m corner of the test terrain, heading +x, v = 0.6 m/s, v2.
- The step edges of the tall pyramid stairs light up. The edges approached head-on are brightest.
- The recorded falls sit on them.
- Slopes, rough ground and the shallow stairs stay dark.

## Closed loop: slow down where the model sees risk ahead
**Setup** (`isaac/eval_risk_slowdown.py`):
- geo_test terrain, 1 023 envs (341 v1, 682 v2), 20 s each, two rollout seeds.
- Mostly-forward commands: vx 0.3–1.0, |vy| ≤ 0.2, |wz| ≤ 0.5.
- Every 0.2 s, the risk ahead at 0.5 s is computed with the original command. If it exceeds the policy's threshold, the cautious gait holds for 1 s: |v| ≤ 0.3 m/s, |wz| ≤ 0.5, written into the policy observation.
- The thresholds were set on the training splits' held-out tiles, as the values that trigger on 10 % or 20 % of frames:
  - v1: 0.0297 / 0.0129;
  - v2: 0.0228 / 0.0086.
- **Random:** the same fraction of cautious time as the 20 % arm (40 %), at random moments.

| policy | arm | cautious time | falls / 100 m | vs none | mean speed | 20 s success |
|---|---|---:|---:|---:|---:|---:|
| v2 | none | 0 % | 1.44 (197 falls) | | 0.50 m/s | 87.8 % |
| v2 | risk, 10 % threshold | 28 % | **0.92** | **−36 %** | 0.43 m/s | 93.0 % |
| v2 | risk, 20 % threshold | 46 % | **0.78** | **−46 %** | 0.38 m/s | 94.6 % |
| v2 | random, 40 % of the time | 40 % | 1.28 | −11 % | 0.39 m/s | 91.1 % |
| v2 | always cautious | 100 % | 1.34 | −7 % | 0.22 m/s | 94.6 % |
| v1 | none | 0 % | 1.33 | | 0.47 m/s | 89.6 % |
| v1 | risk, 10 % threshold | 22 % | 1.26 | −5 % | 0.43 m/s | 90.6 % |
| v1 | risk, 20 % threshold | 36 % | 1.71 | +29 % | 0.39 m/s | 89.0 % |
| v1 | random, 40 % of the time | 40 % | 1.39 | +5 % | 0.36 m/s | 91.3 % |
| v1 | always cautious | 100 % | 3.24 | +144 % | 0.19 m/s | 89.6 % |

- **v2 gains from slowing in the right places.** Risk-triggered slowdown cuts falls per distance by 36–46 % and costs 14–24 % of speed.
  - Spending the same cautious time at random barely helps (−11 %), and always walking cautiously does not help per distance (−7 %).
  - The per-seed numbers agree: 0.93 / 0.90 and 0.81 / 0.76, against none 1.35 / 1.53.
  - With 81–197 falls per arm, the Poisson uncertainty is about ±15–22 %. The 20 % arm vs none is a rate ratio of 0.54.
- **v1 does not benefit.** v1 is clumsy at low speed: always cautious triples its falls per metre. So slowing v1 down only helps at the 10 % threshold, by a margin within noise. In the game, the slowdown should be used with v2 only.
- **Fall rate per time vs per distance.** The "20 s success" column rewards walking less. Falls per 100 m is the fair comparison.

## Open points
- **Visible vs hidden ice (phase 2):** done; see `experiments/w6_ice_risk/`. The deployed model, `go2_terrain_risk.onnx`, is now the ice model, and its slowdown rule replaced the thresholds above: v2 total 0.0125, v1 ice excess 0.00799.
- **Unity:**
  - Calibration: the thresholds were set in Isaac Lab, and player-built levels and Unity's slower gait are a distribution shift, so recheck there.
  - Window size: the egocentric window only covers forward commands.
- **Lead-time metric:** it requires the warning to last until the fall. A trigger that fires and holds earlier, as in the closed loop, is what actually helps.
