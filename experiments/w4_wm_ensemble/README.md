# W4 world-model ensemble training

The world model is an ensemble of 5 Gaussian MLPs (`models/world_model/ensemble.py`).
- **Input** `wm_in` is 471-d: its own 6-frame proprio history, the current action, the height scan, and a policy one-hot.
- **Each member** is a 469 → 512 → 512 → 512 SiLU trunk with two heads:
  - mean and log-variance of (Δs 33, base displacement 4);
  - a fall head (3 horizons) on the detached trunk features plus the one-hot.
- **Size:** about 4.3M parameters in total; the exported ONNX is 16.6 MB.
- **Loss** (`models/world_model/train.py`):
  - 1-step Gaussian NLL;
  - plus the K = 5 open-loop MSE of the model's own rollouts, each k-step term divided by k (scan held at step t, recorded actions);
  - plus the fall BCE (censored labels masked, pos_weight ≤ 100);
  - plus the PETS log-variance bound penalty, scaled to the mean NLL.
- **Optimizer:** AdamW, lr 1e-3, cosine schedule, 1024 windows per member per iteration, 20 passes, seed 0.

Two models were trained on different data:

| run | dynamics data | wall | best iteration | val NLL | val 1-step MSE | val K-step term | use |
|---|---|---|---|---|---|---|---|
| `logs/wm/2026-11-07_22-45-27_w4_wm_nominal` | `w4_train_nominal`: normal physics only (μ 0.6–1.25, no pushes) | 37.0 min | 102 000 | −4.251 | 0.099 | 0.119 | **deployed** (`export/models/go2_wm.onnx`, with the refit fall head) |
| `logs/wm/2026-11-07_21-56-08_w4_wm` | `w4_train`: full randomization (μ 0.05–1.25 changing mid-episode, pushes) | 36.9 min | 108 000 | −3.903 | 0.133 | 0.147 | ablation |

```
PY=~/miniforge3/envs/env_isaaclab/bin/python
$PY -m models.world_model.train --data logs/wm_data/w4_train_nominal --name w4_wm_nominal
$PY -m models.world_model.refit_fall --run logs/wm/2026-11-07_22-45-27_w4_wm_nominal --data logs/wm_data/w4_train
$PY -m models.world_model.train --data logs/wm_data/w4_train --name w4_wm          # ablation
```
Each run directory has `model.pt`, `last.pt`, `history.json`, `run_info.json` (args, git, data, sha of the yaml) and TensorBoard logs. Both ran on the RTX 4070 laptop at 51.5 it/s, with 3.8 GB of data held on the GPU.

## Why the deployed dynamics never saw ice
The first model was trained on the full randomization, where ice and friction changes are everywhere. With its 6-frame history it learned to predict walking on ice well enough that ice was no longer surprising.
- At the target false-alarm rate, the best detector configuration caught only 0.5–17 % of μ → 0.1 / 0.05 drops within 2 s. Pushes were still caught: 98.5–100 %.
- Trained on normal physics only, the same detector catches 81–96 % (`experiments/w5_surprisal/`). There, surprisal stays high for as long as the robot is on ice.
- This is the intended semantics: the world model encodes how the world normally behaves, and surprisal measures deviation from that.
- Cost: on ice the predictions are miscalibrated by design. On the full-randomization val envs, z std is 1.59 on μ < 0.3 vs 0.93 on μ ≥ 0.6. So ghost paths on ice will be off, and `disagreement` / surprisal show that.

## Fall head refit
The fall head has to know about falls on ice and after pushes, which the nominal data does not contain. `refit_fall.py`:
- freezes the trunk and dynamics heads and re-initializes the fall head;
- trains it on the `w4_train` train envs (falls in all three horizons: 5 680 / 14 200 / 28 153 positive steps);
- stops on the val-env BCE (patience 5 × 1000 iterations).

Best val BCE was 0.191 after 2 000 iterations. Val AUROC was 0.998 / 0.997 / 0.97 for h = 10 / 25 / 50 steps. The result is written to `model_fall.pt`.

During joint training, the fall head overfits: val BCE climbs from 0.14 to 0.33 while train BCE falls below 0.05. The ranking stays good (AUROC about 0.97–0.999). The refit with early stopping fixes the probability calibration as well.

## Training notes
- The first launch used the PETS bound penalty `0.01 · (Σ max_logvar − Σ min_logvar)` unscaled. With our NLL averaged over 37 dims and 5 members, it was about 185 times too strong, and val NLL got worse after 2 000 iterations (−2.27 → −1.77). It is now divided by E · out_dim. The broken run was deleted.
- Val NLL was still improving slowly at the end of both runs (nominal: −4.23 at 100k → −4.25 at 102k). More passes would help a little.
