# W4 world-model data

Rollouts recorded in Isaac Lab by `isaac/collect_wm_data.py` for the world model.
- Location: `logs/wm_data/w4_<split>/` (gitignored, about 12 GB in total).
- Each split is a set of `[T, N, ...]` `.npy` memmaps plus `meta.json` (config, seed, git, checkpoints, terrain table, `steps_written`) and `env_info.npz` (policy per env, noise, event type, reset-before-event).
- The data was collected at commit 71d478b, with the collector not yet committed; it is committed in b0f7f31.
- `train_nominal` was added afterwards, at b0f7f31.

```
V2=logs/rsl_rl/bmb_go2_v2/2026-11-06_16-34-51_v2_student/model_999.pt
V1=logs/rsl_rl/unitree_go2_rough/2026-10-12_21-27-17_w1_go2_rough_baseline/model_1499.pt
for s in train train_nominal clean events events_pits; do
  PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python isaac/collect_wm_data.py --split $s --v2 $V2 --v1 $V1 --out_dir logs/wm_data/w4_$s
done
```

## Setup
Common to all splits:
- **Env:** `Go2V2EnvCfg_PLAY`: clean observations; payload, COM and gain randomization stay on.
  - The terrain-level curriculum is off. Each env stays on its tile, and a new episode respawns there.
  - Terrain: 10 × 20 tiles of 8 m, with a random sub-terrain type and difficulty per tile.
- **Policies:** env i runs `(v2, v2, v1)[i % 3]`.
  - v2 is the distilled student.
  - v1 is the W1 rough baseline, included so that "v1 + surprisal trigger" (experiment 1) has in-distribution data.
- **Disturbances:** none come from the env's own random events. Every disturbance is scheduled by the script and flagged per step (`push`, `mu_change`, `event`, `deceived`).
- **Recording order per policy step t:**
  1. o_t and the state it came from: root pose, foot μ, `episode_length_buf`, tile;
  2. a_t (as applied, including noise);
  3. `env.step(a_t)`;
  4. `terminated_t` (base contact = fall) and `truncated_t` (20 s time-out).
- **Checks:**
  - Every split asserts that the last-action term of o_{t+1} equals a_t whenever the episode continues. There were 0 mismatches.
  - The first 8 envs also store Isaac's own 25-frame `proprio_history`. `models/world_model/test_dataset.py` checks that the offline history reconstruction matches it bit for bit, across resets. This was run on the smoke splits.

| split | envs × steps | seed | conditions | use |
|---|---|---|---|---|
| `train` | 4096 × 1500 (6.14M) | 100 | foot μ 0.05–1.25, re-drawn every 4–8 s (half single-foot); pushes 0–1.0 m/s in a random direction every 5–10 s; terrain difficulty 0–1; N(0, 0.1²) action noise on every 4th env | training (env i % 23 == 0 held out) |
| `train_nominal` | 4096 × 1500 (6.14M) | 500 | "normal physics": like clean (μ fixed per episode in [0.6, 1.25], no pushes) plus action noise on every 4th env | training data of the surprisal model's dynamics (see `experiments/w4_wm_ensemble/`) |
| `clean` | 2048 × 1500 (3.07M) | 200 | μ fixed per episode in [0.6, 1.25], no pushes, no noise, terrain as train | threshold calibration, false alarms |
| `events` | 4096 × 600 (2.46M) | 300 | clean, difficulty 0–0.5. At step 200 (4 s) env i gets event `i % 7`: none, all feet μ→0.3 / 0.1 / 0.05, one foot μ→0.05, push 0.5 / 1.0 m/s (fixed magnitude, random direction) | detection rate and delay |
| `events_pits` | 2048 × 600 (1.23M) | 400 | pit terrain only, blocks 15–30 cm (training had 4–16 cm), difficulty 0.5–1. From step 200 the height scan reports flat ground in about half the envs | hidden pits / breakable ice |
| `events_test`, `events_pits_test` (W7) | 4096 × 600 / 2048 × 600 | 310 / 410 | same conditions as `events` / `events_pits` (`SPLIT_KIND` in `isaac/wm_splits.py`) | report only: the W7 world-model comparison chooses detector settings on `events` / `events_pits` and scores them here |

## Contents
| split | wall time | falls (v1 / v2) | falls per robot-minute (v1 / v2) | other |
|---|---|---|---|---|
| train | 396 s | 835 (428 / 407) | 0.63 / 0.30 | 14 397 pushes, 18 455 friction changes, 3 766 time-outs |
| train_nominal | 364 s | 503 (203 / 300) | 0.30 / 0.22 | 3 876 time-outs; collected at b0f7f31 |
| clean | 201 s | 192 (83 / 109) | 0.24 / 0.16 | 1 969 time-outs |
| events | 123 s | 239 (158 / 81) | 0.58 / 0.15 | 9 envs reset before the event; about 584 envs per event type |
| events_pits | 58 s | 568 (221 / 347) | 1.62 / 1.27 | 1 071 envs deceived |
| events_test | 122 s | 248 (163 / 85) | 0.60 / 0.16 | 6 envs reset before the event; collected 2026-11-09 into `logs/wm_data/w7_events_test` |
| events_pits_test | 59 s | 569 (217 / 352) | 1.59 / 1.29 | 1 061 envs deceived; `logs/wm_data/w7_events_pits_test` |

- Terrain tiles (train and clean; 200 tiles): random_rough 39, inverted stairs 34, boxes 29, pits 25, slopes 25, stairs 25, inverted slopes 23.
- Fall labels for the world model use horizons of 10, 25 and 50 steps. In the train envs that gives 5 105, 13 595 and 27 488 positive steps; censored steps and noisy envs are excluded.
