# v2 student (stage B: distillation from the v2 teacher)

The deployable v2 policy. It only uses inputs that also exist in Unity:
- the 235-dim v1 observation (the height scan may lie during training),
- a 25-step (0.5 s) proprioception history,
- 1360 dims in total.

It was trained with rsl-rl's `DistillationRunner` in DAgger style: the student acts, and the frozen teacher labels the states the student visits.

- Command: `isaac/train_policy.sh BMB-Go2-Rough-v2-Distill 4096 0 1000 v2_student --agent rsl_rl_distillation_cfg_entry_point --load_run 2026-11-05_22-07-41_v2_teacher --checkpoint model_2999.pt`
- Config:
  - `Go2V2EnvCfg_DISTILL`: same randomization as the teacher; the friction curriculum starts finished, so μ ranges over 0.05–1.25.
  - `Go2V2DistillationRunnerCfg`: MSE behavior loss, lr 1e-3, 2 epochs, gradient_length 12, student noise std 0.1.
  - Seed 0. The code is at commit b040b15; `repo_dirty` is caused by untracked experiment outputs.
- Teacher: `logs/rsl_rl/bmb_go2_v2/2026-11-05_22-07-41_v2_teacher/model_2999.pt` (`experiments/v2_teacher/`).
- Log dir: `logs/rsl_rl/bmb_go2_v2/2026-11-06_16-34-51_v2_student` (final checkpoint `model_999.pt`).
  `isaac/interface.py <log_dir>/params` reports OK.
- Run details: wall time 6412 s (1.8 h), about 6.4 s per iteration, peak VRAM 6.4 GB, 98M env steps.

## Progress
| iteration | behavior loss | mean reward | ep. length | track_lin | terrain level | base_contact |
|---:|---:|---:|---:|---:|---:|---:|
| 10  | 0.803 | −0.66 | 100 | 0.03 | — | 76 % |
| 50  | 0.429 | 3.03  | 480 | 0.12 | 1.10 | 68 % |
| 100 | 0.273 | 13.76 | 861 | 0.53 | 0.61 | 40 % |
| 200 | 0.128 | 30.55 | 972 | 1.19 | 1.94 | 7.7 % |
| 300 | 0.222 | 26.86 | 903 | 1.06 | 3.43 | 11.1 % |
| 400 | 0.103 | 31.10 | 968 | 1.21 | 4.72 | 8.1 % |
| 600 | 0.106 | 30.80 | 967 | 1.21 | 5.98 | 9.7 % |
| 800 | 0.091 | 29.42 | 947 | 1.15 | 5.79 | 9.9 % |
| 999 | 0.091 | 29.59 | 948 | 1.14 | 5.80 | 9.1 % |

Final values for metrics not in the table:
- track_ang_vel_z_exp: 0.636 out of 0.75.
- Terrain level of envs with foot μ < 0.3: 5.81.

## Observations
- **Early falls.** The student starts as a random network and is the one acting, so for the first ~100 iterations it falls a lot. The terrain curriculum demotes it to level ~0.6. Once it walks, the level climbs back to the teacher's (~5.8) by iteration 500.
- **Low friction.** Low-friction robots reach the same terrain level as the others, as with the teacher.
- **Reward vs. the teacher.** The student's training reward (29.6) is higher than the teacher's (18.6). This does not mean the student is better: PPO logs the teacher while it samples with exploration noise std ≈ 0.65, while the student acts with std 0.1. `experiments/v2_eval/` compares all policies with deterministic actions.
- **Loss plateau.** Behavior loss levels off at about 0.09 (MSE in action units) after iteration 700.

## Export and parity
- `export/models/go2_v2.onnx`: input `obs` [1, 1360], output `actions` [1, 12], opset 18. Layout per `policies.v2` in `configs/robot_interface.yaml`.
- Reference rollout `experiments/w3_reference/go2_v2.npz` / `.json`:
  - 500 steps on `random_rough`, foot μ 1.0, command (0.5, 0, 0).
  - No resets.
  - The robot covers 4.1 m in 10 s, about 82 % of the commanded 5 m.
  - The base stays 0.37 m above the terrain.
- `pytest`: all v2 parity tests pass. They cover:
  - observation rebuild including the 25-step history and the reset fill,
  - torch vs ONNX,
  - ONNX replay of the recorded actions,
  - the IO contract.
