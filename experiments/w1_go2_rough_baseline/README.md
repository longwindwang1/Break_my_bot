# W1 Go2 rough-terrain baseline

The base locomotion policy for the project.

- Command: `isaac/train_policy.sh Isaac-Velocity-Rough-Unitree-Go2-v0 4096 0 1500 w1_go2_rough_baseline`
- Config: stock Isaac Lab Go2 rough env and PPO (actor/critic 512-256-128), seed 0. No changes to
  rewards or domain randomization; pushes are still disabled (`push_robot = None`).
- Run details: `run_info.txt`. Log dir: `logs/rsl_rl/unitree_go2_rough/2026-10-12_21-27-17_w1_go2_rough_baseline`
  (final checkpoint `model_1499.pt`)
- Wall time 9791 s (~2.7 h) at ~14–16k steps/s with `nvidia-powerd` running; peak VRAM 5.94 GB.
  The repo had no commits yet, so `repo_commit` is `none`.

## Progress
| iteration | mean reward | terrain level (0–9) |
|---:|---:|---:|
| 150  | 9.85  | 0.48 |
| 300  | 14.48 | 0.94 |
| 600  | 18.20 | 4.10 |
| 900  | 19.54 | 5.34 |
| 1200 | 20.65 | 5.29 |
| 1500 | 22.44 | 5.45 |

The terrain curriculum stops climbing at level ~5.4 after about 900 iterations. Reward keeps rising slowly after that.

## Final metrics (iteration 1499)
| metric | value |
|---|---|
| mean reward | 22.44 |
| mean episode length | 992 / 1000 |
| track_lin_vel_xy_exp | 1.181 (weight 1.5 → ~79%) |
| track_ang_vel_z_exp | 0.587 (weight 0.75 → ~78%) |
| terrain level | 5.45 |
| termination by base contact | 6.4% |

## Still to do
- Look at it in `play.py` with `Isaac-Velocity-Rough-Unitree-Go2-Play-v0` (this also exports `exported/policy.onnx`).
