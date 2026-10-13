# W1 Go2 flat-ground baseline

Policy for the W3 Unity smoke test.

- Command: `isaac/train_policy.sh Isaac-Velocity-Flat-Unitree-Go2-v0 4096 0 300 w1_go2_flat_baseline`
- Config: stock Isaac Lab Go2 flat env and PPO (actor/critic 128-128-128), seed 0
- Run details (git hash, peak VRAM, log dir): see `run_info.txt`.
  Log dir: `logs/rsl_rl/unitree_go2_flat/2026-10-12_21-19-01_w1_go2_flat_baseline`
- Wall time 269 s (~110k steps/s), peak VRAM 3.96 GB. Note: `nvidia-powerd` was enabled partway through the run.
  The repo had no commits yet, so `repo_commit` is `none`.

## Final metrics (iteration 299)
| metric | value |
|---|---|
| mean reward | 34.55 |
| mean episode length | 1000 / 1000 (full 20 s episodes) |
| track_lin_vel_xy_exp | 1.419 (weight 1.5 → ~95%) |
| track_ang_vel_z_exp | 0.641 (weight 0.75 → ~85%) |
| termination by base contact | 0.3% |

## Still to do
- Look at it in `play.py` (this also exports `exported/policy.onnx`).
