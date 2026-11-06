# v2 debug runs (teacher, 1024 envs, 300 iterations, seed 0)

Short runs to check that the v2 teacher (`BMB-Go2-Rough-v2`) learns before starting the full run.
Raw logs: `experiments/v2_debug_teacher_try1/`, `experiments/v2_debug_teacher/`, `experiments/v2_debug_v1_baseline_1024/`
(each has a `run_info.txt` with the log dir).

| run | change | it. 300: reward | ep. length | track_lin | base_contact | terrain level |
|---|---|---:|---:|---:|---:|---:|
| try1 | first version: `bad_orientation` termination (1.2 rad), pushes 0.8 m/s every 6–10 s, payload −1…+5 kg, gains ±20 % | 4.49 | 598 | 0.18 | ~0 (70 % ended by bad_orientation) | 0.00 |
| try2 | no `bad_orientation`; pushes 0.5 m/s every 10–15 s; payload −1…+3 kg; gains ±10 % | 5.03 | 911 | 0.36 | 25 % | 0.00 |
| v1 baseline | stock `Isaac-Velocity-Rough-Unitree-Go2-v0`, same 1024 envs / 300 it. | 8.49 | 940 | 0.54 | 6.5 % | 0.09 |

Findings:
- try1 had the classic "learning to terminate" problem. Early per-step reward is negative, and the extra tip-over
  termination made ending the episode easy: episode length fell from 315 to 121 steps by iteration 60. Fixed in try2.
- The terrain level collapsing toward 0 early is normal at 1024 envs. The v1 baseline does it too. v1 at 4096 envs
  recovered later and plateaued at about 5.4.
- v2 learns about 40 % slower than v1 and falls more, as expected with the extra randomization. It is still improving.
  Rewards are not directly comparable: v2 adds feet_slide and base_height penalties.
- Consequence: the friction curriculum was stretched from 24k to 36k env steps (≈1500 iterations) for the full run.
