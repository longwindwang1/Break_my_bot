# v2 teacher (stage A: PPO with privileged observations)

The teacher for robust policy v2. It sees foot friction, payload, COM offset, motor gains, the scan-deception
flag and foot contact forces (280-dim input). It is not deployable; the student distilled from it is.

- Command: `isaac/train_policy.sh BMB-Go2-Rough-v2 4096 0 3000 v2_teacher`
- Config: `isaac/bmb_tasks/go2_v2_env_cfg.py` (`Go2V2EnvCfg`) and `agents.py` (`Go2V2TeacherPPORunnerCfg`), seed 0.
  The code is at commit ff86f29. `repo_dirty: yes` is caused by this untracked directory; no tracked files were modified.
- Log dir: `logs/rsl_rl/bmb_go2_v2/2026-11-05_22-07-41_v2_teacher` (final checkpoint `model_2999.pt`).
  `isaac/interface.py <log_dir>/params` reports OK.
- Run details:
  - Wall time 19967 s (5.5 h): about 6.6 s per iteration, about 14.8k env steps/s.
  - Peak VRAM 6.9 GB.
  - Total 295M env steps.
  - After about iteration 2550 the laptop ran in the `balanced` power profile. This did not change the speed.

## Progress
| iteration | mean reward | ep. length | track_lin | terrain level | friction lower bound | base_contact |
|---:|---:|---:|---:|---:|---:|---:|
| 300  | 13.06 | 942 | 0.74 | 0.59 | 0.49 | 18.4 % |
| 600  | 16.97 | 918 | 1.04 | 3.69 | 0.38 | 13.3 % |
| 900  | 17.22 | 925 | 1.02 | 5.27 | 0.27 | 14.5 % |
| 1500 | 17.85 | 955 | 1.07 | 5.52 | 0.05 | 12.7 % |
| 2000 | 19.43 | 959 | 1.15 | 5.57 | 0.05 | 9.7 % |
| 2500 | 18.30 | 966 | 1.11 | 5.72 | 0.05 | 9.8 % |
| 2999 | 18.60 | 940 | 1.13 | 5.74 | 0.05 | 10.1 % |

## Final metrics
For the noisy metrics, the value in brackets is the mean over iterations 2900–2999.

| metric | v2 teacher | v1 rough baseline |
|---|---|---|
| mean reward | 18.60 (19.35) | 22.44 (not comparable: v2 adds feet_slide and base_height penalties) |
| mean episode length | 940 (956) / 1000 | 992 |
| track_lin_vel_xy_exp | 1.129 (1.133), about 75 % of 1.5 | 1.181 |
| track_ang_vel_z_exp | 0.570, about 76 % of 0.75 | 0.587 |
| terrain level | 5.74 | 5.45 (different terrain mix: v2 adds pits) |
| terrain level of envs with mean foot μ < 0.3 | 5.78 | — |
| termination by base contact | 10.1 % (8.9 %) | 6.4 % |
| action noise std | 0.64 | 0.62 |

## Observations
- The friction curriculum reached its final lower bound (μ = 0.05) at iteration 1500. Terrain level and episode length did not drop when it did.
- Low-friction robots (μ < 0.3) reach the same terrain level as the others, so the terrain curriculum does not push them back to easy terrain.
- Velocity tracking is 4–5 % below v1, and falls are about 1.5× as frequent. That is with friction down to 0.05, mid-episode friction changes, pushes, COM and gain randomization, and pits with a deceptive scan, none of which v1 trains on.
- Action noise std stayed at about 0.65 throughout. v1 ends at 0.62, so this is normal for this PPO config.
- These are training-time numbers with privileged inputs. The real test is the student in `experiments/v2_eval/`.

## Next
Distill the student (stage B):
```
isaac/train_policy.sh BMB-Go2-Rough-v2-Distill 4096 0 1000 v2_student \
  --agent rsl_rl_distillation_cfg_entry_point --load_run 2026-11-05_22-07-41_v2_teacher --checkpoint model_2999.pt
```
