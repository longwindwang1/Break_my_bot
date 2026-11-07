# v2 evaluation: v1 vs v2 teacher vs v2 student

Policies:
- **v1**: W1 rough baseline, `w1_go2_rough_baseline/model_1499.pt`.
- **v2_teacher**: `v2_teacher/model_2999.pt`. Privileged inputs, so a reference only.
- **v2_student**: `v2_student/model_999.pt`. Deployed as `export/models/go2_v2.onnx`.

All three run in the same env instance; env i is driven by policy i % 3. That gives 512 envs per policy,
deterministic actions, one 20 s episode (1000 steps), seed 0. Each scenario runs in its own process (`isaac/eval_policy.py`, about 14 min in total).

```
PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python isaac/eval_policy.py --scenarios all \
  --policy v1=v1:logs/rsl_rl/unitree_go2_rough/2026-10-12_21-27-17_w1_go2_rough_baseline/model_1499.pt \
  --policy v2_teacher=teacher:logs/rsl_rl/bmb_go2_v2/2026-11-05_22-07-41_v2_teacher/model_2999.pt \
  --policy v2_student=student:logs/rsl_rl/bmb_go2_v2/2026-11-06_16-34-51_v2_student/model_999.pt \
  --out experiments/v2_eval/results.jsonl
python isaac/eval_summary.py experiments/v2_eval/results.jsonl   # prints the tables below
```

Setup:
- `Go2V2EnvCfg_PLAY`: no observation noise, no payload/COM/gain randomization during play.
- Terrain: 10 × 10 tiles of the full mix, difficulty 0–0.5.
- Commands: the training distribution, |v| ≤ 1 m/s.

Metrics:
- **success**: the episode reaches the 20 s timeout without a base contact.
- **tracking error**: mean ‖v_xy − v_cmd‖ in m/s while alive.
- **slip**: mean horizontal speed of feet in contact (m/s).

With 512 envs, the standard error of a success rate around 90 % is about ±1.3 points.

| scenario | what changes |
|---|---|
| mu_X | foot friction fixed at X for the whole episode |
| mu_drop | μ 1.0, dropped to 0.1 on all feet at t = 5 s |
| push | μ 1.0, velocity pushes of up to 1.0 m/s every 3–5 s (training used 0.5 m/s every 10–15 s) |
| pits_clean / pits_deceived | pit terrain only, difficulty 0.5–1.0 (blocks 10–16 cm up or down); deceived = the scan reports flat ground all episode |
| mu_0.1_cautious | μ 0.1 with the command capped to the cautious gait (0.3 m/s, 0.5 rad/s) |

## Results

**Success rate**

| scenario | v1 | v2 teacher | v2 student |
|---|---:|---:|---:|
| mu_1.0 | 97.1 % | 97.9 % | **98.0 %** |
| mu_0.5 | 97.7 % | 99.8 % | 97.7 % |
| mu_0.2 | 82.0 % | 97.5 % | **95.1 %** |
| mu_0.1 | 52.0 % | 95.9 % | **90.4 %** |
| mu_0.05 | 33.8 % | 95.3 % | **83.0 %** |
| mu_drop | 63.9 % | 95.5 % | **89.8 %** |
| push | 94.7 % | 97.7 % | 94.7 % |
| pits_clean | 87.1 % | 91.2 % | 87.7 % |
| pits_deceived | 88.5 % | 89.1 % | 87.7 % |
| mu_0.1_cautious | 73.0 % | 98.2 % | 88.9 % |

**Tracking error (m/s)**

| scenario | v1 | v2 teacher | v2 student |
|---|---:|---:|---:|
| mu_1.0 | 0.167 | 0.117 | 0.143 |
| mu_0.5 | 0.154 | 0.118 | 0.136 |
| mu_0.2 | 0.274 | 0.190 | 0.191 |
| mu_0.1 | 0.411 | 0.305 | 0.334 |
| mu_0.05 | 0.518 | 0.421 | 0.468 |
| mu_drop | 0.306 | 0.251 | 0.250 |
| push | 0.194 | 0.150 | 0.187 |
| pits_clean | 0.180 | 0.145 | 0.190 |
| pits_deceived | 0.195 | 0.152 | 0.201 |
| mu_0.1_cautious | 0.119 | 0.140 | 0.114 |

**Foot slip in contact (m/s)**

| scenario | v1 | v2 teacher | v2 student |
|---|---:|---:|---:|
| mu_1.0 | 0.212 | 0.104 | 0.098 |
| mu_0.5 | 0.269 | 0.124 | 0.122 |
| mu_0.2 | 0.577 | 0.229 | 0.247 |
| mu_0.1 | 0.896 | 0.373 | 0.406 |
| mu_0.05 | 1.169 | 0.531 | 0.575 |
| mu_drop | 0.674 | 0.301 | 0.328 |
| push | 0.227 | 0.110 | 0.103 |
| pits_clean | 0.234 | 0.107 | 0.097 |
| pits_deceived | 0.234 | 0.115 | 0.097 |
| mu_0.1_cautious | 0.395 | 0.146 | 0.207 |

## Findings
1. **Low friction: the main goal of v2 is met.**
   - The student survives μ = 0.1 in 90 % of episodes (v1: 52 %) and μ = 0.05 in 83 % (v1: 34 %).
   - After a sudden 1.0 → 0.1 drop, the student survives 90 % (v1: 64 %), the same as at constant μ = 0.1. So the sudden change itself costs nothing extra.
   - Foot slip is roughly halved at every friction level.
2. **No cost on normal ground.**
   - At μ = 1.0, student success is 98 % vs 97 % for v1, and tracking is better (0.143 vs 0.167 m/s).
   - The base-height and feet-slide rewards also cut slip from 0.21 to 0.10 m/s.
3. **Distillation gap.**
   - The student is within 6 points of the teacher everywhere except μ = 0.05, where the gap is 12 points (83 vs 95 %). That is where inferring friction from history matters most.
   - Options if this matters for the game: a longer history (25 → 50 steps), more distillation iterations, or a recurrent student.
4. **Pushes.** At 1.0 m/s every 3–5 s, which is harder than training, the student equals v1 (94.7 %). v2 trained only with 0.5 m/s pushes. Raise that in the next version if the game pushes hard.
5. **The pit / deceptive-scan test does not discriminate.**
   - A deceived scan does not hurt any policy, v1 included (v1: 87.1 % clean vs 88.5 % deceived).
   - Checked separately: in `pits_deceived` the deception flag is 1 for every env and the scan is exactly flat, so the scenario works as intended.
   - The conclusion is that blocks of at most 16 cm over a 0.3–0.8 m footprint can be crossed blind; the ~12 % failures on pit terrain come from the terrain itself.
   - Testing "breakable ice" properly needs deeper holes (about 25–40 cm, in training and in eval).
6. **Cautious gait.**
   - Capping the command at μ = 0.1 helps v1 a lot (52 → 73 %) and the teacher a bit (95.9 → 98.2 %).
   - It does not raise student success (90.4 → 88.9 %, within noise), although tracking error drops.
   - For the surprisal experiment, the alarm's benefit with the v2 student will therefore have to come mostly from clearing and re-filling the history, not from slowing down. Look at the student's μ = 0.1 failure cases (e.g. turning or standing commands) before relying on this.

## Files
- `results.jsonl`: one row per scenario and policy (success_rate, tracking_error, slip_speed, envs, checkpoint, seed).
