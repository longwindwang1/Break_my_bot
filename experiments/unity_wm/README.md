# World model in Unity physics (W4–5 Unity side, Experiment 3)

The Unity side of `docs/unity_handoff.md` §11. It covers:
- verification of `go2_wm.onnx` and the C# monitor against the Isaac traces;
- the blue "ghost" preview;
- recalibration of the surprisal alarm in Unity physics;
- event detection, Unity vs Isaac.

Code: `unity/Assets/Scripts/Runtime/WorldModel/` (runner, monitor, surprisal, dreamer), `Runtime/View/Ghost*.cs`, `Editor/WorldModelEval.cs`.

## Verification (§11.3)
| level | what | result (tolerance) |
|---|---|---|
| 1 | ONNX outputs on `go2_wm*.json` `wm_in`, all 500 steps | state_mean 7.6e-6 abs (1e-4), variances 7.7e-5 rel (1e-3) |
| 2 | `wm_in` built in C# from the v2 reference state | 2.4e-7 (1e-5) |
| 3 | z, S, x, g from stored states and outputs | 3e-7 (1e-5). Alarms identical: clean none; mudrop 12, the first at step 262 |
| 4 | closed loop in Unity with Unity's calibration | v2 on flat ground: 0 alarms in 30 s; foot μ → 0.05 detected after 0.88 s |

Inference: 0.83 ms per call on CPU and 2.3 ms on GPUCompute. CPU is used.

## Ghost
The ghost is the world model's imagined robot, 0.5 s ahead.
- **How it is computed.** The policy runs in the loop on predicted states, with a copied v2 history. The height scan is ray-cast at the imagined pose.
- **Accuracy.** It was compared with where the robot then really is in Unity (v2, 0.5 m/s, flat, 6 dreams):
  - base error 0.3 cm, while the robot moved 22 cm;
  - yaw 0.3°;
  - joints 0.016 rad RMS.
- **Forward kinematics.** The ghost's FK matches Isaac's link poses to 0.002 mm.
- **On ice.** The model only knows normal physics. On ice the ghost walks on while the robot slips, and every alarm flashes it magenta.

## Recalibration (§11.4): `calibration.md`
Clean walking setup:
- 64 envs per policy, 60 s each;
- foot μ 0.6–1.25 per episode;
- payload, COM and gain randomization;
- commands as in training, no pushes.

Each split is halved: half A sets the thresholds and half B checks them. Target: ≤ 33 false alarms per 10⁶ scored steps (1 per 10 min).

**False alarms per 10⁶ scored steps on clean walking**

| clean terrain | v1 Isaac thresholds | v1 Unity calibration (A / B) | v2 Isaac thresholds | v2 Unity calibration (A / B) |
|---|---:|---:|---:|---:|
| flat + random rough (handoff; **used**) | 214 | 32 / 53 | 449 | 32 / 21 |
| Isaac's full training mix, difficulty 0–1 (`calibration_isaac_terrain.md`) | 1719 | 32 / 32 | 1362 | 32 / 21 |

**Detection within 2 s, events split** (`detection.md`)
- Isaac's events-split terrain: training mix, difficulty 0–0.5.
- Event at 4 s; 28 robots per event and policy (Isaac: about 195 / 390).
- Unity is scored on the same rollouts with each set of thresholds.

| event | v1 Isaac | v1 Unity, Isaac thr. | v1 Unity, calibrated | v2 Isaac | v2 Unity, Isaac thr. | v2 Unity, calibrated |
|---|---:|---:|---:|---:|---:|---:|
| none (control) | 0 % | 0 % | 0 % | 0 % | 7 % | 4 % |
| all feet μ → 0.3 | 23 % | 41 % | 19 % | 4 % | 21 % | 18 % |
| all feet μ → 0.1 | 90 % | 96 % | 96 % | 81 % | 86 % | 89 % |
| all feet μ → 0.05 | 96 % | 96 % | 96 % | 96 % | 96 % | 96 % |
| one foot μ → 0.05 | 48 % | 52 % | 0 % | 21 % | 43 % | 21 % |
| push 0.5 m/s | 97 % | 96 % | 100 % | 99 % | 96 % | 100 % |
| push 1.0 m/s | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |

Median delay for μ → 0.1, calibrated (Isaac): v1 29 steps (14), v2 17 steps (34). 50 steps = 1 s.

## Findings
1. **The world model runs in Unity exactly as exported.**
   - Levels 1–3 match to float precision.
   - Its one-step predictions stay accurate in Unity physics: the 0.5 s ghost lands within 0.3 cm on flat ground.
2. **Isaac's thresholds give too many false alarms in Unity.** Ice and pushes are still caught, as well as in Isaac or better. But clean walking alarms far too often:
   - 6–13× the target on flat and rough ground;
   - 40–50× on Isaac's full terrain mix.
   - That is one false alarm every 45 s (v2) to 90 s (v1) on flat and rough ground, and every 12–15 s on the full mix.
   - Why: Unity's base-velocity residuals are wider (z std 1.08 vs 0.87). The joint residuals are more peaked (|z| < 1: 89–93 % vs 84–85 %) but heavier-tailed.
3. **Calibrated on flat and rough ground (what §11.4 asks), the detector meets the budget and keeps its power.**
   - Ice at μ ≤ 0.1 is caught 89–96 % of the time within 2 s, and pushes 100 %. That is as good as Isaac or better.
   - Single-foot ice stays weak, as in Isaac: v2 21 % in both. v1 drops to 0 % (Isaac 48 %).
   - This calibration is in `experiments/w3_unity/wm_calibration_unity.json`, and the game uses it.
4. **On stairs, boxes and pits up to difficulty 1, normal walking in Unity surprises the model as much as ice does.**
   - Calibrating on that terrain meets the budget only with thresholds that miss ice: v1 μ 0.1 at 0 %, v2 32 % (`detection_isaac_terrain.md`).
   - This matches `experiments/unity_eval`, where v1's Unity falls are on stairs and boxes.
   - A game level full of stairs therefore needs terrain-aware thresholds, or help from the terrain-risk model (§12).
5. **The z correction in §11.4 (median / 1.4826·MAD per dim) does not suit Unity's residuals.** For the Linux side:
   - Unity's joint residuals are peaked, so the MAD scale is 0.3–0.4.
   - Dividing by it inflates every normal joint step about 3×, and S hits the clip. s0 grows to about 5× Isaac's (2× with the std correction), and ice separates less.
   - Three corrections were compared: MAD, none, and Unity std / Isaac std per state term. All three were recalibrated the same way and judged by Isaac's own rule: mean detection within 2 s over the key events, at most 2× the false-alarm budget.
   - The std ratio won narrowly: 80 % vs 77 % for MAD and 79 % for none, which was over budget for v1.

## Files
- `calibration.md`, `detection.md`: generated tables for the calibration in use (flat + rough clean data), per state term and per tile.
- `calibration_isaac_terrain.md`, `detection_isaac_terrain.md`: the same with clean data on Isaac's full mix (finding 4).
- `experiments/w3_unity/wm_calibration_unity.json`: the thresholds the runtime prefers over the yaml (`SurprisalParams.Load`).
- Raw z of every episode is saved to `logs/unity_wm/{clean, clean_isaac_terrain, events}.bin` (gitignored). `BMB/World model/Re-analyse saved runs` rewrites the tables from these files without simulating.

Commands (Unity menu):
- `BMB/World model/Calibrate + detect events (v1 + v2, ~15 min)`;
- `.../Recalibrate (clean split only, reuse saved events)`, about 9 min;
- `.../Re-analyse saved runs`.

All are background jobs; entering Play or recompiling stops them.
