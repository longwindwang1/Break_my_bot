# Unity eval: the v2_eval scenarios in Unity physics (Experiment 3, transfer)

The 10 scenarios of `isaac/eval_policy.py` (see `experiments/v2_eval/`), run in Unity with the deployed ONNX models:
- **v1**: `go2_rough.onnx` (W1 rough baseline).
- **v2**: `go2_v2.onnx` (v2 student).

The v2 teacher needs privileged inputs and is not deployed, so it is not run here.

Run: Unity menu `BMB/Eval/Run Isaac scenarios (v1 + v2, 64 envs)` (code: `unity/Assets/Scripts/Editor/UnityEval.cs`).
- It runs as a background job: about 10 min for all 10 scenarios on this laptop, ~1 min per scenario for 128 robots × 20 s.
- Progress shows in Unity's progress bar, where it can also be cancelled. Entering Play mode or recompiling stops it.
- Runs are deterministic: a rerun reproduces every number exactly.
- `BMB/Eval/Rewrite summary.md` regenerates the tables from the jsonl files.

The TGS variant was started from code with the project solver switched, then switched back:
`UnityEval.SetSolver(true); UnityEval.Start(EvalScenario.Isaac, new EvalOptions(), "_tgs")`, then `SetSolver(false)`.

## Setup

| | Isaac (`v2_eval`) | Unity (here) |
|---|---|---|
| Envs per policy | 512 | 64. Standard error ≈ ±3.7 points at 90 %, ±6 at 50 % |
| Pairing | env i → policy i % P | env e of v1 and v2 share tile, spawn, payload, gains, commands and push times |
| Terrain | 10 × 10 tiles of the v2 mix, difficulty 0–0.5; pit scenarios: pit tiles only, difficulty 0.5–1.0 | 6 × 6 tiles, same mix and ranges, from the `IsaacTiles` port (same distributions, different samples), 20 m flat border |
| Reset | tile centre ± 0.5 m, random yaw, default joints, zero velocity | same |
| Commands | heading mode, \|v\| ≤ 1 m/s, resampled every 10 s, 2 % standing | same (`CommandSampler`) |
| Randomization | payload −1…3 kg, COM ±5/±5/±2 cm, Kp/Kd × 0.9–1.1 per joint | same |
| Friction | foot μ = X on all feet, terrain 1.0, multiply | same (dynamic = 0.8 · static) |
| mu_drop | all feet → 0.1 at step 250 (t = 5 s) | same |
| push | Δv uniform ±1.0 m/s per axis every 3–5 s | same, added to the base velocity |
| pits_deceived | scan = flat ground all episode | same (`HeightScanner.Deceive`) |
| Episode / success | 1000 steps = 20 s, no `base_contact` | same; base contact = the base box within 5 mm of the terrain |
| Slip | feet with contact force > 1 N | feet whose collision sphere is within 3 mm of the terrain |
| Physics | PhysX 5 GPU, TGS, 4 / 0 iterations | PhysX (Unity 6000.3, CPU), PGS, 4 / 1 iterations (project setting since W3); TGS also run |

Note on the Isaac setup: `experiments/v2_eval/README.md` says play mode has no payload / COM / gain randomization.
- But `Go2V2EnvCfg_PLAY` only disables scan deception, pushes and the friction curriculum.
- So `payload_mass`, `payload_com` and `actuator_gains` stay on, and `record_reference.py` turns them off explicitly.
- The Isaac eval therefore ran with them, and Unity does the same. The Linux side should fix that README line.

## Results

**Success rate**

| scenario | v1 Isaac | v1 Unity | v1 Unity TGS | v2 Isaac | v2 Unity | v2 Unity TGS |
|---|---:|---:|---:|---:|---:|---:|
| mu_1.0 | 97.1 % | 78.1 % | 81.2 % | 98.0 % | 96.9 % | 96.9 % |
| mu_0.5 | 97.7 % | 71.9 % | 78.1 % | 97.7 % | 92.2 % | 93.8 % |
| mu_0.2 | 82.0 % | 45.3 % | 48.4 % | 95.1 % | 95.3 % | 82.8 % |
| mu_0.1 | 52.0 % | 26.6 % | 12.5 % | 90.4 % | 73.4 % | 78.1 % |
| mu_0.05 | 33.8 % | 17.2 % | 12.5 % | 83.0 % | 60.9 % | 48.4 % |
| mu_drop | 63.9 % | 31.2 % | 29.7 % | 89.8 % | 76.6 % | 68.8 % |
| push | 94.7 % | 65.6 % | 75.0 % | 94.7 % | 92.2 % | 95.3 % |
| pits_clean | 87.1 % | 84.4 % | 79.7 % | 87.7 % | 84.4 % | 89.1 % |
| pits_deceived | 88.5 % | 79.7 % | 85.9 % | 87.7 % | 90.6 % | 95.3 % |
| mu_0.1_cautious | 73.0 % | 28.1 % | 40.6 % | 88.9 % | 89.1 % | 89.1 % |

Tracking error and foot slip (Unity with PGS) are in `summary.md`:
- tracking error is higher in every scenario: by +0.01 to +0.07 m/s on normal ground, and by +0.2 to +0.27 m/s at
  μ ≤ 0.1;
- slip is about equal on pits and with pushes, and higher at low μ: by up to +0.28 m/s for v1 and +0.13 m/s for v2.

**Falls by spawn tile (PGS; falls / episodes)**

| scenario | policy | PyramidStairs | PyramidStairsInv | Boxes | RandomRough | PyramidSlope | PyramidSlopeInv | Pits |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mu_1.0 | v1 | 6/15 | 4/8 | 1/1 | 1/16 | 1/8 | 1/7 | 0/9 |
| mu_1.0 | v2 | 1/15 | 1/8 | 0/1 | 0/16 | 0/8 | 0/7 | 0/9 |
| push | v1 | 8/15 | 5/8 | 1/1 | 2/16 | 1/8 | 3/7 | 2/9 |
| push | v2 | 2/15 | 2/8 | 1/1 | 0/16 | 0/8 | 0/7 | 0/9 |
| mu_0.1 | v1 | 13/15 | 5/8 | 1/1 | 11/16 | 4/8 | 7/7 | 6/9 |
| mu_0.1 | v2 | 6/15 | 1/8 | 0/1 | 6/16 | 0/8 | 2/7 | 2/9 |

## Findings
1. **v2 transfers to Unity physics; v1 does not.**
   - At μ 1.0 / 0.5 and with pushes, v2 loses 1–6 points against Isaac; v1 loses 19–29.
   - On pit tiles both stay within 3–9 points of Isaac.
   - v2's training randomization (friction, payload, gains, pushes, deceptive scan) also makes it robust to a different
     physics engine. This is the main result of Experiment 3.
2. **v1's falls on normal ground are on stairs and boxes.**
   - At μ 1.0, 11 of its 14 falls start on pyramid stairs (both directions) or boxes. Those are Isaac's mesh terrains,
     which `IsaacTiles` rasterizes on 0.1 m cells with vertical walls (stair treads exact, box grid approximate).
   - v2 crosses the same tiles (2 falls in 24).
   - To separate "our stair geometry" from "Unity contact on edges", put one stair tile in each simulator and compare
     v1 on it.
3. **Low friction is harder in Unity for both policies.**
   - v2: μ 0.1 gives 73 % (Isaac 90 %), μ 0.05 gives 61 % (Isaac 83 %), mu_drop 77 % (90 %).
   - Tracking error rises by about 0.25 m/s and slip by about 0.12 m/s.
   - The ice is effectively more slippery in Unity's friction model at the same μ.
4. **The cautious gait helps in Unity.**
   - At μ 0.1, capping the command (0.3 m/s, 0.5 rad/s) lifts v2 from 73 % to 89 %, the same as in Isaac (89 %).
   - In Isaac, where v2 already reached 90 % at μ 0.1, the cap changed nothing.
   - So the surprisal alarm → cautious gait mechanism (W6–8) is worth having in the game, where the robot runs in Unity
     physics.
5. **The solver is not the cause.**
   - Switching Unity to TGS (Isaac Lab's default) changes success by −14 … +12 points per scenario with no consistent sign: mean +1.6 for v1, −1.4 for v2.
   - The project stays on PGS, which is what W3 was validated with.
6. **Pits behave as in Isaac.**
   - Both policies reach 80–95 %, and the deceptive scan does not hurt.
   - Same conclusion as `v2_eval`: 4–16 cm blocks can be crossed blind.
7. **Side observation.** With a zero command, v2 stands on three legs and holds the rear-right foot about 14 cm up
   (seen in the drag/ball tests). It is stable, but looks odd in the game.

## Files
- `results.jsonl` / `results_tgs.jsonl`: one row per scenario and policy. Fields: success_rate, tracking_error,
  slip_speed, mean_fall_time, solver, envs, model, seed, steps, wall_seconds.
- `episodes.csv` / `episodes_tgs.csv`: one row per episode. Fields: tile type and difficulty at spawn, payload,
  success, fall time, per-episode tracking error and slip.
- `summary.md` / `summary_tgs.md`: Isaac vs Unity tables for all three metrics, generated after every scenario.
