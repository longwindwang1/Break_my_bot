# W3 Unity smoke test (Windows machine)

The policies trained in Isaac Lab run in Unity physics, and the interface is verified to match Isaac Lab at all four
levels of `docs/unity_handoff.md` §8. Flat, rough and v2 (distilled student) all walk in Unity without falling.

## Setup

| | |
|---|---|
| Unity | 6000.3.8f1, URP 17.3.0, project in `unity/` |
| Inference | `com.unity.ai.inference` 2.6.1, CPU backend (opset 18 models import as exported) |
| Robot import | `com.unity.robotics.urdf-importer` v0.5.2; official Go2 URDF, unitree_ros `a3b70ca` |
| YAML | YamlDotNet 18.1.0 (`Assets/Plugins/YamlDotNet`), reads `configs/robot_interface.yaml` directly |
| Input | Input System 1.18.0 (keyboard + gamepad) |
| Editor automation | MCP for Unity 10.3.0 (`main` @ `81542a7`) |
| Physics | `fixedDeltaTime` = 0.005 exactly (705600 / 141120000), solver 4 / 1, Robot×Robot collisions off |
| Models | `export/models/go2_{flat,rough,v2}.onnx` at merge `ecf5732` |

## How to reproduce

All from the Unity editor menu `BMB/` (or through the MCP `execute_menu_item`):

1. `Apply interface settings` – time step, solver iterations, Terrain / Robot layers.
2. `Sync models` – copies `export/models/*.onnx` to `Assets/Resources/BMB/Models` (gitignored) and checks them.
3. `Setup Go2` – URDF → `Assets/Robots/Go2.prefab` matching Isaac's model.
4. `Build W3_Flat scene` / `Build W3_Rough scene` – playable scenes (Play, keep the editor focused; Tab = manual
   control with W/S, A/D, Q/E; R = reset).
5. `Headless trials/Run all and write report` – closed-loop runs in a preview scene, writes `trials.md` and the CSVs here.
6. Test Runner → EditMode → `BMB.Tests.EditMode` (44 tests), or
   `Unity.exe -batchmode -projectPath unity -runTests -testPlatform EditMode -testResults results.xml` with the project closed.

## Results

| Level (§8) | What | Result | Tolerance |
|---|---|---|---|
| 1 Kinematics | link poses from reference base pose + joint angles, 19 bodies, every 25th step | max 0.0008 mm (flat), 0.0057 mm (rough); orientation ≈ 0° | ~1 mm |
| 1 Joint signs | +effort increases the angle, all 12 joints; flipping one joint's sign moves links > 10 mm | pass | |
| 2 ONNX parity | Unity Inference Engine vs Isaac's GPU actions, 500 steps | flat 1.4e-6, rough 2.2e-6, v2 1.9e-6 | 1e-4 |
| 3 Observation | C# builder from the recorded state converted to Unity frame | flat 1.2e-7, rough 3.6e-7, v2 1.2e-7 (1360-d incl. history) | 1e-5 |
| 3 Height scan | ray xy vs Isaac hit points; values on a known step terrain | 0.002 mm; exact | 1e-4 m |
| 4 Closed loop | vx = 0.5 m/s for 10 s, no resets | flat 0.475 m/s, rough 0.466, v2 0.395 (rough) / 0.427 (flat), 0 falls | walks, tracks |

Gait statistics Unity vs Isaac (speed, drift, base height and attitude, gait period, torques, per-joint angles):
[`trials.md`](trials.md); per-step trajectories: `*_trial.csv`. In short: joint means agree to ~0.03 rad, base
height to ~0.01 m, torque RMS to ~7 %. Systematic differences: Unity's gait period is 15–18 % longer for flat and v2
(rough: identical), and heading / lateral drift under a fixed straight command is larger. From an identical start on
flat ground the joint trajectories agree to 0.005 rad after 5 policy steps and part ways after ~1 s (gait phase).

The v1 flat policy's odd gait (splayed front hips ±0.5 rad, asymmetric rear thighs, base at 0.20 m) is the policy's,
not Unity's: Isaac shows the same statistics.

## Robot model: URDF vs what Isaac simulates

Isaac's `go2.usd` has 19 bodies (15.019 kg). The official URDF has 42 links (16.09 kg: the 19 plus 12 motor rotors
of 0.089 kg); imported as is, Unity gets 25.3 kg because the importer gives links without an inertial a default mass
(1 kg for the camera and each of the 8 calf-lower links). `Setup Go2` therefore:

- keeps the 19 Isaac bodies; the URDF's motor rotors (12 × 0.089 kg), camera, radar, imu and calf-lower links lose
  their ArticulationBody (visuals and colliders stay, attached to the parent; mass dropped);
- writes mass, COM and full inertia tensor of every body from the reference rollout (they equal the URDF values of
  those links); joint limits and PhysX joint velocity limits (30.1 rad/s hips/thighs, **15.7 rad/s calves**) likewise;
- sets drives to zero gains, no damping, no joint friction (torques only from `DCMotor`), v1 friction 0.8 / 0.6
  Multiply on all colliders; v2 runs set the feet to μ (dynamic 0.8 μ) at runtime.

## Unity gotchas found on the way

- Unity 6 stores the fixed time step as a rational; setting `Time.fixedDeltaTime = 0.005f` gives 0.004999993.
  `Apply interface settings` writes the exact count.
- `ArticulationBody` velocity caps, depenetration velocity and solver iterations are not serialized (a prefab reads 0);
  `Go2Body.ApplyRuntimeSettings` sets them on Awake.
- `ArticulationBody.jointForce` persists until overwritten; the controller writes it every physics step.
- The URDF Importer resolves `package://pkg/...` relative to the URDF's folder, so the URDF sits next to the package
  folder (`Assets/Robots/go2_description.urdf`).
- The .dae meshes embed built-in Standard materials (magenta in URP); `Setup Go2` converts them to URP Lit.
- Play mode does not advance while the editor is in the background; headless trials step a preview scene by hand
  instead. An unsaved, modified untitled scene makes the test runner block on a save dialog (and the MCP time out).

## Open questions for the Linux side (answered 2026-11-07)

1. **Does `go2.usd` have the calf-lower collision geometry (`*_calflower*` in the URDF)? Unity keeps it on the calf.**
   - **Yes, and Unity's choice matches it.**
   - Each calf body in `go2.usd` carries three collision cylinders: the URDF's `calf`, `calflower` and `calflower1`, merged into the calf body exactly as Unity does.
   - Sizes and poses in the calf frame equal the URDF's, including:
     - the FL calf's thinner main cylinder (r 0.012 vs 0.013);
     - rotations about y of −0.21 / −0.20, +0.05 and +0.53 rad.

   | shape (calf frame) | radius | height | position | rotation about y |
   |---|---|---|---|---|
   | calf, main | 0.012 (FL), 0.013 (others) | 0.12 | (0.008 \| 0.010, 0, −0.060) | −0.21 \| −0.20 rad |
   | calflower | 0.011 | 0.065 | (0.020, 0, −0.148) | +0.05 rad |
   | calflower1 | 0.0155 | 0.03 | (0.008, 0, −0.1875) | +0.53 rad |

   The other collision shapes also match the URDF. There are 27 shapes in total, all primitives, no meshes:
   - base box 0.376 × 0.094 × 0.114;
   - hip cylinder r 0.046, h 0.04 at y ±0.08;
   - thigh box 0.213 × 0.0245 × 0.034 at z −0.1065;
   - foot sphere r 0.022 at x −0.002;
   - Head_upper cylinder r 0.05, h 0.09;
   - Head_lower sphere r 0.047.

   Checked with `inspect_go2_collision.py` (this folder; needs Isaac Sim, run in `env_isaaclab`). It lists every collision prim of the `go2.usd` that `UNITREE_GO2_CFG` loads (Isaac Sim 5.1 asset).
2. **Height-scan value for a missed ray: Unity assumes −∞ → −1 after the clip (Isaac's `inf` hits).**
   - **Correct.**
   - `raycast_mesh` (`isaaclab/utils/warp/ops.py`) fills missed hits with `inf`.
   - `height_scan` = base_z − hit_z − 0.5 gives −∞, and the obs clip makes it −1.
   - Rays start 20 m above the base with `max_distance` 1e6, so a miss only happens beyond the edge of the terrain mesh.
3. **Exact `random_rough` parameters.**
   - Identical in the v1 rough run and the v2 runs (`params/env.yaml`): `random_uniform_terrain` on 8 m × 8 m tiles, `noise_range` (0.01, 0.06), `noise_step` 0.01, `horizontal_scale` 0.1, `vertical_scale` 0.005, `border_width` 0.25, `slope_threshold` 0.75, `downsampled_scale` null.
   - Unity's inferred values are right. Two additions:
     - this terrain ignores the curriculum difficulty;
     - `slope_threshold` has no effect here: the largest step between neighbouring cells is 0.05 m over 0.1 m (slope 0.5 < 0.75), so the mesh is a plain triangulated heightfield with no vertical walls.
4. **Step-by-step comparison on rough ground would need Isaac's exact terrain.**
   - Open; planned for experiment 3.
   - Simpler than reconstructing it from the hit points: the Linux side can save the terrain mesh (vertices and triangles) the reference rollout ran on, in the npz/json, when the references are next recorded.
