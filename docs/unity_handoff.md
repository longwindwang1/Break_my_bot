# Break My Bot — Unity handoff

For the Claude Code instance working on the **Unity side** on a separate machine. Training and export
run on a Linux machine (Isaac Lab); results reach you through this repo. Read `CLAUDE.md` too. This file
covers what you need for Unity. Last updated 2026-11-07.

Sync: `git clone https://github.com/longwindwang1/Break_my_bot.git` (branch `main`). Put the Unity project in
`unity/`; `.gitignore` already excludes Unity's Library/Temp/Obj/Build/Logs/UserSettings. Commit Unity work
to a branch or `main` only when the user asks, and `git pull` before starting: new ONNX models and
references arrive from the Linux machine.

## 1. Project in brief

10-week course project (Physical AI / world models / games). A human player builds obstacle levels and
disrupts a **Unitree Go2** robot dog in real time. The robot walks with an RL policy (trained in Isaac
Lab, exported to ONNX), and a learned world model predicts outcomes and flags physics changes
("surprisal") so the robot can adapt.

Pipeline: **Isaac Lab + PyTorch (train) → ONNX (export) → Unity + Inference Engine (play)**. Unity runs the
policy inside Unity physics, which also serves as our sim-to-sim transfer test (experiment 3).

Working rules that apply to you:
- You set up the Unity editor and scenes (terrain, model import, components, physics settings) through the Unity MCP server, and write C# scripts under `unity/Assets/Scripts`. If the MCP can't do something, tell the user exactly what to set up by hand.
- Ask before installing packages.
- `configs/robot_interface.yaml` is the single source of truth for obs ordering, action scaling, PD gains,
  dt and decimation. Unity C# must **read it** (or a generated copy), not hardcode values. If something in
  it looks wrong, report it. Don't change it on the Unity side: any change goes through the Isaac side and
  its parity tests (`export/test_parity.py`).

## 2. Status

| Milestone | State |
|---|---|
| W1–2 baseline | Done. Go2 PPO policies, flat and rough (v1) |
| W3 smoke test, Python side | Done. Interface yaml, ONNX export, parity tests (Isaac ↔ ONNX) all pass |
| W3 smoke test, Unity side | Done (2026-11-07). Flat, rough and v2 walk in Unity; §8 levels 1–4 pass. Results, model notes and open questions: `experiments/w3_unity/README.md` |
| Robust policy v2 | Trained (friction/ice, pushes, payload, hidden pits). `export/models/go2_v2.onnx` and reference `go2_v2.json` are in the repo; parity tests pass (see §9). Runs in Unity (`W3_Rough` scene) |
| W4–5 world model | Trained and exported: `export/models/go2_wm.onnx`, spec in the yaml `world_model`, Unity reference traces `go2_wm*.json`. See §11. Needs its own buffer, surprisal + CUSUM in C#, and a Unity recalibration |
| Terrain-risk ("vision") model, phase 1 | Trained and exported: `export/models/go2_terrain_risk.onnx`, spec in the yaml `terrain_risk`, reference windows `go2_terrain_risk.json`. See §12. Needs a ray-cast grid, the ONNX, a HUD heat map and the slowdown trigger (v2 only) |
| W6–8 game, W9–10 results | Later |

## 3. Your task: W3 Unity smoke test

Goal: **the v1 flat policy walks in Unity physics, driven by a velocity command, and the interface is
verified to match Isaac Lab.**

Definition of done:
1. The Go2 is imported into Unity from the official URDF as an ArticulationBody, with physics settings as in §7.
2. A C# runtime runs the controller exactly as specified in §5: obs builder, ONNX inference, action → target, DC motor torque.
3. Verification levels 1–3 in §8 pass. Level 4 (closed loop) walks stably on flat ground and follows keyboard/gamepad commands.
4. Results go in `experiments/w3_unity/README.md`: what was run, Unity and package versions, test results.

Start with the **flat** policy (48-dim obs, no height scan). The rough policy additionally needs the
height-scan raycasts (§5.6). Do it after flat works.

## 4. Files you need

| Path | What |
|---|---|
| `configs/robot_interface.yaml` | The interface contract (timing, joint order, default pose, action, actuator, obs layout, height scan, friction, commands, policy specs) |
| `export/models/go2_flat.onnx` | v1 flat policy. Input `obs` float[1,48], output `actions` float[1,12]. Opset 18. Ops: Gemm, Elu |
| `export/models/go2_rough.onnx` | v1 rough policy. Input `obs` float[1,235] |
| `export/models/go2_v2.onnx` | v2 robust policy (distilled student). Input `obs` float[1,1360], see §9 |
| `experiments/w3_reference/go2_flat.json`, `go2_rough.json`, `go2_v2.json` | Isaac Lab reference rollouts (500 policy steps = 10 s, command vx = 0.5). Every array is `{"shape": [...], "data": [...]}` flattened row-major, so it is JsonUtility-friendly. Floats round-trip exactly to float32 |
| `experiments/w3_reference/README.md` | Field-by-field description of the reference files |
| `export/test_parity.py` → `build_obs()` | **Reference implementation** of the observation builder (numpy). Port this logic to C# |
| `CLAUDE.md` | Project-wide context |

ONNX import: Unity **Inference Engine / Sentis** (package `com.unity.ai.inference`). Version 2.6+
officially supports opset ≤ 25. The 2.2/2.3 docs only list opset 7–15. If the import complains, ask the
user to re-export on the Linux machine with
`python export/export_onnx.py --run <log_dir> --variant flat --opset 15` (verified identical outputs).

## 5. The interface contract (summary; the yaml is authoritative)

### 5.1 Control loop timing
- Physics: `Time.fixedDeltaTime = 0.005` (200 Hz). Policy: every **4** physics steps (50 Hz, `policy_dt = 0.02`).
- Per policy step (same order as Isaac Lab):
  ```
  every 4th FixedUpdate (before applying torques):
      obs      = BuildObs(current state, command, last_action)   // state AFTER the previous 4 physics steps
      a        = policy(obs)                                     // raw output, 12 floats, no clipping
      q_target = default_joint_pos + 0.25 * a
      last_action = a
  every FixedUpdate:
      tau = DCMotor(q_target, q, qd)   // recomputed with the current q, qd every physics step
      apply tau as joint effort
  ```
- On reset: `last_action = 0`, joints at `default_joint_pos`, zero velocities, base 0.4 m above the ground.

### 5.2 Joint order (policy order) — **grouped by joint type, not by leg**
```
0 FL_hip   1 FR_hip   2 RL_hip   3 RR_hip
4 FL_thigh 5 FR_thigh 6 RL_thigh 7 RR_thigh
8 FL_calf  9 FR_calf 10 RL_calf 11 RR_calf      (names end in "_joint")
default_joint_pos = [0.1, -0.1, 0.1, -0.1,  0.8, 0.8, 1.0, 1.0,  -1.5, -1.5, -1.5, -1.5]   (rad)
```
Map Unity ArticulationBodies to this order **by joint name**, never by hierarchy order.

### 5.3 Action
`q_target = default_joint_pos + 0.25 * a`, held for all 4 physics steps.

### 5.4 Actuator: explicit DC motor (Isaac Lab `DCMotor`)
Compute the torque yourself every physics step and apply it as joint effort. Set the ArticulationBody
drive stiffness and damping to **0**: the PhysX drive must not add its own PD.
```
Kp = 25, Kd = 0.5, effort_limit = 23.5, saturation_effort = 23.5, velocity_limit = 30   (SI)
tau     = Kp * (q_target - q) - Kd * qd
v_e     = velocity_limit * (1 + effort_limit / saturation_effort)
qd_c    = clamp(qd, -v_e, v_e)
tau_max = min( saturation_effort * ( 1 - qd_c / velocity_limit),  effort_limit)
tau_min = max( saturation_effort * (-1 - qd_c / velocity_limit), -effort_limit)
tau     = clamp(tau, tau_min, tau_max)
```

### 5.5 Observation (no scaling; only the height scan is clipped)
Concatenated in this order. All vectors are in the **Isaac Lab base frame** (x forward, y left, z up). See §6.

| idx | term | dim | definition |
|---|---|---|---|
| 0–2 | base_lin_vel | 3 | base **COM** linear velocity, world → base frame |
| 3–5 | base_ang_vel | 3 | base angular velocity in the base frame |
| 6–8 | projected_gravity | 3 | `R_base^T * (0, 0, -1)`; upright = (0, 0, -1) |
| 9–11 | velocity_commands | 3 | (vx, vy, wz) command, base frame |
| 12–23 | joint_pos | 12 | `q - default_joint_pos` |
| 24–35 | joint_vel | 12 | `qd` |
| 36–47 | actions | 12 | **raw** policy output of the previous step (0 after reset) |
| 48–234 | height_scan | 187 | rough/v2 only, see §5.6 |

Command limits in training: vx, vy ∈ [-1, 1] m/s, wz ∈ [-1, 1] rad/s. In Unity, feed (vx, vy, wz)
directly from the input device. Heading mode was training-only.

### 5.6 Height scan (rough / v2 only)
- 17 × 11 = 187 rays on a grid of 1.6 m (x) × 1.0 m (y), resolution 0.1 m, centred on the base.
- Ray origins follow the base **position and yaw only**, ignoring roll and pitch. They start 20 m above the base and cast straight down (world −z), hitting the **terrain only**.
- Ray order "xy" (x fastest): ray k has ix = k % 17, iy = k / 17, offset (x, y) = (−0.8 + 0.1·ix, −0.5 + 0.1·iy) in the base yaw frame.
- Value: `clip(base_z − hit_z − 0.5, −1, 1)`. Here base_z is the base height itself, not including the +20 m.

### 5.7 Friction
Terrain friction 1.0, **combine mode Multiply**, so the effective contact friction equals the foot
friction. v1 used 0.8 static / 0.6 dynamic on all robot bodies. In Unity, use a PhysicMaterial with
Multiply friction combine on both robot and ground. "Ice" in the game = a low foot × ground product.

### 5.8 Timing of the first observation
In the reference rollouts, `obs[t]` is computed from the state recorded at index t, before action t is
applied. `last_action` in `obs[t]` = `actions[t-1]` (0 at t = 0).

## 6. Coordinate conventions (Isaac Lab ↔ Unity)
Isaac / URDF: right-handed, x forward, y left, z up. Unity: left-handed, x right, y up, z forward. The
Unity URDF importer maps ROS → Unity positions as `(x, y, z)_unity = (−y, z, x)_ros`.

Build the observation in Unity like this, then **verify every rule numerically** (§8, level 1):
- Polar vectors (linear velocity, gravity): `v_base = Quaternion.Inverse(baseRot) * v_world`, then
  `isaac = (v_base.z, −v_base.x, v_base.y)`.
- Axial vectors (angular velocity): the handedness flip adds a sign:
  `w_base = Quaternion.Inverse(baseRot) * w_world`, then `isaac = (−w_base.z, w_base.x, −w_base.y)`.
  Check: a robot yawing left (counter-clockwise seen from above) must give isaac wz > 0.
- Gravity: `g_world_unity = (0, −1, 0)` → base frame → polar rule. An upright robot gives (0, 0, −1).
- Joint angles: check that each Unity `jointPosition` has the same sign as the URDF joint. The importer's
  axis handling can flip it. Test: put the robot in `default_joint_pos` and compare with the Isaac pose
  (§8, level 1, using `body_link_pos_w`). Also check that positive applied effort increases `jointPosition`.
- Base velocity: use the COM velocity of the base body (ArticulationBody.velocity should be at the COM;
  if in doubt use the point velocity at `worldCenterOfMass`).

## 7. Robot model and physics settings
- **URDF**: the official Go2 URDF (unitree_ros, `robots/go2_description`). Isaac Lab simulates
  `go2.usd` from NVIDIA Nucleus, presumably converted from that URDF (not verified). The reference files include the
  physical model **as Isaac simulates it**: `body_names`, `body_mass`, `body_inertia`, `body_com_pose`,
  `joint_pos_limits`, `joint_vel_limits`, `joint_effort_limits`, `joint_armature`,
  `joint_friction_coeff`. Compare the imported URDF against these and report differences, especially
  masses, inertias and foot collision.
- Key numbers from the reference files (Isaac's simulated model):
  - Total mass 15.02 kg. base 6.921, each hip 0.678, thigh 1.152, calf 0.154, foot 0.040. `Head_upper` / `Head_lower` are 0.001 each.
  - 19 bodies in Isaac order: base, FL/FR_hip, Head_upper, RL/RR_hip, thighs, Head_lower, calves, feet.
  - Joint limits (rad): hip ±1.047, thigh [−1.571, 3.491], calf [−2.723, −0.838].
  - PhysX joint effort limit 1e9, so torque limiting happens **only** in the DC-motor code (§5.4). PhysX joint velocity limit 30.1 rad/s for hips and thighs, 15.7 rad/s for calves (`joint_vel_limits`), armature 0, joint friction 0.
  - Feet sit about 0.022 m above the ground in stance, so the foot collision sphere radius is about 0.022 m.
- ArticulationBody, floating base (root not immovable). Self-collision **off** (Isaac: disabled).
- Solver: Isaac uses 4 position iterations and 0 velocity iterations. Use 4 / 1 in Unity (minimum).
- No linear or angular damping on links. Joint friction 0. Drive stiffness and damping 0 (torques come from §5.4).
- Isaac `max_depenetration_velocity = 1.0`.
- Joint armature as in `joint_armature`, if Unity exposes it.

## 8. Verification protocol (do these in order)
Write these as Unity Edit Mode / Play Mode tests or a debug scene. Report max errors.

1. **Kinematics and conventions.** For a few frames t of `go2_flat.json`: set the base pose
   (`root_link_pos_w`, `root_link_quat_w` (w, x, y, z), converted to Unity) and the joint positions
   (`joint_pos`, policy order). Compare Unity link world positions with `body_link_pos_w` (by
   `body_names`). Tolerance about 1 mm. This catches joint sign and axis errors and URDF differences.
2. **ONNX parity.** Feed `obs[t]` for all 500 steps into the Unity model and compare with `actions[t]`.
   Tolerance 1e-4: Isaac ran the policy on a GPU, and CPU onnxruntime already matches it to < 1e-4.
3. **Observation builder.** From the recorded state at each step t (`root_link_quat_w`,
   `root_com_lin_vel_w`, `root_com_ang_vel_w`, `joint_pos`, `joint_vel`, `command`, previous `actions`),
   build the obs with your C# code, using the world-frame inputs **after converting them into Unity
   coordinates**, so the conversion code is exercised. Compare with `obs[t]`. Tolerance 1e-5. Python
   reproduces this to about 2e-7 (`test_obs_layout`).
4. **Closed loop.** Run in Unity physics with command vx = 0.5. The Isaac reference travels about 4.6 m in
   10 s at 0.47 m/s, and the flat policy's base sits low, at about 0.20 m (§10). Then keyboard/gamepad control.
   Trajectories will drift from the reference (different physics engine). That gap is experiment 3, not a bug.

## 9. Robust policy v2 (available)
Specified in `configs/robot_interface.yaml` under `policies.v2`.
- Model: `export/models/go2_v2.onnx`. Reference: `experiments/w3_reference/go2_v2.json`, recorded with foot μ = 1.0 on `random_rough`; it walks about 4.1 m in 10 s with the base about 0.37 m above the terrain.
- Training notes: `experiments/v2_teacher/` and `experiments/v2_student/`. Isaac Lab results per scenario: `experiments/v2_eval/`.
- Run the same verification levels as §8 with the v2 files. Level 2 now also checks the history buffer.
- Deployed network = distilled student. Input 1360 = the 235-dim obs above + a **proprio history** of 25
  steps × 45 dims (base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, actions; no command).
- History layout: term by term, each term block oldest → newest (current step last). After a reset —
  and when "clearing the adaptation history" after a surprisal alarm — **fill all 25 slots with the
  current frame** (not zeros).
- **Cautious gait** = clamp the command to |v| ≤ 0.3 m/s, |wz| ≤ 0.5 rad/s (no separate network).
- Trained with foot friction 0.05–1.25 (changing mid-episode), pushes, payload −1 to +3 kg, ±10 % motor
  gains, and pits where the height scan sometimes lies. Unity game mechanics like ice patches, pushes and
  breakable ice should stay within these ranges at first.

## 10. Known issues / notes
- **The v1 flat policy walks crouched** (base about 0.20 m vs about 0.40 m for rough). The stock flat config
  has no base-height reward. It still tracks commands. v2 adds a base-height reward.
- v1 rough requires the height scan. Without correct raycasts it will behave badly.
- base_lin_vel is not directly measurable on the real robot. That's fine in simulation.
- If anything in this file disagrees with `configs/robot_interface.yaml` or `export/test_parity.py`, those win. Tell the user so this file gets fixed.

## 11. World model (W4–5, available)
A learned dynamics monitor that runs next to the policy at 50 Hz and **never produces actions**.
- The dynamics part is trained on **normal physics only** (foot μ 0.6–1.25, no pushes), so ice, pushes and hidden pits stay surprising.
- The fall head was refit on the full randomization, ice and pushes included.
- At policy step t it predicts the next state s_{t+1}, the base displacement and the fall probability.
- At step t+1 the observed state is scored against that prediction (surprisal). A CUSUM over the scores raises an alarm on a physics change: ice, a push, or a hidden pit.
- Spec: `configs/robot_interface.yaml` → `world_model` (input layout, outputs, surprisal and CUSUM parameters).
- Training and evaluation notes: `experiments/w4_wm_data/`, `experiments/w4_wm_ensemble/`, `experiments/w5_surprisal/`.

| Path | What |
|---|---|
| `export/models/go2_wm.onnx` | Input `wm_in` [1, 471]. Outputs `state_mean`, `state_var` [1, 33]; `disp_mean`, `disp_var` [1, 4]; `fall_prob` [1, 3]; `disagreement` [1, 1]. Opset 18, about 17 MB. The 5 ensemble members are unrolled into Gemm chains; it also uses Sigmoid, Mul, Exp, Softplus, ReduceMean, Gather and Concat |
| `experiments/w3_reference/go2_wm.json` | Every step of the v2 reference rollout (`go2_v2.json`): `wm_in`, all six outputs, `z`, `S`, `x`, `scored`, `spike`, `g`, `alarm`. Clean rollout: no alarm |
| `experiments/w3_reference/go2_wm_mudrop.json` (+ `go2_v2_mudrop.json`) | Same format; the foot friction drops to 0.05 at `friction_drop.at_step` |

### 11.1 Input `wm_in` (471 = 270 + 12 + 187 + 2)
1. **History (270).** The world model's **own** buffer of 6 frames: the current frame and the 5 before it.
   - A frame is the 45-d proprioception: `base_lin_vel`, `base_ang_vel`, `projected_gravity`, `joint_pos`, `joint_vel`, `actions` (the previous action), with the definitions in §5.5.
   - Flatten term by term, each term oldest → newest, exactly like the v2 `proprio_history` (§9) but with 6 frames.
   - On an episode reset, fill all 6 slots with the first frame.
   - **Do not clear it on a surprisal alarm.** Only the policy's adaptation history is cleared on an alarm. Clearing this buffer too would feed the model copied frames mid-stride and trigger the next alarm.
   - Keep this buffer for v1 too (v1 has no history of its own).
2. **Action (12).** a_t, the raw output the policy just produced for this step.
3. **Height scan (187).** The height-scan term of this step (§5.6).
4. **Policy one-hot (2).** `[v1, v2]`, in the order of `world_model.policies`. Only the fall head reads it.

### 11.2 Per policy step
```
obs_t  = BuildObs(...)                 // as now
a_t    = policy(obs_t)
if step_in_episode > 0: score(s_t)     // s_t = first 33 dims of the current frame (state_terms)
wm_buffer.push(frame_t)                // after a reset: fill all 6 slots
pred   = world_model(wm_in(wm_buffer, a_t, scan_t, onehot))   // keep state_mean, state_var for the next step
```
Scoring s_{t+1} against the prediction from step t (`world_model.surprisal` in the yaml):
```
z_d = ((s_d - state_mean_d) / sqrt(state_var_d + var_floor_d) - z_bias_d) / z_scale_d
S   = sum_d weights_d * 0.5 * (min(z_d^2, clip_z2) - 1)                       // statistic "norm"
      (statistic "nll": 0.5 * (min(z_d^2, clip_z2) + log(2 pi (state_var_d + var_floor_d))) instead)
x     = (S - m0) / s0                  // m0, s0, k, h, spike_z: per_policy.<policy>
g     = max(0, g + x - k)
spike = max over d in spike_dims of |z_d| > spike_z      // spike_dims = base_lin_vel x, y: pushes
if g > h or spike: ALARM, g = 0
```
- g = 0 on every episode reset.
- Nothing is scored while the episode step is below `warmup_steps`.
- On an alarm (W6–8):
  - clear the **policy's** adaptation history (§9) by filling it with the current frame;
  - switch to the cautious gait;
  - leave the world-model buffer alone.

### 11.3 Verification (same idea as §8)
1. **ONNX.** Feed `go2_wm.json`'s `wm_in` rows and compare all six outputs.
   - Tolerance: `state_mean` and `disp_mean` 1e-4 absolute; the variance outputs and `disagreement` 1e-3 relative.
2. **Input builder.** Build `wm_in` from the `go2_v2.json` reference state with your C# code. Compare with `go2_wm.json` `wm_in` (1e-5).
3. **Surprisal and CUSUM.** From the stored states and outputs, compute `z`, `S`, `x`, `spike`, `g` and `alarm` and compare (1e-5; `spike` and `alarm` exactly).
   - `go2_wm.json` must stay alarm-free.
   - `go2_wm_mudrop.json` shows what the Isaac model does after the friction drop.
4. **Closed loop in Unity**, with the recalibration below.

### 11.4 Recalibration in Unity (expected to be necessary)
- **Why.** The thresholds were set on Isaac Lab data for one false alarm per `target_arl_steps` steps (10 min) of clean walking. Unity's physics differs: W3 measured a 15–18 % longer gait period. So in Unity the residuals are larger and shaped differently per dim, and the Isaac thresholds will raise false alarms.
- **Procedure.** Calibrate on clean walking in Unity: varied commands, flat and rough ground, μ ≥ 0.6, no pushes.
  1. Log z_d over at least 20 min of simulated walking. Headless stepping can run faster than real time.
  2. Set `z_bias_d` = median(z_d) and `z_scale_d` = 1.4826 · MAD(z_d).
  3. Recompute m0 and s0 of S with the corrected z. Keep k.
  4. Set `spike_z` to the 1 − 1/(2 · `target_arl_steps`) quantile of the spike statistic.
  5. Choose h as the smallest value with at most 1 alarm (CUSUM or spike) per `target_arl_steps` scored steps on that log. To be credible this needs at least 2 × `target_arl_steps` scored steps.
- **Where to store it.** Write the values to `experiments/w3_unity/wm_calibration_unity.json`, which the runtime prefers over the yaml values. Do not edit the yaml. Report the per-dim bias and scale: they are data for experiment 3.


## 12. Terrain-risk ("vision") model (W6 phase 1, available)
**What it predicts.** From terrain height alone, the probability that the robot falls within 0.5 s or 1 s if its base were at a given cell, facing along the grid's x axis, under the current command and policy.
- It warns *before* the robot gets there. This complements the world model (§11), which notices a problem only after the fact.
- "Vision" here means height from vertical ray casts, not a camera image, so Unity and Isaac see the same numbers.
- Spec: `configs/robot_interface.yaml` → `terrain_risk`. Results: `experiments/w6_terrain_risk/README.md`.

**Result in Isaac Lab (v2, unseen terrain).** Slowing down where the model sees risk ahead cut falls per 100 m by 36–46 % and cost 14–24 % of speed. Random slowdown with the same cautious time did not help. v1 gets worse when slowed, so its threshold is `null`.

| Path | What |
|---|---|
| `export/models/go2_terrain_risk.onnx` | 113 KB. Inputs `height` [1, 1, H, W] (H, W dynamic) and `cond` [1, 5] = (vx, vy, wz, policy_v1, policy_v2). Output `risk` [1, 2, H − 16, W − 16], probabilities for 25 and 50 steps. Ops: Conv (dilations 1, 1, 2, 4, no padding), MatMul, Add, Unsqueeze, Relu, Mul, Sigmoid |
| `experiments/w3_reference/go2_terrain_risk.json` | 8 windows cut from Isaac terrain: `height` [37][47], `cond`, `risk` [2][21][31], `corridor_mask` [21][31], `risk_ahead`, `threshold`, `slow_down` |

### 12.1 Egocentric window (yaml `terrain_risk.window`)
- 47 × 37 vertical ray casts, as for the height scan (§5.6): the origin follows the base position and yaw, roll and pitch are ignored, and rays hit the terrain only.
- Point (i, j) sits at (x0 + 0.1·i, y0 + 0.1·j) in the base yaw frame (Isaac axes: x forward, y left), with x0 = −1.3 and y0 = −1.8. It goes into `height[0, 0, j, i]`.
- The value is the hit z in metres. Any constant offset is fine; subtract the centre height first for float precision.
- Output cell (i, j) is centred at (x0 + 0.1·(i + 8), y0 + 0.1·(j + 8)). The 31 × 21 output covers x ∈ [−0.5, 2.5] m and y ∈ [−1, 1] m ahead of the robot.
- For the HUD, convert these Isaac-frame cell centres with `IsaacFrame`.
- Cost: about 33M multiply-adds per window, a few milliseconds on CPU. Run it at the update period (0.2 s) with `Schedule`, not inside FixedUpdate.

### 12.2 Slowdown trigger (yaml `terrain_risk.slowdown`)
Every 0.2 s:
1. Take the max of risk channel 0 over the output cells whose centres lie within 0.2 m of the segment from the base along the (vx, vy) command, from 0 to 1.2 m ahead. If |(vx, vy)| ≤ 0.1, use the cells within 0.4 m of the base instead.
2. If that max exceeds `threshold.<policy>`, apply the cautious gait for 1 s. The cautious gait scales |(vx, vy)| down to 0.3 m/s and clamps |wz| to 0.5 (`policies.v2.cautious_gait`).
3. Use the player's original command for the risk, never the slowed one; otherwise the trigger oscillates.

The window only looks forward, so the trigger is valid for vx ≥ 0.3, |vy| ≤ 0.2 and |wz| ≤ 0.5. This rule and the surprisal alarm (§11) both lead to the cautious gait. Either can trigger it.

### 12.3 Whole-level danger map
Run the same ONNX on a larger height grid, for example the level at 0.1 m (H and W are dynamic). Output cell (i, j) is the risk of the input cell (i + 8, j + 8) for heading +x.
- For other headings, rotate the grid. A useful summary is the max over 4–8 headings.
- Compute it when the level loads or changes, not every frame.

### 12.4 Verification
1. **ONNX.** Feed each reference window's `height` and `cond` and compare `risk` (tolerance 1e-5).
2. **Corridor and trigger.** From `cond[0:2]`, rebuild `corridor_mask`, then `risk_ahead` and `slow_down`. All must match exactly.
3. **Ray grid.** In a Unity scene with known geometry (for example a single 0.15 m step), check that the ray grid lands at the documented positions and order.

The thresholds were set on Isaac Lab terrain. Player-built levels and Unity's slower gait are a distribution shift, so watch how often the trigger fires in Unity. If it fires far more than about 20–45 % of walking time, report it.
