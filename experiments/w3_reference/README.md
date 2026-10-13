# W3 Isaac Lab reference rollouts

Reference data for interface parity: what Isaac Lab fed the policy and the physics state it computed that from.
The W3 Unity observation builder must rebuild `obs` from its own state the same way (`build_obs` in
`export/test_parity.py` is the reference implementation).

| file | policy (checkpoint) | task | terrain | command (vx, vy, wz) |
|---|---|---|---|---|
| `go2_flat.npz`  | `w1_go2_flat_baseline/model_299.pt`   | `Isaac-Velocity-Flat-Unitree-Go2-Play-v0`  | plane | (0.5, 0, 0) |
| `go2_rough.npz` | `w1_go2_rough_baseline/model_1499.pt` | `Isaac-Velocity-Rough-Unitree-Go2-Play-v0` | `random_rough` only (0–6 cm relief) | (0.5, 0, 0) |

Each covers 500 policy steps (10 s), 1 env, seed 0, clean observations (Play config), and the command is fixed (heading mode off, no resampling). Neither rollout resets partway through.

Re-record with (from the repo root):
```
PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python isaac/record_reference.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 --load_run 2026-10-12_21-19-01_w1_go2_flat_baseline \
  --out experiments/w3_reference/go2_flat.npz
PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python isaac/record_reference.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-Play-v0 --load_run 2026-10-12_21-27-17_w1_go2_rough_baseline \
  --sub_terrain random_rough --out experiments/w3_reference/go2_rough.npz
```
`--sub_terrain` is needed for rough because the stairs/boxes sub-terrains spawn the robot on a flat platform,
where every height-scan ray hits z = 0.

## Fields (T = 500; world frame z-up; quaternions w, x, y, z)
Per step, captured **before** the action is applied, so it matches the state `obs[t]` was computed from:
- `obs` [T, 48|235]: the policy input
- `command` [T, 3]
- `joint_pos`, `joint_vel` [T, 12]: in `joint_names` order
- `root_link_pos_w` [T, 3], `root_link_quat_w` [T, 4]
- `root_com_lin_vel_w`, `root_com_ang_vel_w` [T, 3]: base_lin_vel in the obs uses the **COM** velocity
- rough only: `scan_sensor_z` [T] (= base z), `scan_hits_z` [T, 187], `scan_hits_xy` [T, 187, 2]

Per step, captured **after** the step:
- `actions` [T, 12]: raw policy output a_t (Isaac Lab ran it on the GPU)
- `joint_pos_target` [T, 12] (= default + 0.25·a_t); `applied_torque` [T, 12]: DC-motor output at the last physics substep
- `dones` [T]

Constants: `joint_names`, `default_joint_pos`, `stiffness`, `damping`, `effort_limit`, `velocity_limit`,
`saturation_effort`, `physics_dt`, `step_dt`, `task`, `checkpoint`, `seed`, `command_fixed`, `sub_terrain`.

## Observations from recording
- **Joint order is grouped by joint type** (4 hips, then 4 thighs, then 4 calves), not by leg.
- Both policies track the 0.5 m/s command: about 4.6 m travelled in 10 s.
- **The flat policy walks crouched.** The base sits at about 0.20 m, with thighs at 0.9–1.6 rad and calves at about −1.85 rad. The rough policy stands about 0.40 m above the terrain, with straighter legs.
  The stock flat config has no base-height reward. This still needs a visual check in `play.py`.
