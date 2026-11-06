"""Record a deterministic Isaac Lab rollout of a trained Go2 policy as a parity reference.

For every policy step t it stores the observation the policy saw, the action it produced and the
raw physics state the observation was computed from, so another simulator (Unity) can rebuild the
observation from its own state and compare step by step.

Run from the repo root (Isaac Lab resolves logs/ relative to the cwd):
    PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python isaac/record_reference.py \
        --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
        --load_run 2026-10-12_21-19-01_w1_go2_flat_baseline \
        --out experiments/w3_reference/go2_flat.npz
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.expanduser("~/IsaacLab/scripts/reinforcement_learning/rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record a reference rollout for interface parity tests.")
parser.add_argument("--task", type=str, required=True, help="Play task, e.g. Isaac-Velocity-Flat-Unitree-Go2-Play-v0.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--out", type=str, required=True, help="Output .npz path.")
parser.add_argument("--steps", type=int, default=500, help="Policy steps to record (50 Hz).")
parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0], help="Fixed command: vx vy wz.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
parser.add_argument(
    "--foot_friction",
    type=float,
    default=None,
    help="BMB tasks: fix the foot friction to this value (no mid-episode changes).",
)
parser.add_argument(
    "--sub_terrain",
    type=str,
    default=None,
    help="Keep only this generated sub-terrain (e.g. random_rough). Stairs/boxes spawn on a flat centre platform.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import isaac.bmb_tasks  # noqa: E402, F401  (registers BMB-* tasks)
from isaaclab_tasks.utils.hydra import hydra_task_config


def np32(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.seed = args_cli.seed
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = 1

    # Fixed, non-resampled command so the rollout is reproducible and easy to replay elsewhere.
    vx, vy, wz = args_cli.cmd
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.rel_heading_envs = 0.0
    cmd_cfg.resampling_time_range = (1.0e6, 1.0e6)
    cmd_cfg.ranges.lin_vel_x = (vx, vx)
    cmd_cfg.ranges.lin_vel_y = (vy, vy)
    cmd_cfg.ranges.ang_vel_z = (wz, wz)
    cmd_cfg.debug_vis = False

    terrain_gen = env_cfg.scene.terrain.terrain_generator
    if args_cli.sub_terrain is not None and terrain_gen is not None:
        sub = terrain_gen.sub_terrains[args_cli.sub_terrain]
        sub.proportion = 1.0
        terrain_gen.sub_terrains = {args_cli.sub_terrain: sub}

    events = env_cfg.events
    # References use the nominal robot: no payload / COM / gain randomization, pushes or scan deception.
    for name in ("payload_mass", "payload_com", "actuator_gains", "push", "scan_deception"):
        if getattr(events, name, None) is not None:
            setattr(events, name, None)
    if args_cli.foot_friction is not None:
        for k in ("lower_start", "lower_end", "upper"):
            events.foot_friction_reset.params[k] = args_cli.foot_friction
        events.foot_friction_change = None

    # Observation groups the deployed actor consumes, concatenated in this order.
    obs_groups = agent_cfg.obs_groups if isinstance(agent_cfg.obs_groups, dict) else {}
    actor_groups = obs_groups.get("policy", ["policy"])

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = DistillationRunner if agent_cfg.class_name == "DistillationRunner" else OnPolicyRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    scanner = base_env.scene.sensors.get("height_scanner")
    actuator = robot.actuators["base_legs"]

    rec = {k: [] for k in [
        "obs", "actions", "command", "joint_pos", "joint_vel", "root_link_pos_w", "root_link_quat_w",
        "root_com_lin_vel_w", "root_com_ang_vel_w", "body_link_pos_w", "body_link_quat_w",
        "joint_pos_target", "applied_torque", "dones",
    ]}
    if scanner is not None:
        rec["scan_sensor_z"] = []
        rec["scan_hits_z"] = []
        rec["scan_hits_xy"] = []

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            # State that obs was computed from (nothing has stepped since the obs was produced).
            rec["obs"].append(np32(torch.cat([obs[g][0] for g in actor_groups])))
            rec["command"].append(np32(base_env.command_manager.get_command("base_velocity")[0]))
            rec["joint_pos"].append(np32(robot.data.joint_pos[0]))
            rec["joint_vel"].append(np32(robot.data.joint_vel[0]))
            rec["root_link_pos_w"].append(np32(robot.data.root_link_pos_w[0]))
            rec["root_link_quat_w"].append(np32(robot.data.root_link_quat_w[0]))
            rec["root_com_lin_vel_w"].append(np32(robot.data.root_com_lin_vel_w[0]))
            rec["root_com_ang_vel_w"].append(np32(robot.data.root_com_ang_vel_w[0]))
            rec["body_link_pos_w"].append(np32(robot.data.body_link_pos_w[0]))
            rec["body_link_quat_w"].append(np32(robot.data.body_link_quat_w[0]))
            if scanner is not None:
                rec["scan_sensor_z"].append(np32(scanner.data.pos_w[0, 2]))
                rec["scan_hits_z"].append(np32(scanner.data.ray_hits_w[0, :, 2]))
                rec["scan_hits_xy"].append(np32(scanner.data.ray_hits_w[0, :, :2]))

            actions = policy(obs)
            rec["actions"].append(np32(actions[0]))
            obs, _, dones, _ = env.step(actions)
            # Targets/torques produced by this action during the step just taken.
            rec["joint_pos_target"].append(np32(robot.data.joint_pos_target[0]))
            rec["applied_torque"].append(np32(robot.data.applied_torque[0]))
            rec["dones"].append(bool(dones[0]))

    out = {k: np.stack(v) for k, v in rec.items()}
    out.update(
        task=np.array(args_cli.task),
        checkpoint=np.array(os.path.relpath(resume_path)),
        sub_terrain=np.array(args_cli.sub_terrain or ""),
        foot_friction=np.array(np.nan if args_cli.foot_friction is None else args_cli.foot_friction, dtype=np.float32),
        actor_obs_groups=np.array(actor_groups),
        runner=np.array(agent_cfg.class_name),
        seed=np.array(args_cli.seed),
        command_fixed=np.array(args_cli.cmd, dtype=np.float32),
        joint_names=np.array(robot.joint_names),
        body_names=np.array(robot.body_names),
        # Physical model as simulated (for checking the URDF imported elsewhere), in body_names / joint_names order.
        body_mass=np32(robot.data.default_mass[0]),
        body_inertia=np32(robot.data.default_inertia[0]),  # [B, 9] row-major, about the COM, body axes
        body_com_pose=np32(robot.root_physx_view.get_coms()[0]),  # [B, 7] COM pos + quat (x, y, z, w) in link frame
        joint_pos_limits=np32(robot.data.joint_pos_limits[0]),
        joint_vel_limits=np32(robot.data.joint_vel_limits[0]),
        joint_effort_limits=np32(robot.data.joint_effort_limits[0]),
        joint_armature=np32(robot.data.joint_armature[0]),
        joint_friction_coeff=np32(robot.data.joint_friction_coeff[0]),
        default_joint_pos=np32(robot.data.default_joint_pos[0]),
        stiffness=np32(actuator.stiffness[0]),
        damping=np32(actuator.damping[0]),
        effort_limit=np32(actuator.effort_limit[0]),
        velocity_limit=np32(actuator.velocity_limit[0]),
        saturation_effort=np.array(getattr(actuator, "_saturation_effort", float("nan")), dtype=np.float32),
        physics_dt=np.array(base_env.physics_dt, dtype=np.float32),
        step_dt=np.array(base_env.step_dt, dtype=np.float32),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    np.savez_compressed(args_cli.out, **out)

    print(f"[INFO] Saved {args_cli.steps} steps to {args_cli.out}; episodes ended mid-rollout: {int(out['dones'].sum())}")
    print("[INFO] joint_names:", list(robot.joint_names))
    print("[INFO] default_joint_pos:", out["default_joint_pos"].tolist())
    print("[INFO] stiffness/damping:", out["stiffness"].tolist(), out["damping"].tolist())

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
