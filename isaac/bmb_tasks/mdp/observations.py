"""Observation terms: the training-time deceptive height scan and the teacher's privileged inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

from . import state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "height_scan_deceptive",
    "foot_friction",
    "added_base_mass",
    "base_com_offset",
    "actuator_gain_scale",
    "scan_deception_active",
    "foot_contact_forces",
]


def height_scan_deceptive(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """mdp.height_scan, except that during a deception window (state.deception_active) every ray reports
    the median reading, i.e. flat ground at the robot's current ground level. Training-only: at
    deployment this is the plain height scan."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    scan = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    active = state.deception_active(env)
    if torch.any(active):
        flat = scan.median(dim=1, keepdim=True).values.expand_as(scan)
        scan = torch.where(active[:, None], flat, scan)
    return scan


def foot_friction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """[4] static friction of each foot (= effective contact friction, terrain mu 1.0 with multiply)."""
    return state.foot_friction(env)


def added_base_mass(env: ManagerBasedRLEnv) -> torch.Tensor:
    """[1] payload added to the base (kg)."""
    return state.added_base_mass(env)


def base_com_offset(env: ManagerBasedRLEnv) -> torch.Tensor:
    """[3] base centre-of-mass offset (m)."""
    return state.base_com_offset(env)


def actuator_gain_scale(env: ManagerBasedRLEnv, actuator_name: str = "base_legs") -> torch.Tensor:
    """[24] per-joint Kp and Kd relative to nominal, minus 1 (randomize_actuator_gains writes the actuator model)."""
    robot = env.scene["robot"]
    act = robot.actuators[actuator_name]
    ids = act.joint_indices
    kp = act.stiffness / robot.data.default_joint_stiffness[:, ids]
    kd = act.damping / robot.data.default_joint_damping[:, ids]
    return torch.cat([kp, kd], dim=1) - 1.0


def scan_deception_active(env: ManagerBasedRLEnv) -> torch.Tensor:
    """[1] 1.0 while the height scan is deceptive."""
    return state.deception_active(env).float().unsqueeze(1)


def foot_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """[12] net contact force on each foot (world frame, N)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w[:, sensor_cfg.body_ids].flatten(1)
