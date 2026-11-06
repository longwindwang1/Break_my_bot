"""Reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["base_height_above_terrain_l2"]


def base_height_above_terrain_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared error of base height above the terrain below it (median of the height-scan hits).

    Unlike mdp.base_height_l2 (mean of hits) this ignores rays that missed (inf) and is robust to
    pits and steps inside the scan footprint."""
    asset = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    base_z = asset.data.root_pos_w[:, 2]
    hits = sensor.data.ray_hits_w[..., 2]
    hits = torch.where(torch.isfinite(hits), hits, (base_z - target_height)[:, None])
    ground = hits.median(dim=1).values
    return torch.square(base_z - ground - target_height)
