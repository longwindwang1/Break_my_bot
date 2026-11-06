"""Curriculum terms (here: logging only; the friction curriculum itself lives in randomize_foot_friction)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["friction_stats"]


def friction_stats(env: ManagerBasedRLEnv, env_ids: torch.Tensor, event_term: str = "foot_friction_reset") -> dict:
    """Log the friction curriculum: current sampling lower bound, mean foot friction, and the mean terrain
    level of low-friction envs (the terrain curriculum demotes robots that walk too little, so low
    friction can drag them to easy terrain)."""
    mu = state.foot_friction(env).mean(dim=1)
    levels = env.scene.terrain.terrain_levels.float()
    low = mu < 0.3
    term = env.event_manager.get_term_cfg(event_term).func
    return {
        "lower_bound": float(getattr(term, "lower_bound", float("nan"))),
        "mean_mu": mu.mean().item(),
        "low_mu_terrain_level": levels[low].mean().item() if torch.any(low) else float("nan"),
        "terrain_level": levels.mean().item(),
    }
