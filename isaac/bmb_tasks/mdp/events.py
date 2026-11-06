"""Randomization events that cache what they sample (see state.py) for the privileged observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from . import state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

__all__ = ["randomize_foot_friction", "randomize_base_mass_cached", "randomize_base_com_cached", "schedule_scan_deception"]


def _all_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    return torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device)


class randomize_foot_friction(ManagerTermBase):
    """Foot friction randomization with a curriculum on the lower bound, quantized to `num_levels` materials.

    Writes static friction mu to every collision shape of the foot bodies (dynamic = dynamic_ratio * mu,
    restitution 0) and caches mu per foot in ``state.foot_friction``. The terrain material has friction 1.0
    with combine mode "multiply", so the effective contact friction equals the foot value.

    - reset mode: all feet get the same new mu (a new surface).
    - interval mode with ``single_foot_prob`` > 0: with that probability only one random foot changes
      (stepping onto an ice patch), otherwise all feet change.

    The sampling lower bound moves linearly from ``lower_start`` to ``lower_end`` over ``curriculum_steps``
    env steps, counted from ``start_step`` (``common_step_counter`` restarts at 0 in every process, so
    distillation / play / eval configs pass ``start_step >= curriculum_steps``).

    Values are snapped to ``num_levels`` evenly spaced levels to bound the number of unique PhysX
    materials (PhysX allows at most 64k per scene).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        # Collision shapes are indexed per articulation in link order; find each foot's shape range.
        shapes_per_body = [
            self.asset._physics_sim_view.create_rigid_body_view(path).max_shapes
            for path in self.asset.root_physx_view.link_paths[0]
        ]
        if sum(shapes_per_body) != self.asset.root_physx_view.max_shapes:
            raise ValueError(f"shape count mismatch: {shapes_per_body} vs {self.asset.root_physx_view.max_shapes}")
        starts = np.cumsum([0] + shapes_per_body[:-1])
        body_ids = self.asset_cfg.body_ids
        if isinstance(body_ids, slice) or len(body_ids) != state.NUM_FEET:
            raise ValueError(f"expected {state.NUM_FEET} foot bodies, got {self.asset_cfg.body_names}")
        self.foot_shapes = [(int(starts[b]), int(starts[b] + shapes_per_body[b])) for b in body_ids]
        if any(s == e for s, e in self.foot_shapes):
            raise ValueError(f"a foot body has no collision shape: {list(zip(self.asset_cfg.body_names, self.foot_shapes))}")
        lo, hi = cfg.params["level_range"]
        self.levels = torch.linspace(lo, hi, cfg.params["num_levels"], device=env.device)
        self.lower_bound = cfg.params["lower_start"]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        level_range: tuple[float, float],
        num_levels: int,
        lower_start: float,
        lower_end: float,
        upper: float,
        curriculum_steps: int,
        start_step: int = 0,
        dynamic_ratio: float = 0.8,
        single_foot_prob: float = 0.0,
    ):
        env_ids = _all_env_ids(env, env_ids)
        n = len(env_ids)
        if n == 0:
            return
        progress = min(1.0, (env.common_step_counter + start_step) / max(1, curriculum_steps))
        self.lower_bound = lower_start + (lower_end - lower_start) * progress
        mu = self.lower_bound + (upper - self.lower_bound) * torch.rand(n, device=env.device)
        mu = self.levels[torch.argmin((mu[:, None] - self.levels[None, :]).abs(), dim=1)]

        cache = state.foot_friction(env)
        new = mu[:, None].repeat(1, state.NUM_FEET)
        if single_foot_prob > 0.0:
            single = torch.rand(n, device=env.device) < single_foot_prob
            foot = torch.randint(0, state.NUM_FEET, (n,), device=env.device)
            only_that_foot = torch.nn.functional.one_hot(foot, state.NUM_FEET).bool() | ~single[:, None]
            new = torch.where(only_that_foot, new, cache[env_ids])
        cache[env_ids] = new

        ids_cpu, new_cpu = env_ids.cpu(), new.cpu()
        materials = self.asset.root_physx_view.get_material_properties()
        for k, (s, e) in enumerate(self.foot_shapes):
            materials[ids_cpu, s:e, 0] = new_cpu[:, k : k + 1]
            materials[ids_cpu, s:e, 1] = new_cpu[:, k : k + 1] * dynamic_ratio
            materials[ids_cpu, s:e, 2] = 0.0
        self.asset.root_physx_view.set_material_properties(materials, ids_cpu)


def randomize_base_mass_cached(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    mass_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
):
    """Add a uniform random payload to the base (startup only), rescale its inertia, cache the added mass."""
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = _all_env_ids(env, env_ids)
    ids_cpu = env_ids.cpu()
    (body_id,) = asset_cfg.body_ids
    added = torch.empty(len(env_ids)).uniform_(*mass_range)
    masses = asset.root_physx_view.get_masses()
    default = asset.data.default_mass[ids_cpu, body_id].cpu()
    masses[ids_cpu, body_id] = default + added
    asset.root_physx_view.set_masses(masses, ids_cpu)
    inertias = asset.root_physx_view.get_inertias()
    ratio = (masses[ids_cpu, body_id] / default)[:, None]
    inertias[ids_cpu, body_id] = asset.data.default_inertia[ids_cpu, body_id].cpu() * ratio
    asset.root_physx_view.set_inertias(inertias, ids_cpu)
    state.added_base_mass(env)[env_ids] = added[:, None].to(env.device)


def randomize_base_com_cached(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
):
    """Shift the base centre of mass by a uniform random offset (startup only: it adds to the current COM)."""
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = _all_env_ids(env, env_ids)
    ids_cpu = env_ids.cpu()
    (body_id,) = asset_cfg.body_ids
    ranges = torch.tensor([com_range.get(k, (0.0, 0.0)) for k in "xyz"])
    offset = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand(len(env_ids), 3)
    coms = asset.root_physx_view.get_coms().clone()
    coms[ids_cpu, body_id, :3] += offset
    asset.root_physx_view.set_coms(coms, ids_cpu)
    state.base_com_offset(env)[env_ids] = offset.to(env.device)


class schedule_scan_deception(ManagerTermBase):
    """On reset, decide whether (and when) the height scan will lie during the new episode.

    Emulates breakable ice / hidden holes: during the window the scan reports flat ground, so the robot
    can step into a pit it did not see. Only envs whose terrain column is the ``pit_sub_terrain`` type are
    eligible (with a curriculum terrain generator, columns have fixed sub-terrain types); if the
    generator consists of that sub-terrain only, every env is eligible.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        terrain = env.scene.terrain
        gen = terrain.cfg.terrain_generator
        name = cfg.params["pit_sub_terrain"]
        num_cols = gen.num_cols if gen is not None else 1
        eligible = torch.zeros(num_cols, dtype=torch.bool, device=env.device)
        if gen is not None and list(gen.sub_terrains) == [name]:
            eligible[:] = True
        elif gen is not None and gen.curriculum and name in gen.sub_terrains:
            # Same column assignment as TerrainGenerator._generate_curriculum_terrains.
            names = list(gen.sub_terrains)
            proportions = np.array([c.proportion for c in gen.sub_terrains.values()], dtype=float)
            cumsum = np.cumsum(proportions / proportions.sum())
            for col in range(num_cols):
                eligible[col] = names[int(np.min(np.where(col / num_cols + 0.001 < cumsum)[0]))] == name
        self.eligible_cols = eligible

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        prob: float,
        duration_s: tuple[float, float],
        pit_sub_terrain: str,
        start_s: tuple[float, float] | None = None,
    ):
        env_ids = _all_env_ids(env, env_ids)
        start, end = state.deception_window(env)
        cols = env.scene.terrain.terrain_types[env_ids].clamp(max=len(self.eligible_cols) - 1)
        enabled = self.eligible_cols[cols] & (torch.rand(len(env_ids), device=env.device) < prob)
        if start_s is None:  # anywhere in the episode
            s = torch.randint(0, int(env.max_episode_length), (len(env_ids),), device=env.device)
        else:
            s = (torch.empty(len(env_ids), device=env.device).uniform_(*start_s) / env.step_dt).long()
        dur = torch.empty(len(env_ids), device=env.device).uniform_(*duration_s) / env.step_dt
        start[env_ids] = s
        end[env_ids] = torch.where(enabled, s + dur.long(), s)
