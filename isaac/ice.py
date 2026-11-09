"""Ice patches for the terrain-risk model, phase 2: visible vs hidden ice.

Ice is flush with the ground and changes only friction (configs/robot_interface.yaml terrain_risk.ice). Isaac Lab
realizes it per foot: before every policy step, a foot whose position (now, or 1 step ahead at its current velocity)
lies over an ice cell gets friction (episode foot friction) * ice.friction, every other foot gets the episode value.
This is the same per-foot material write as the single-foot friction changes v2 was trained with
(isaac/bmb_tasks/mdp/events.py randomize_foot_friction), and it works on every sub-terrain, stairs and slopes
included. The ice map is a uint8 grid on the terrain height map's grid (isaac/terrain_map.py):
0 = no ice, 1 = visible ice, 2 = hidden ice (physically identical; only visible ice enters the model).

Patches (deterministic from the seed): per tile, with probability 1 - p_free, Poisson(mean_patches) rotated
rectangles with sides U(side_range) and centres uniform in the tile, clipped to the tile; nothing within
spawn_clear m of the tile's origin (envs spawn within +-0.5 m of it). Each patch is hidden with prob. hidden_frac.

The generator is numpy only (importable without Isaac Sim); IcePhysics needs a running env.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

NONE, VISIBLE, HIDDEN = 0, 1, 2


@dataclass
class IceSpec:
    p_free: float = 0.2  # fraction of tiles without ice
    mean_patches: float = 6.0  # Poisson mean per icy tile
    side_range: tuple[float, float] = (0.4, 2.5)  # m, each side
    spawn_clear: float = 1.2  # m around the tile origin
    hidden_frac: float = 0.0  # probability that a patch is hidden


def grid_meta(gen_cfg, res: float = 0.025) -> dict:
    """The terrain height map's grid (isaac/terrain_map.py raycast_height_map) without ray casting."""
    from isaac.terrain_map import map_extent

    x_min, x_max, y_min, y_max = map_extent(gen_cfg)
    return {"res": res, "x_min": x_min, "y_min": y_min, "nx": int(round((x_max - x_min) / res)),
            "ny": int(round((y_max - y_min) / res)), "axis0": "x"}


def tile_centres(gen_cfg) -> np.ndarray:
    """[num_rows, num_cols, 2] tile centres (= terrain origins xy for our centred sub-terrains)."""
    sx, sy = gen_cfg.size
    r = (np.arange(gen_cfg.num_rows) + 0.5) * sx - gen_cfg.num_rows * sx / 2
    c = (np.arange(gen_cfg.num_cols) + 0.5) * sy - gen_cfg.num_cols * sy / 2
    return np.stack(np.meshgrid(r, c, indexing="ij"), axis=-1)


def make_ice_map(gen_cfg, grid: dict, seed: int, spec: IceSpec = IceSpec()) -> tuple[np.ndarray, list[dict]]:
    """Ice map uint8 [nx, ny] (0 none / 1 visible / 2 hidden) and the patch list."""
    rng = np.random.default_rng(seed)
    res, x_min, y_min, nx, ny = grid["res"], grid["x_min"], grid["y_min"], grid["nx"], grid["ny"]
    ice = np.zeros((nx, ny), np.uint8)
    sx, sy = gen_cfg.size
    centres = tile_centres(gen_cfg)
    patches = []
    for r in range(gen_cfg.num_rows):
        for c in range(gen_cfg.num_cols):
            if rng.random() < spec.p_free:
                continue
            ox, oy = centres[r, c]
            for _ in range(rng.poisson(spec.mean_patches)):
                px, py = ox + (rng.random() - 0.5) * sx, oy + (rng.random() - 0.5) * sy
                a, b = rng.uniform(*spec.side_range, size=2)
                yaw = rng.uniform(0.0, math.pi)
                hidden = bool(rng.random() < spec.hidden_frac)
                patches.append({"tile": [r, c], "centre": [float(px), float(py)], "sides": [float(a), float(b)],
                                "yaw": float(yaw), "hidden": hidden})
                # cells of the bounding box, clipped to the tile
                ext = 0.5 * math.hypot(a, b)
                lo_x, hi_x = max(px - ext, ox - sx / 2), min(px + ext, ox + sx / 2)
                lo_y, hi_y = max(py - ext, oy - sy / 2), min(py + ext, oy + sy / 2)
                i0, i1 = int(math.floor((lo_x - x_min) / res)), int(math.ceil((hi_x - x_min) / res))
                j0, j1 = int(math.floor((lo_y - y_min) / res)), int(math.ceil((hi_y - y_min) / res))
                if i1 <= i0 or j1 <= j0:
                    continue
                gx = x_min + (np.arange(i0, i1) + 0.5) * res
                gy = y_min + (np.arange(j0, j1) + 0.5) * res
                dx, dy = gx[:, None] - px, gy[None, :] - py
                u = math.cos(yaw) * dx + math.sin(yaw) * dy
                v = -math.sin(yaw) * dx + math.cos(yaw) * dy
                inside = (np.abs(u) <= a / 2) & (np.abs(v) <= b / 2)
                inside &= (np.abs(gx[:, None] - ox) <= sx / 2) & (np.abs(gy[None, :] - oy) <= sy / 2)
                sub = ice[i0:i1, j0:j1]
                code = HIDDEN if hidden else VISIBLE
                # visible wins where patches overlap (it is drawn); hidden only fills cells without ice
                sub[inside & ((sub == NONE) | (code == VISIBLE))] = code
            # spawn clearance
            i0, i1 = int(math.floor((ox - spec.spawn_clear - x_min) / res)), int(math.ceil((ox + spec.spawn_clear - x_min) / res))
            j0, j1 = int(math.floor((oy - spec.spawn_clear - y_min) / res)), int(math.ceil((oy + spec.spawn_clear - y_min) / res))
            gx = x_min + (np.arange(i0, i1) + 0.5) * res
            gy = y_min + (np.arange(j0, j1) + 0.5) * res
            near = (gx[:, None] - ox) ** 2 + (gy[None, :] - oy) ** 2 <= spec.spawn_clear ** 2
            ice[i0:i1, j0:j1][near] = NONE
    return ice, patches


def ice_meta(grid: dict, seed: int, spec: IceSpec, friction: float, patches: list[dict], ice: np.ndarray) -> dict:
    n = ice.size
    return {**grid, "dtype": "uint8", "codes": {"none": NONE, "visible": VISIBLE, "hidden": HIDDEN}, "seed": seed,
            "spec": asdict(spec), "friction": friction, "patches": len(patches),
            "hidden_patches": sum(p["hidden"] for p in patches),
            "cover_visible": float((ice == VISIBLE).sum() / n), "cover_hidden": float((ice == HIDDEN).sum() / n)}


class IcePhysics:
    """Per-foot friction from the ice map, written before every policy step (call update() before env.step)."""

    def __init__(self, env, ice: np.ndarray, grid: dict, friction: float, foot_ids, fric_term_name: str = "foot_friction_reset"):
        from isaac.bmb_tasks.mdp import state

        self.state = state
        self.env = env
        self.robot = env.scene["robot"]
        self.foot_ids = foot_ids
        term_cfg = env.event_manager.get_term_cfg(fric_term_name)
        self.foot_shapes = term_cfg.func.foot_shapes  # [(start, end)] per foot, policy foot order
        self.dynamic_ratio = float(term_cfg.params.get("dynamic_ratio", 0.8))
        self.friction = float(friction)
        self.dev = env.device
        self.map = torch.as_tensor(ice, device=self.dev)
        self.res, self.x_min, self.y_min = float(grid["res"]), float(grid["x_min"]), float(grid["y_min"])
        self.nx, self.ny = self.map.shape
        self.mat = self.robot.root_physx_view.get_material_properties()  # CPU [N, shapes, 3], cached
        N = env.num_envs
        self.base = torch.zeros(N, device=self.dev)
        self.on = torch.zeros(N, len(foot_ids), dtype=torch.bool, device=self.dev)
        self.code = torch.zeros(N, len(foot_ids), dtype=torch.uint8, device=self.dev)
        self.writes = 0

    def lookup(self, xy: torch.Tensor) -> torch.Tensor:
        i = torch.floor((xy[..., 0] - self.x_min) / self.res).long().clamp(0, self.nx - 1)
        j = torch.floor((xy[..., 1] - self.y_min) / self.res).long().clamp(0, self.ny - 1)
        return self.map[i, j]

    def update(self) -> None:
        env, cache = self.env, self.state.foot_friction(self.env)
        reset = env.episode_length_buf == 0  # the reset event just wrote the new episode friction to all feet
        self.base = torch.where(reset, cache[:, 0], self.base)
        self.on &= ~reset[:, None]
        p = self.robot.data.body_pos_w[:, self.foot_ids, :2]
        v = self.robot.data.body_lin_vel_w[:, self.foot_ids, :2]
        now, ahead = self.lookup(p), self.lookup(p + v * env.step_dt)
        self.code = torch.where(now > 0, now, ahead)
        on = self.code > 0
        changed = (on != self.on).any(dim=1)
        self.on = on
        if not bool(changed.any()):
            return
        mu = self.base[:, None] * torch.where(on, self.friction, 1.0)
        ids = changed.nonzero().squeeze(1)
        cache[ids] = mu[ids]
        ids_cpu, mu_cpu = ids.cpu(), mu[ids].cpu()
        for k, (s, e) in enumerate(self.foot_shapes):
            self.mat[ids_cpu, s:e, 0] = mu_cpu[:, k : k + 1]
            self.mat[ids_cpu, s:e, 1] = mu_cpu[:, k : k + 1] * self.dynamic_ratio
            self.mat[ids_cpu, s:e, 2] = 0.0
        self.robot.root_physx_view.set_material_properties(self.mat, ids_cpu)
        self.writes += len(ids)
