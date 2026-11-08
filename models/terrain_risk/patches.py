"""Height samples for the terrain-risk model: the single Python reference for how the model's input is built.

A global height map (isaac/terrain_map.py: cell (i, j) centred at (x_min + (i + .5) res, y_min + (j + .5) res),
axis 0 = x) is sampled with nearest-neighbour lookups at the points of a yaw-aligned grid:
point (i, j) of a grid with origin (x0, y0) and spacing `res` is at  c + R(yaw) (x0 + res i, y0 + res j)
and lands in height[..., j, i] (x fastest, as configs/robot_interface.yaml terrain_risk.window).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isaac.interface import load_interface


@dataclass
class RiskLayout:
    res: float
    rf: int
    nx: int
    ny: int
    x0: float
    y0: float
    horizons: list[int]
    cond: list[str]

    @property
    def half(self) -> int:
        return self.rf // 2


def load_risk_layout(itf: dict | None = None) -> RiskLayout:
    t = (itf or load_interface())["terrain_risk"]
    w = t["window"]
    return RiskLayout(res=t["resolution"], rf=t["receptive_field"], nx=w["nx"], ny=w["ny"], x0=w["x0"], y0=w["y0"],
                      horizons=list(t["horizons_steps"]), cond=list(t["cond"]))


class HeightMap:
    """Global height map on a device, with nearest-neighbour lookup."""

    def __init__(self, heights: np.ndarray, meta: dict, device: str = "cuda"):
        self.h = torch.as_tensor(np.asarray(heights), device=device)
        self.res, self.x_min, self.y_min = float(meta["res"]), float(meta["x_min"]), float(meta["y_min"])
        self.nx, self.ny = self.h.shape

    @classmethod
    def from_split(cls, split_dir: str | Path, device: str = "cuda") -> HeightMap:
        split_dir = Path(split_dir)
        meta = json.loads((split_dir / "meta.json").read_text())["height_map"]
        return cls(np.load(split_dir / "terrain_height.npy"), meta, device)

    def lookup(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        i = torch.floor((x - self.x_min) / self.res).long().clamp(0, self.nx - 1)
        j = torch.floor((y - self.y_min) / self.res).long().clamp(0, self.ny - 1)
        return self.h[i, j].float()


def grid_xy(cx: torch.Tensor, cy: torch.Tensor, yaw: torch.Tensor, nx: int, ny: int, x0: float, y0: float,
            res: float) -> tuple[torch.Tensor, torch.Tensor]:
    """World x, y [B, ny, nx] of a yaw-aligned grid centred on (cx, cy)."""
    dev = cx.device
    lx = (x0 + res * torch.arange(nx, device=dev, dtype=torch.float32))[None, None, :]
    ly = (y0 + res * torch.arange(ny, device=dev, dtype=torch.float32))[None, :, None]
    c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
    return cx[:, None, None] + c * lx - s * ly, cy[:, None, None] + s * lx + c * ly


def sample_grid(hmap: HeightMap, cx, cy, yaw, nx: int, ny: int, x0: float, y0: float, res: float) -> torch.Tensor:
    """Heights [B, 1, ny, nx]."""
    gx, gy = grid_xy(cx, cy, yaw, nx, ny, x0, y0, res)
    return hmap.lookup(gx, gy)[:, None]


def window(hmap: HeightMap, lay: RiskLayout, cx, cy, yaw) -> torch.Tensor:
    """Egocentric input window [B, 1, ny, nx] around the base (robot_interface.yaml terrain_risk.window)."""
    return sample_grid(hmap, cx, cy, yaw, lay.nx, lay.ny, lay.x0, lay.y0, lay.res)


def patch(hmap: HeightMap, lay: RiskLayout, cx, cy, yaw) -> torch.Tensor:
    """Receptive-field patch [B, 1, rf, rf] whose single output cell is centred on (cx, cy)."""
    o = -lay.res * lay.half
    return sample_grid(hmap, cx, cy, yaw, lay.rf, lay.rf, o, o, lay.res)


def cond_vector(cmd: torch.Tensor, policy_id: torch.Tensor, n_policies: int = 2) -> torch.Tensor:
    """[B, 5]: command (vx, vy, wz) + policy one-hot."""
    return torch.cat([cmd, torch.nn.functional.one_hot(policy_id.long(), n_policies).float()], dim=1)
