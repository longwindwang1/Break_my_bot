"""Global terrain height map by ray-casting the terrain mesh (terrain-risk model, W6).

The map covers the whole generated terrain including its border, on a regular grid whose cell (i, j) has its
centre at (x_min + (i + 0.5) * res, y_min + (j + 0.5) * res); axis 0 is x. x_min / y_min are multiples of
0.1 m and res divides 0.1, so cell centres never lie on the 0.05 m grid that step and box edges sit on.
Heights are the z of the first hit of a ray cast straight down from z = 20 m, against the same warp mesh the
height scanner uses (RayCaster.meshes["/World/ground"]). Call before env.close(): the mesh cache is cleared
when the last RayCaster is destroyed.
"""

from __future__ import annotations

import math

import numpy as np
import torch

MESH_PATH = "/World/ground"


def map_extent(gen_cfg) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) of the terrain incl. border, rounded out to 0.1 m."""
    hx = gen_cfg.size[0] * gen_cfg.num_rows / 2 + gen_cfg.border_width
    hy = gen_cfg.size[1] * gen_cfg.num_cols / 2 + gen_cfg.border_width
    r = lambda v: math.ceil(round(v / 0.1, 6)) * 0.1  # noqa: E731
    return -r(hx), r(hx), -r(hy), r(hy)


def raycast_height_map(gen_cfg, res: float = 0.025, chunk_rays: int = 2_000_000, device: str = "cuda:0"):
    """Ray-cast the terrain mesh on the map grid. Returns (heights float16 [nx, ny], meta dict)."""
    from isaaclab.sensors import RayCaster
    from isaaclab.utils.warp import raycast_mesh

    mesh = RayCaster.meshes[MESH_PATH]
    x_min, x_max, y_min, y_max = map_extent(gen_cfg)
    nx, ny = int(round((x_max - x_min) / res)), int(round((y_max - y_min) / res))
    ys = y_min + (torch.arange(ny, device=device, dtype=torch.float64) + 0.5) * res
    out = np.empty((nx, ny), np.float16)
    rows = max(1, chunk_rays // ny)
    misses = 0
    for i0 in range(0, nx, rows):
        i1 = min(nx, i0 + rows)
        xs = x_min + (torch.arange(i0, i1, device=device, dtype=torch.float64) + 0.5) * res
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        starts = torch.stack([gx, gy, torch.full_like(gx, 20.0)], dim=-1).reshape(1, -1, 3).float()
        dirs = torch.zeros_like(starts)
        dirs[..., 2] = -1.0
        hits = raycast_mesh(starts, dirs, mesh, max_dist=100.0)[0][..., 2].reshape(i1 - i0, ny)
        bad = ~torch.isfinite(hits)
        misses += int(bad.sum())
        out[i0:i1] = torch.where(bad, torch.zeros_like(hits), hits).cpu().numpy().astype(np.float16)
    meta = {"res": res, "x_min": x_min, "y_min": y_min, "nx": nx, "ny": ny, "axis0": "x", "dtype": "float16",
            "ray_start_z": 20.0, "mesh": MESH_PATH, "misses": misses}
    return out, meta
