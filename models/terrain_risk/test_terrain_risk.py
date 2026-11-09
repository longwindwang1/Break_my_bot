"""Terrain-risk tests: map / grid conventions, patch-window equivalence, offset invariance, and the height map
against Isaac Lab's own height scans (smoke data from isaac/collect_wm_data.py --split geo_train, skipped if
missing).

    ~/miniforge3/envs/env_isaaclab/bin/python -m pytest models -k risk -v
"""

import json
import math

import numpy as np
import pytest
import torch

from isaac.interface import REPO_ROOT, load_interface
from models.terrain_risk.model import RiskNet
from models.terrain_risk.patches import HeightMap, cond_vector, grid_xy, load_risk_layout, patch, sample_grid, window

LAY = load_risk_layout()


def ramp_map(res=0.025) -> HeightMap:
    """10 m x 10 m map centred on 0 whose height is 0.1 * x + 0.01 * y (x, y = cell centres)."""
    n = int(10 / res)
    c = -5 + (np.arange(n) + 0.5) * res
    h = 0.1 * c[:, None] + 0.01 * c[None, :]
    return HeightMap(h.astype(np.float32), {"res": res, "x_min": -5.0, "y_min": -5.0}, device="cpu")


def test_layout_matches_yaml_and_network():
    t = load_interface()["terrain_risk"]
    assert (LAY.nx, LAY.ny) == (t["window"]["nx"], t["window"]["ny"])
    assert RiskNet().rf == LAY.rf == 17
    # the output map covers x in [-0.5, 2.5], y in [-1.0, 1.0]
    assert math.isclose(LAY.x0 + LAY.res * LAY.half, -0.5) and math.isclose(LAY.y0 + LAY.res * LAY.half, -1.0)
    assert math.isclose(LAY.x0 + LAY.res * (LAY.nx - 1 - LAY.half), 2.5)


def test_grid_orientation_x_forward_y_left():
    hm = ramp_map()
    zero = torch.zeros(1)
    # yaw 0: column index i runs along +x (height rises 0.1 per m), row index j along +y (0.01 per m)
    w = sample_grid(hm, zero, zero, zero, 5, 3, 0.0, 0.0, 0.5)[0, 0]
    assert w.shape == (3, 5)
    assert torch.allclose(w[0, 1:] - w[0, :-1], torch.full((4,), 0.05), atol=1e-3)
    assert torch.allclose(w[1:, 0] - w[:-1, 0], torch.full((2,), 0.005), atol=1e-3)
    # yaw +90 deg: "forward" (grid x) is world +y
    gx, gy = grid_xy(zero, zero, torch.tensor([math.pi / 2]), 2, 1, 1.0, 0.0, 1.0)
    assert torch.allclose(gx[0, 0], torch.tensor([0.0, 0.0]), atol=1e-6) and torch.allclose(gy[0, 0], torch.tensor([1.0, 2.0]), atol=1e-6)


def test_nearest_lookup_uses_cell_centres():
    hm = HeightMap(np.arange(16, dtype=np.float32).reshape(4, 4), {"res": 1.0, "x_min": 0.0, "y_min": 0.0}, device="cpu")
    # point (1.9, 2.1) is in cell (1, 2) -> value 1 * 4 + 2
    assert hm.lookup(torch.tensor([1.9]), torch.tensor([2.1])).item() == 6.0


def test_patch_equals_window_cell():
    """The single output of a 17 x 17 patch centred on a window cell equals that cell of the window's output."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    res = 0.025
    n = 400
    h = rng.normal(0, 0.05, (n, n)).cumsum(0).cumsum(1) * 0.01
    hm = HeightMap(h.astype(np.float32), {"res": res, "x_min": -5.0, "y_min": -5.0}, device="cpu")
    net = RiskNet().eval()
    cx, cy, yaw = torch.tensor([0.3]), torch.tensor([-0.2]), torch.tensor([0.7])
    cond = cond_vector(torch.tensor([[0.6, 0.1, -0.2]]), torch.tensor([1]))
    with torch.no_grad():
        full = net(window(hm, LAY, cx, cy, yaw), cond)[0]  # [2, 21, 31]
        for i, j in [(0, 0), (8, 10), (30, 20), (12, 3)]:
            # output cell (i, j) is centred at window point (i + half, j + half)
            lx, ly = LAY.x0 + LAY.res * (i + LAY.half), LAY.y0 + LAY.res * (j + LAY.half)
            px = cx + torch.cos(yaw) * lx - torch.sin(yaw) * ly
            py = cy + torch.sin(yaw) * lx + torch.cos(yaw) * ly
            one = net(patch(hm, LAY, px, py, yaw), cond)[0, :, 0, 0]
            assert torch.allclose(one, full[:, j, i], atol=1e-6), (i, j)


def test_offset_invariance():
    torch.manual_seed(0)
    net = RiskNet().eval()
    h = torch.randn(3, 1, 25, 30) * 0.1
    cond = torch.randn(3, 5)
    with torch.no_grad():
        assert torch.allclose(net(h, cond), net(h + 1.7, cond), atol=1e-6)


GEO_SMOKE = REPO_ROOT / "logs" / "wm_data" / "smoke_geo"


def test_height_map_matches_isaac_scans():
    """Map sampled at the recorded height-scan rays == the scan Isaac Lab computed (incl. yaw, x-fastest order)."""
    if not (GEO_SMOKE / "terrain_height.npy").exists():
        pytest.skip("no smoke geo data; run isaac/collect_wm_data.py --split geo_train --num_envs 256 --steps 160 --out_dir logs/wm_data/smoke_geo")
    hm = HeightMap.from_split(GEO_SMOKE, device="cpu")
    T = json.loads((GEO_SMOKE / "meta.json").read_text())["steps_written"]
    es = np.load(GEO_SMOKE / "episode_step.npy")[:T]
    sel = es >= 1
    pos = torch.tensor(np.load(GEO_SMOKE / "root_pos.npy")[:T][sel])
    q = torch.tensor(np.load(GEO_SMOKE / "root_quat.npy")[:T][sel])
    scan = torch.tensor(np.load(GEO_SMOKE / "scan.npy")[:T][sel].astype(np.float32))
    w, x, y, z = q.T
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    hs = load_interface()["height_scan"]
    nx, ny = hs["grid_shape"]
    h = sample_grid(hm, pos[:, 0], pos[:, 1], yaw, nx, ny, -hs["grid_size"][0] / 2, -hs["grid_size"][1] / 2, hs["resolution"])
    expected = pos[:, 2:3] - hs["offset"] - scan
    ok = scan.abs() < 0.99
    err = (h[:, 0].reshape(len(pos), -1) - expected)[ok].abs()
    assert float(err.median()) < 1e-3
    assert float(torch.quantile(err[:1_000_000], 0.95)) < 0.01
