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


# --- phase 2: ice ---------------------------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from isaac.ice import HIDDEN, VISIBLE, IceSpec, grid_meta, make_ice_map, tile_centres  # noqa: E402
from models.terrain_risk.patches import ice_channel  # noqa: E402

GEN = SimpleNamespace(size=(8.0, 8.0), num_rows=3, num_cols=4, border_width=2.0)


def test_ice_map_layout():
    g = grid_meta(GEN)
    assert (g["nx"], g["ny"], g["x_min"], g["y_min"]) == (1120, 1440, -14.0, -18.0)
    ice, patches = make_ice_map(GEN, g, 3, IceSpec(hidden_frac=0.5))
    again, _ = make_ice_map(GEN, g, 3, IceSpec(hidden_frac=0.5))
    assert (ice == again).all() and set(np.unique(ice)) == {0, VISIBLE, HIDDEN}
    assert any(p["hidden"] for p in patches) and not all(p["hidden"] for p in patches)
    xs = g["x_min"] + (np.arange(g["nx"]) + 0.5) * g["res"]
    ys = g["y_min"] + (np.arange(g["ny"]) + 0.5) * g["res"]
    on_tiles = (np.abs(xs) <= 12)[:, None] & (np.abs(ys) <= 16)[None, :]
    assert not ice[~on_tiles].any(), "ice on the border"
    for ox, oy in tile_centres(GEN).reshape(-1, 2):
        near = (xs[:, None] - ox) ** 2 + (ys[None, :] - oy) ** 2 <= IceSpec().spawn_clear ** 2
        assert not ice[near].any(), "ice at a spawn point"
    # hiding patches changes only what is drawn, not where the ice is
    shown, _ = make_ice_map(GEN, g, 3, IceSpec(hidden_frac=0.0))
    assert not (shown == HIDDEN).any() and ((shown > 0) == (ice > 0)).all()
    # a patch centre (away from spawn points) is ice
    p = next(p for p in patches if min(math.dist(p["centre"], c) for c in tile_centres(GEN).reshape(-1, 2)) > 1.5)
    i, j = int((p["centre"][0] - g["x_min"]) / g["res"]), int((p["centre"][1] - g["y_min"]) / g["res"])
    assert ice[i, j] > 0


def test_ice_channel_modes():
    codes = torch.tensor([0.0, 1.0, 2.0])
    assert ice_channel(codes, "map").tolist() == [0, 1, 0]
    assert ice_channel(codes, "all").tolist() == [0, 1, 1]
    assert ice_channel(codes, "none").tolist() == [0, 0, 0]


def test_patch_equals_window_cell_with_ice():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    res, n = 0.025, 400
    meta = {"res": res, "x_min": -5.0, "y_min": -5.0}
    hm = HeightMap((rng.normal(0, 0.05, (n, n)).cumsum(0).cumsum(1) * 0.01).astype(np.float32), meta, device="cpu")
    codes = np.kron(rng.integers(0, 3, (n // 20, n // 20)), np.ones((20, 20))).astype(np.uint8)  # 0.5 m blocks
    im = HeightMap(codes, meta, device="cpu")
    net = RiskNet(n_ice=1).eval()
    cx, cy, yaw = torch.tensor([0.3]), torch.tensor([-0.2]), torch.tensor([0.7])
    cond = cond_vector(torch.tensor([[0.6, 0.1, -0.2]]), torch.tensor([1]))
    with torch.no_grad():
        wi = ice_channel(window(im, LAY, cx, cy, yaw))
        assert 0 < float(wi.mean()) < 1
        full = net(window(hm, LAY, cx, cy, yaw), cond, wi)[0]
        assert not torch.allclose(full, net(window(hm, LAY, cx, cy, yaw), cond)[0]), "ice channel has no effect"
        for i, j in [(0, 0), (8, 10), (30, 20), (12, 3)]:
            lx, ly = LAY.x0 + LAY.res * (i + LAY.half), LAY.y0 + LAY.res * (j + LAY.half)
            px = cx + torch.cos(yaw) * lx - torch.sin(yaw) * ly
            py = cy + torch.sin(yaw) * lx + torch.cos(yaw) * ly
            one = net(patch(hm, LAY, px, py, yaw), cond, ice_channel(patch(im, LAY, px, py, yaw)))[0, :, 0, 0]
            assert torch.allclose(one, full[:, j, i], atol=1e-6), (i, j)
        h, ice = torch.randn(3, 1, 25, 30) * 0.1, (torch.rand(3, 1, 25, 30) < 0.3).float()
        c = torch.randn(3, 5)
        assert torch.allclose(net(h, c, ice), net(h + 1.7, c, ice), atol=1e-6)


ICE_SMOKE = REPO_ROOT / "logs" / "wm_data" / "smoke_ice"


def test_ice_friction_follows_the_map():
    """Recorded foot friction = episode friction * ice.friction exactly where the collector put a foot on ice, and
    every foot standing over an ice cell was put on ice."""
    if not (ICE_SMOKE / "ice_mask.npy").exists():
        pytest.skip("no smoke ice data; run isaac/collect_wm_data.py --split ice_train --steps 150 --out_dir logs/wm_data/smoke_ice")
    meta = json.loads((ICE_SMOKE / "meta.json").read_text())
    T = meta["steps_written"]
    mu = np.load(ICE_SMOKE / "foot_mu.npy")[:T].astype(np.float32)
    code = np.load(ICE_SMOKE / "foot_ice.npy")[:T]
    fp = np.load(ICE_SMOKE / "feet_pos.npy")[:T]
    on = code > 0
    base = np.where(on, -1.0, mu).max(axis=2, keepdims=True)  # a non-ice foot's friction in the same frame
    m = on & (base > 0)
    assert m.sum() > 1000
    ratio = mu[m] / np.broadcast_to(base, mu.shape)[m]
    assert np.abs(ratio - load_interface()["terrain_risk"]["ice"]["friction"]).max() < 2e-3  # float16 storage
    im = HeightMap.ice_from_split(ICE_SMOKE, device="cpu")
    now = im.lookup(torch.from_numpy(fp[..., 0]), torch.from_numpy(fp[..., 1])).numpy()
    assert (code[now > 0] == now[now > 0]).all()
