"""Terrain-risk interface + export parity tests (W6).

- ONNX IO contract (dynamic H, W; "ice" input for ice models, phase 2) and metadata;
- PyTorch == ONNX for the egocentric window and a larger (whole-level style) input, with ice; the tracked ONNX is
  the export of its run;
- the Unity reference windows (experiments/w3_reference/go2_terrain_risk.json) are reproduced: risk maps (with and
  without the ice input), corridor masks, risk ahead (total and ice excess) and the slowdown decision from
  robot_interface.yaml terrain_risk.slowdown (per-policy trigger kind and threshold).

    ~/miniforge3/envs/env_isaaclab/bin/python -m pytest -k risk -v
"""

import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

from export.export_risk_onnx import OUT, export
from isaac.interface import REPO_ROOT, load_interface
from models.terrain_risk.evaluate import corridor_mask
from models.terrain_risk.model import load_risk_model
from models.terrain_risk.patches import load_risk_layout

REF = REPO_ROOT / "experiments" / "w3_reference" / "go2_terrain_risk.json"
LAY = load_risk_layout()


@pytest.fixture(scope="module")
def session():
    if not OUT.exists():
        pytest.skip(f"missing {OUT}; run export/export_risk_onnx.py")
    return ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def run_dir(session):
    run = REPO_ROOT / session.get_modelmeta().custom_metadata_map["run"]
    if not (run / "model.pt").exists():
        pytest.skip(f"missing training run {run}")
    return run


def has_ice(session) -> bool:
    return session.get_modelmeta().custom_metadata_map.get("ice_input") == "1"


def feed(session, h, cond, ice=None) -> dict:
    f = {"height": h, "cond": cond}
    if has_ice(session):
        f["ice"] = np.zeros_like(h) if ice is None else ice
    return f


def test_risk_io_contract(session):
    ins = {i.name: i.shape for i in session.get_inputs()}
    expected = {"height": [1, 1, "H", "W"], "cond": [1, len(LAY.cond)]}
    if has_ice(session):
        expected["ice"] = [1, 1, "H", "W"]
    assert ins == expected
    (out,) = session.get_outputs()
    assert out.name == "risk" and out.shape[:2] == [1, len(LAY.horizons)]
    meta = session.get_modelmeta().custom_metadata_map
    assert meta["kind"] == "terrain_risk" and int(meta["receptive_field"]) == LAY.rf
    h = np.zeros((1, 1, LAY.ny, LAY.nx), np.float32)
    r = session.run(["risk"], feed(session, h, np.zeros((1, 5), np.float32)))[0]
    assert r.shape == (1, 2, LAY.ny - LAY.rf + 1, LAY.nx - LAY.rf + 1)


@pytest.mark.parametrize("hw", [(37, 47), (120, 200)])
def test_risk_torch_vs_onnx(run_dir, session, hw):
    model, _ = load_risk_model(run_dir / "model.pt")
    rng = np.random.default_rng(0)
    h = (rng.normal(0, 0.08, (1, 1, *hw)).cumsum(2) + 3.0).astype(np.float32)
    ice = (rng.random((1, 1, *hw)) < 0.3).astype(np.float32)
    cond = np.array([[0.7, -0.1, 0.3, 0.0, 1.0]], np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(h), torch.from_numpy(cond), torch.from_numpy(ice) if model.has_ice else None).numpy()
    # heights around 3 m: the zero-sum first layer cancels large terms, so float32 rounding differs by ~1e-5
    np.testing.assert_allclose(session.run(["risk"], feed(session, h, cond, ice))[0], expected, atol=5e-5)


def test_risk_tracked_onnx_is_this_run(run_dir, session, tmp_path):
    fresh = ort.InferenceSession(str(export(run_dir, tmp_path / "risk.onnx")), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    h = rng.normal(0, 0.1, (1, 1, LAY.ny, LAY.nx)).astype(np.float32)
    ice = (rng.random(h.shape) < 0.3).astype(np.float32)
    cond = np.array([[0.5, 0.0, 0.0, 1.0, 0.0]], np.float32)
    np.testing.assert_allclose(session.run(["risk"], feed(session, h, cond, ice))[0],
                               fresh.run(["risk"], feed(fresh, h, cond, ice))[0], atol=1e-7)


def test_risk_reference_windows(session):
    if not REF.exists():
        pytest.skip(f"missing {REF}")
    doc = json.loads(REF.read_text())
    slow = load_interface()["terrain_risk"]["slowdown"]
    assert doc["layout"] == {"nx": LAY.nx, "ny": LAY.ny, "x0": LAY.x0, "y0": LAY.y0, "res": LAY.res}
    for w in doc["windows"]:
        h = np.asarray(w["height"], np.float32)[None, None]
        cond = np.asarray(w["cond"], np.float32)[None]
        ice = np.asarray(w["ice"], np.float32)[None, None] if "ice" in w else None
        assert has_ice(session) == (ice is not None), "reference windows and tracked ONNX disagree on the ice input"
        risk = session.run(["risk"], feed(session, h, cond, ice))[0][0]
        np.testing.assert_allclose(risk, np.asarray(w["risk"]), atol=1e-6)
        no_ice = session.run(["risk"], feed(session, h, cond))[0][0]
        np.testing.assert_allclose(no_ice, np.asarray(w["risk_no_ice"]), atol=1e-6)
        m = corridor_mask(torch.from_numpy(cond[:, :3]), LAY)[0].numpy()
        assert (m == np.asarray(w["corridor_mask"], bool)).all()
        ra = float(np.where(m, risk[0], 0.0).max())
        ra_ex = float(np.where(m, risk[0] - no_ice[0], 0.0).max())
        assert abs(ra - w["risk_ahead"]) < 1e-6 and abs(ra_ex - w["risk_ahead_ice_excess"]) < 1e-6
        kind, thr = slow["trigger"][w["policy"]], slow["threshold"][w["policy"]]
        assert w["trigger"] == kind and w["threshold"] == thr
        assert w["slow_down"] == bool(thr is not None and (ra if kind == "total" else ra_ex) > thr)
    if has_ice(session):
        assert any(np.asarray(w["ice"]).any() for w in doc["windows"]), "no reference window shows ice"
