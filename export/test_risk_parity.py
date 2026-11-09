"""Terrain-risk interface + export parity tests (W6).

- ONNX IO contract (dynamic H, W) and metadata;
- PyTorch == ONNX for the egocentric window and a larger (whole-level style) input; the tracked ONNX is the
  export of its run;
- the Unity reference windows (experiments/w3_reference/go2_terrain_risk.json) are reproduced: risk maps,
  corridor masks, risk ahead and the slowdown decision from robot_interface.yaml terrain_risk.slowdown.

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


def test_risk_io_contract(session):
    ins = {i.name: i.shape for i in session.get_inputs()}
    assert ins == {"height": [1, 1, "H", "W"], "cond": [1, len(LAY.cond)]}
    (out,) = session.get_outputs()
    assert out.name == "risk" and out.shape[:2] == [1, len(LAY.horizons)]
    meta = session.get_modelmeta().custom_metadata_map
    assert meta["kind"] == "terrain_risk" and int(meta["receptive_field"]) == LAY.rf
    h = np.zeros((1, 1, LAY.ny, LAY.nx), np.float32)
    r = session.run(["risk"], {"height": h, "cond": np.zeros((1, 5), np.float32)})[0]
    assert r.shape == (1, 2, LAY.ny - LAY.rf + 1, LAY.nx - LAY.rf + 1)


@pytest.mark.parametrize("hw", [(37, 47), (120, 200)])
def test_risk_torch_vs_onnx(run_dir, session, hw):
    model, _ = load_risk_model(run_dir / "model.pt")
    rng = np.random.default_rng(0)
    h = (rng.normal(0, 0.08, (1, 1, *hw)).cumsum(2) + 3.0).astype(np.float32)
    cond = np.array([[0.7, -0.1, 0.3, 0.0, 1.0]], np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(h), torch.from_numpy(cond)).numpy()
    # heights around 3 m: the zero-sum first layer cancels large terms, so float32 rounding differs by ~1e-5
    np.testing.assert_allclose(session.run(["risk"], {"height": h, "cond": cond})[0], expected, atol=5e-5)


def test_risk_tracked_onnx_is_this_run(run_dir, session, tmp_path):
    fresh = ort.InferenceSession(str(export(run_dir, tmp_path / "risk.onnx")), providers=["CPUExecutionProvider"])
    h = np.random.default_rng(1).normal(0, 0.1, (1, 1, LAY.ny, LAY.nx)).astype(np.float32)
    cond = np.array([[0.5, 0.0, 0.0, 1.0, 0.0]], np.float32)
    np.testing.assert_allclose(session.run(["risk"], {"height": h, "cond": cond})[0],
                               fresh.run(["risk"], {"height": h, "cond": cond})[0], atol=1e-7)


def test_risk_reference_windows(session):
    if not REF.exists():
        pytest.skip(f"missing {REF}")
    doc = json.loads(REF.read_text())
    slow = load_interface()["terrain_risk"]["slowdown"]
    assert doc["layout"] == {"nx": LAY.nx, "ny": LAY.ny, "x0": LAY.x0, "y0": LAY.y0, "res": LAY.res}
    for w in doc["windows"]:
        h = np.asarray(w["height"], np.float32)[None, None]
        cond = np.asarray(w["cond"], np.float32)[None]
        risk = session.run(["risk"], {"height": h, "cond": cond})[0][0]
        np.testing.assert_allclose(risk, np.asarray(w["risk"]), atol=1e-6)
        m = corridor_mask(torch.from_numpy(cond[:, :3]), LAY)[0].numpy()
        assert (m == np.asarray(w["corridor_mask"], bool)).all()
        ra = float(np.where(m, risk[0], 0.0).max())
        assert abs(ra - w["risk_ahead"]) < 1e-6
        thr = slow["threshold"][w["policy"]]
        assert w["slow_down"] == bool(thr is not None and ra > thr)
