"""World-model interface + export parity tests (W5).

- the input layout in robot_interface.yaml, rebuilt from raw physics state, equals what Isaac Lab's own
  history buffer recorded in the v2 reference rollout (fill semantics included);
- PyTorch ensemble == ONNX (export/models/go2_wm.onnx), and the tracked ONNX is the export of its run;
- the Unity reference traces (experiments/w3_reference/go2_wm*.json) are reproduced from the yaml.

    ~/miniforge3/envs/env_isaaclab/bin/python -m pytest -k wm -v
"""

import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

from export.export_wm_onnx import OUTPUTS, build_wm_input, export, surprisal_trace
from isaac.interface import REPO_ROOT, expected_obs_groups, load_interface
from models.world_model.ensemble import load_model
from models.world_model.layout import load_layout

ONNX_PATH = REPO_ROOT / "export" / "models" / "go2_wm.onnx"
REF_DIR = REPO_ROOT / "experiments" / "w3_reference"
TRACES = ["go2_wm.json", "go2_wm_mudrop.json"]


@pytest.fixture(scope="module")
def itf():
    return load_interface()


@pytest.fixture(scope="module")
def ref_v2():
    path = REF_DIR / "go2_v2.npz"
    if not path.exists():
        pytest.skip(f"missing {path}")
    return dict(np.load(path))


@pytest.fixture(scope="module")
def session():
    if not ONNX_PATH.exists():
        pytest.skip(f"missing {ONNX_PATH}; run export/export_wm_onnx.py")
    return ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def run_dir(session):
    run = REPO_ROOT / session.get_modelmeta().custom_metadata_map["run"]
    if not (run / ckpt_name(session)).exists():
        pytest.skip(f"missing training run {run}")
    return run


def ckpt_name(session) -> str:
    return session.get_modelmeta().custom_metadata_map.get("checkpoint", "model.pt")


def run_onnx(session, wm_in: np.ndarray) -> dict[str, np.ndarray]:
    rows = [session.run(OUTPUTS, {"wm_in": x[None].astype(np.float32)}) for x in wm_in]
    return {k: np.concatenate([r[i] for r in rows]) for i, k in enumerate(OUTPUTS)}


def unpack(a: dict) -> np.ndarray:
    return np.asarray(a["data"]).reshape(a["shape"])


def test_wm_layout_dims(itf):
    lay = load_layout(itf)
    assert lay.input_dim == itf["world_model"]["input_dim"] == 471
    assert lay.state_dim == itf["world_model"]["outputs"]["state_mean"]["dim"] == 33


def test_wm_input_matches_isaac_history(ref_v2, itf):
    """wm_in rebuilt from raw state == the same blocks cut from the obs Isaac Lab fed the v2 policy
    (whose proprio_history buffer has the same terms and reset semantics, 25 instead of 6 frames)."""
    built = build_wm_input(ref_v2, itf, "v2")
    obs = ref_v2["obs"]
    n = len(obs)
    policy_group, hist_group = expected_obs_groups(itf, "v2")
    off = sum(t["dim"] for t in policy_group["terms"])
    H_isaac = hist_group["history_length"]
    isaac = {}
    for t in hist_group["terms"]:
        isaac[t["name"]] = obs[:, off : off + H_isaac * t["dim"]].reshape(n, H_isaac, t["dim"])
        off += H_isaac * t["dim"]
    hist = itf["world_model"]["input"][0]
    H = hist["history_length"]
    blocks = [isaac[name][:, -H:].reshape(n, -1) for name in hist["terms"]]
    onehot = np.tile([0.0, 1.0], (n, 1))
    expected = np.concatenate(blocks + [ref_v2["actions"], obs[:, 48:235], onehot], axis=1)
    np.testing.assert_allclose(built, expected, atol=1e-5)


def test_wm_io_contract(session, itf):
    lay = load_layout(itf)
    (inp,) = session.get_inputs()
    assert (inp.name, inp.shape) == ("wm_in", [1, lay.input_dim])
    outs = [(o.name, o.shape) for o in session.get_outputs()]
    spec = itf["world_model"]["outputs"]
    assert [o[0] for o in outs] == OUTPUTS
    for name, shape in outs:
        assert shape == [1, spec[name]["dim"]], name
    meta = session.get_modelmeta().custom_metadata_map
    assert meta["kind"] == "world_model" and int(meta["input_dim"]) == lay.input_dim


def test_wm_torch_vs_onnx(run_dir, session, ref_v2, itf):
    model, _ = load_model(run_dir / ckpt_name(session), "cpu")
    lay = load_layout(itf)
    rng = np.random.default_rng(0)
    x_ref = build_wm_input(ref_v2, itf, "v2")
    x_rand = np.zeros((128, lay.input_dim), np.float32)
    x_rand[:, : lay.dyn_dim] = (model.in_mean.numpy() + model.in_std.numpy() * rng.normal(size=(128, lay.dyn_dim))).astype(np.float32)
    x_rand[np.arange(128), lay.dyn_dim + rng.integers(0, 2, 128)] = 1.0
    x = np.concatenate([x_ref, x_rand])
    with torch.no_grad():
        p = model.predictive(torch.from_numpy(x))
    S = lay.state_dim
    s_t = x[:, lay.current_state_in_input]
    expected = {
        "state_mean": s_t + p["delta_mean"][:, :S].numpy(),
        "state_var": p["delta_var"][:, :S].numpy(),
        "disp_mean": p["delta_mean"][:, S:].numpy(),
        "disp_var": p["delta_var"][:, S:].numpy(),
        "fall_prob": p["fall_prob"].numpy(),
        "disagreement": (p["epistemic"][:, :S] / model.out_std[:S] ** 2).mean(1, keepdim=True).numpy(),
    }
    got = run_onnx(session, x)
    for k in ("state_mean", "disp_mean", "fall_prob"):
        np.testing.assert_allclose(got[k], expected[k], rtol=1e-4, atol=1e-5, err_msg=k)
    for k in ("state_var", "disp_var", "disagreement"):
        np.testing.assert_allclose(got[k], expected[k], rtol=1e-3, atol=1e-9, err_msg=k)


def test_wm_tracked_onnx_is_this_run(run_dir, session, ref_v2, itf, tmp_path):
    fresh = ort.InferenceSession(str(export(run_dir, tmp_path / "wm.onnx", checkpoint=ckpt_name(session))), providers=["CPUExecutionProvider"])
    x = build_wm_input(ref_v2, itf, "v2")[::10]
    a, b = run_onnx(session, x), run_onnx(fresh, x)
    for k in OUTPUTS:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-6, atol=1e-7, err_msg=k)


@pytest.mark.parametrize("name", TRACES)
def test_wm_reference_trace(name, session, itf):
    """The Unity reference trace is what the yaml + ONNX produce now (inputs, outputs, surprisal, CUSUM)."""
    path = REF_DIR / name
    if not path.exists():
        pytest.skip(f"missing {path}; run export/export_wm_onnx.py --reference ...")
    doc = json.loads(path.read_text())
    ref = dict(np.load(REPO_ROOT / doc["source"]))
    wm_in = unpack(doc["wm_in"]).astype(np.float32)
    np.testing.assert_allclose(build_wm_input(ref, itf, doc["policy"]), wm_in, atol=1e-6)
    got = run_onnx(session, wm_in)
    for k in OUTPUTS:
        np.testing.assert_allclose(got[k], unpack(doc[k]), rtol=1e-5, atol=1e-6, err_msg=k)
    lay = load_layout(itf)
    tr = surprisal_trace(wm_in[:, lay.current_state_in_input], {k: unpack(doc[k]) for k in OUTPUTS}, itf, doc["policy"],
                         unpack(doc["episode_step"]))
    for k in ("z", "S", "x", "g", "spike"):
        np.testing.assert_allclose(tr[k], unpack(doc[k]), rtol=1e-6, atol=1e-6, err_msg=k)
    assert (tr["alarm"] == unpack(doc["alarm"]).astype(bool)).all()
    if doc["friction_drop"] is None:
        assert not tr["alarm"].any(), "the clean reference rollout should not raise an alarm"
