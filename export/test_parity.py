"""Interface + export parity tests.

Driven by the Isaac Lab reference rollouts in experiments/w3_reference/go2_<variant>.npz
(recorded by isaac/record_reference.py). Each npz names the checkpoint it was recorded with,
which in turn gives the run's params/env.yaml.

    ~/miniforge3/envs/env_isaaclab/bin/python -m pytest -v
"""

import copy
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from tensordict import TensorDict

from export.export_onnx import export, load_policy
from isaac.interface import REPO_ROOT, check_against_env_yaml, expected_obs_groups, load_interface, policy_obs_dim

VARIANTS = ["flat", "rough", "v2"]
REF_DIR = REPO_ROOT / "experiments" / "w3_reference"


@pytest.fixture(scope="module")
def itf():
    return load_interface()


@pytest.fixture(scope="module", params=VARIANTS)
def variant(request):
    return request.param


@pytest.fixture(scope="module")
def ref(variant):
    path = REF_DIR / f"go2_{variant}.npz"
    if not path.exists():
        pytest.skip(f"missing {path}; record it with isaac/record_reference.py")
    return dict(np.load(path))


@pytest.fixture(scope="module")
def run_dir(ref):
    run = REPO_ROOT / os.path.dirname(str(ref["checkpoint"]))
    if not run.exists():
        pytest.skip(f"missing run dir {run}")
    return run


@pytest.fixture(scope="module")
def checkpoint(ref):
    return os.path.basename(str(ref["checkpoint"]))


@pytest.fixture(scope="module")
def onnx_session(variant, run_dir, checkpoint, tmp_path_factory):
    # Export fresh so the test never passes against a stale model.
    path = export(run_dir, variant, checkpoint, tmp_path_factory.mktemp("onnx") / f"go2_{variant}.onnx")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run_onnx(session, obs: np.ndarray) -> np.ndarray:
    # Exported with a static batch of 1 (what Unity runs), so feed rows one at a time.
    return np.concatenate([session.run(["actions"], {"obs": row[None].astype(np.float32)})[0] for row in obs])


def quat_apply_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vectors v into the frame given by quaternions q (w, x, y, z); batched over rows."""
    w, xyz = q_wxyz[:, :1], q_wxyz[:, 1:]
    t = 2.0 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def build_obs(ref: dict, itf: dict, variant: str) -> np.ndarray:
    """Rebuild the deployed policy's input from raw physics state, following robot_interface.yaml only.

    This is the reference implementation the Unity observation builder must match: per-term values,
    then each observation group (history groups: per term, the last H frames oldest -> newest, with frames
    before the episode start replaced by the first frame), groups concatenated in order.
    """
    q = ref["root_link_quat_w"]
    n = len(q)
    default = np.asarray(itf["default_joint_pos"], dtype=np.float32)
    last_action = np.vstack([np.zeros((1, 12), np.float32), ref["actions"][:-1]])
    parts = {
        "base_lin_vel": quat_apply_inverse(q, ref["root_com_lin_vel_w"]),
        "base_ang_vel": quat_apply_inverse(q, ref["root_com_ang_vel_w"]),
        "projected_gravity": quat_apply_inverse(q, np.tile([0.0, 0.0, -1.0], (n, 1))),
        "velocity_commands": ref["command"],
        "joint_pos": ref["joint_pos"] - default,
        "joint_vel": ref["joint_vel"],
        "actions": last_action,
    }
    if "scan_hits_z" in ref:
        hs = itf["height_scan"]
        raw = ref["scan_sensor_z"][:, None] - ref["scan_hits_z"] - hs["offset"]
        clip = next(t for t in itf["observation"]["terms"] if t["name"] == "height_scan")["clip"]
        parts["height_scan"] = np.clip(raw, *clip)
    blocks = []
    for group in expected_obs_groups(itf, variant):
        h = max(1, group["history_length"])
        frames = np.maximum(np.arange(n)[:, None] - np.arange(h - 1, -1, -1)[None, :], 0)  # [n, h], oldest first
        for t in group["terms"]:
            x = parts[t["name"]]
            assert x.shape == (n, t["dim"]), t["name"]
            blocks.append(x[frames].reshape(n, h * t["dim"]))
    return np.concatenate(blocks, axis=1).astype(np.float32)


# --- interface file vs Isaac Lab ------------------------------------------------------------------


def test_interface_matches_runs(run_dir):
    assert check_against_env_yaml(run_dir / "params") == []


def test_interface_checker_detects_mismatch(run_dir, itf):
    bad = copy.deepcopy(itf)
    bad["action"]["scale"] = 0.5
    bad["joint_names"] = bad["joint_names"][::-1]  # wrong order -> wrong default_joint_pos
    errors = check_against_env_yaml(run_dir / "params", bad)
    assert any(e.startswith("action.scale") for e in errors)
    assert any(e.startswith("default_joint_pos") for e in errors)


def test_reference_matches_interface(ref, itf, variant):
    """Values read from the live articulation during the rollout."""
    assert list(ref["joint_names"]) == itf["joint_names"]
    np.testing.assert_allclose(ref["default_joint_pos"], itf["default_joint_pos"], atol=1e-6)
    a = itf["actuator"]
    np.testing.assert_allclose(ref["stiffness"], a["stiffness"])
    np.testing.assert_allclose(ref["damping"], a["damping"])
    np.testing.assert_allclose(ref["effort_limit"], a["effort_limit"])
    np.testing.assert_allclose(ref["velocity_limit"], a["velocity_limit"])
    np.testing.assert_allclose(ref["saturation_effort"], a["saturation_effort"])
    np.testing.assert_allclose(ref["physics_dt"], itf["timing"]["physics_dt"], rtol=1e-6)
    np.testing.assert_allclose(ref["step_dt"], itf["timing"]["policy_dt"], rtol=1e-6)
    assert ref["obs"].shape[1] == policy_obs_dim(itf, variant)
    assert not ref["dones"].any(), "reference rollout reset mid-way; re-record it"


# --- observation / action semantics ---------------------------------------------------------------


def test_obs_layout(ref, itf, variant):
    """Observation rebuilt from physics state with the yaml formulas equals what Isaac Lab fed the policy."""
    np.testing.assert_allclose(build_obs(ref, itf, variant), ref["obs"], atol=1e-5)


def test_height_scan_ray_positions(ref, itf, variant):
    """Ray k lands at base_xy + R_yaw * grid_offset_k (x fastest), as documented in the yaml."""
    if "scan_hits_xy" not in ref:
        pytest.skip("policy has no height scan")
    hs = itf["height_scan"]
    nx, ny = hs["grid_shape"]
    k = np.arange(nx * ny)
    offsets = np.stack(
        [-hs["grid_size"][0] / 2 + hs["resolution"] * (k % nx), -hs["grid_size"][1] / 2 + hs["resolution"] * (k // nx)],
        axis=1,
    )
    w, x, y, z = ref["root_link_quat_w"].T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    expected = ref["root_link_pos_w"][:, None, :2] + np.stack(
        [c * offsets[:, 0] - s * offsets[:, 1], s * offsets[:, 0] + c * offsets[:, 1]], axis=2
    )
    np.testing.assert_allclose(ref["scan_hits_xy"], expected, atol=1e-4)


def test_action_targets(ref, itf):
    """q_target = default_joint_pos + scale * a."""
    expected = np.asarray(itf["default_joint_pos"]) + itf["action"]["scale"] * ref["actions"]
    np.testing.assert_allclose(ref["joint_pos_target"], expected, atol=1e-5)


# --- PyTorch vs ONNX ------------------------------------------------------------------------------


def test_torch_vs_onnx(variant, run_dir, checkpoint, onnx_session, ref):
    policy, _ = load_policy(run_dir, variant, checkpoint)
    rng = np.random.default_rng(0)
    obs = np.concatenate([ref["obs"], rng.normal(size=(256, ref["obs"].shape[1])).astype(np.float32)])
    with torch.no_grad():
        expected = policy.act_inference(TensorDict({"policy": torch.from_numpy(obs)}, batch_size=[len(obs)])).numpy()
    np.testing.assert_allclose(run_onnx(onnx_session, obs), expected, atol=1e-5)


def test_reference_replay(onnx_session, ref):
    """ONNX on CPU reproduces the actions the policy produced on GPU inside Isaac Lab."""
    np.testing.assert_allclose(run_onnx(onnx_session, ref["obs"]), ref["actions"], atol=1e-4)


def test_onnx_io_contract(onnx_session, itf, variant):
    (inp,), (out,) = onnx_session.get_inputs(), onnx_session.get_outputs()
    assert (inp.name, inp.shape) == ("obs", [1, policy_obs_dim(itf, variant)])
    assert (out.name, out.shape) == ("actions", [1, len(itf["joint_names"])])
    meta = onnx_session.get_modelmeta().custom_metadata_map
    assert meta["variant"] == variant and int(meta["obs_dim"]) == policy_obs_dim(itf, variant)
