"""World-model dataset tests: masks, history fill, labels (synthetic data), and the history layout against
Isaac Lab's own proprio_history buffer (smoke data from isaac/collect_wm_data.py, skipped if missing).

    ~/miniforge3/envs/env_isaaclab/bin/python -m pytest models -v
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from isaac.interface import REPO_ROOT
from models.world_model.dataset import WMData, displacement, fall_labels, val_envs
from models.world_model.layout import history_perm, load_layout

T, N = 12, 3


def write_split(path: Path, terminated, truncated, push=None, episode_step=None, noise=None, pos=None, quat=None):
    """Synthetic split in collect_wm_data.py's format; proprio[t, n, :] = 100 * n + t (frame identity)."""
    path.mkdir(parents=True, exist_ok=True)
    proprio = (100.0 * np.arange(N)[None, :, None] + np.arange(T)[:, None, None]) * np.ones((1, 1, 48))
    fields = {
        "proprio": proprio.astype(np.float32),
        "scan": np.zeros((T, N, 187), np.float16),
        "action": np.zeros((T, N, 12), np.float32),
        "root_pos": np.zeros((T, N, 3), np.float32) if pos is None else pos,
        "root_quat": np.tile(np.array([1, 0, 0, 0], np.float32), (T, N, 1)) if quat is None else quat,
        "episode_step": episode_step if episode_step is not None else np.arange(T)[:, None].repeat(N, 1).astype(np.int16),
        "terminated": terminated,
        "truncated": truncated,
        "push": np.zeros((T, N), bool) if push is None else push,
    }
    for k, v in fields.items():
        np.save(path / f"{k}.npy", v)
    np.savez(path / "env_info.npz", policy_id=np.array([1, 1, 0], np.int8),
             noise_std=np.zeros(N, np.float32) if noise is None else noise, event_type=np.zeros(N, np.int8))
    (path / "meta.json").write_text(json.dumps({"steps_written": T}))


def episode_steps(done: np.ndarray) -> np.ndarray:
    """episode_length_buf for frames recorded with the given per-step done flags [T, N]."""
    es = np.zeros_like(done, dtype=np.int16)
    for t in range(1, done.shape[0]):
        es[t] = np.where(done[t - 1], 0, es[t - 1] + 1)
    return es


def test_history_perm_term_major():
    # 2 frames of terms with dims [2, 1]: frame-major [a0 a1 b | a0' a1' b'] -> [a0 a1 a0' a1' | b b']
    assert history_perm([2, 1], 2).tolist() == [0, 1, 3, 4, 2, 5]


def test_masks_history_fill_and_labels(tmp_path):
    term = np.zeros((T, N), bool)
    trunc = np.zeros((T, N), bool)
    term[4, 0] = True  # env 0 falls in step 4 -> frame 5 starts a new episode
    trunc[8, 1] = True  # env 1 times out in step 8
    push = np.zeros((T, N), bool)
    push[6, 2] = True
    write_split(tmp_path, term, trunc, push=push, episode_step=episode_steps(term | trunc))
    lay = load_layout()
    d = WMData(tmp_path, device="cpu", layout=lay)

    # masks
    ok = d.trans_ok.numpy()
    assert not ok[0, 4] and ok[0, 3] and not ok[1, 8] and not ok[:, T - 1].any()
    tr = d.train_ok.numpy()
    assert not tr[:, 0].any()  # first frame of every episode
    assert not tr[0, 5]  # first frame after env 0's reset
    assert not tr[2, 6] and tr[2, 5]  # push

    # history: frames before the episode start are the episode's first frame
    n, t = torch.tensor([0, 0]), torch.tensor([6, 2])
    h = d.history_frames(n, t)[..., 0].numpy()  # frame value = 100 n + t
    assert h[0].tolist() == [5, 5, 5, 5, 5, 6]
    assert h[1].tolist() == [0, 0, 0, 0, 1, 2]
    x = d.inputs(n, t)
    assert x.shape == (2, lay.input_dim)
    assert torch.allclose(x[:, lay.current_state_in_input], d.state(n, t))

    # fall labels (horizons 10, 25, 50; T = 12)
    y, known = d.fall_y.numpy(), d.fall_known.numpy()
    assert y[0, 0, 0] and y[0, 4, 0] and not y[0, 5, 0]  # fall in step 4 is within 10 steps of frames 0..4
    assert known[0, :5].all()
    assert not known[1, 0, 0]  # env 1 times out within 10 steps: censored
    assert not known[2, 3, 0]  # env 2 never ends: recording ends within 10 steps of frame 3
    assert known[2, 2, 0] and not y[2, 2, 0]  # frame 2: steps 2..11 all recorded, no fall


def test_fall_labels_fall_and_timeout_same_step():
    term = np.array([[False, True]])
    trunc = np.array([[False, True]])
    y, known = fall_labels(term, trunc, [1])
    assert y[0, 1, 0] and known[0, 1, 0]  # a fall coinciding with the time-out counts as a fall
    assert not y[0, 0, 0] and known[0, 0, 0]  # step 0 itself was recorded and did not end the episode


def test_noisy_envs_have_no_fall_labels(tmp_path):
    term = np.zeros((T, N), bool)
    term[3, 0] = True
    write_split(tmp_path, term, np.zeros((T, N), bool), episode_step=episode_steps(term),
                noise=np.array([0.1, 0, 0], np.float32))
    d = WMData(tmp_path, device="cpu")
    assert not d.fall_known[0].any()


def test_displacement_in_yaw_frame():
    # robot yawed +90 deg moves +1 m along world y = forward; then turns to 180 deg
    pos = np.array([[[0, 0, 0.3], [0, 1, 0.3], [0, 1, 0.2]]], float)
    yaw = np.array([[np.pi / 2, np.pi, np.pi]])
    quat = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], axis=-1)
    d = displacement(pos, quat)
    np.testing.assert_allclose(d[0, 0], [1, 0, 0, np.pi / 2], atol=1e-9)
    np.testing.assert_allclose(d[0, 1], [0, 0, -0.1, 0], atol=1e-9)


def test_val_envs_balanced():
    v = val_envs(4096)
    assert set(v % 3) == {0, 1, 2} and set(v % 4) == {0, 1, 2, 3}


SMOKE = sorted(p for p in (REPO_ROOT / "logs" / "wm_data").glob("smoke_*") if p.is_dir()) if (REPO_ROOT / "logs" / "wm_data").exists() else []


@pytest.mark.parametrize("split_dir", SMOKE or [None], ids=lambda p: p.name if p else "none")
def test_history_matches_isaac_buffer(split_dir):
    """Offline reconstruction == Isaac Lab's own proprio_history group (same terms, H = 25), incl. resets."""
    if split_dir is None or not (split_dir / "check_proprio_history.npy").exists():
        pytest.skip("no smoke data; run isaac/collect_wm_data.py --num_envs 256 --steps 160 --out_dir logs/wm_data/smoke_<split>")
    meta = json.loads((split_dir / "meta.json").read_text())
    C = meta["check_envs"]
    ref = np.load(split_dir / "check_proprio_history.npy")[: meta["steps_written"]]  # [T, C, 1125]
    d = WMData(split_dir, device="cpu", envs=np.arange(C))
    lay = d.layout
    H = 25
    perm = torch.from_numpy(history_perm(lay.frame_term_dims, H))
    offs = torch.arange(H - 1, -1, -1)
    for c in range(C):
        t = torch.arange(d.T)
        n = torch.full_like(t, c)
        tt = torch.maximum(t[:, None] - offs[None], d.ep_start[n, t][:, None])
        hist = d.frames[n[:, None], tt].reshape(d.T, -1)[:, perm]
        np.testing.assert_array_equal(hist.numpy(), ref[:, c])
    # the recorded episode_step agrees with the done flags
    done = (d.terminated | d.truncated).numpy()
    es = d.episode_step.numpy()
    assert (es[:, 1:][done[:, :-1]] == 0).all() and (es[:, 1:][~done[:, :-1]] == es[:, :-1][~done[:, :-1]] + 1).all()
