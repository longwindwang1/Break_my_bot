"""World-model dataset: frames recorded by isaac/collect_wm_data.py, with masks and labels derived here.

Conventions (index t = policy step, per env n):
- o_t / pose_t / episode_step_t were recorded before a_t was applied; terminated_t / truncated_t are the
  flags of the step a_t was applied in. If either is set, o_{t+1} is the first frame of a new episode.
- trans_ok[n, t]: o_t -> o_{t+1} is a real transition (t < T-1 and the episode did not end in step t).
- train_ok[n, t]: trans_ok, and o_t is not the first frame of an episode (stale scan, spawn drop), and no
  push was applied before step t (a velocity jump that (o_t, a_t) cannot explain).
- History: frame t-j for j = H-1..0, replaced by the episode's first frame before the episode start
  (Isaac Lab CircularBuffer semantics, also what Unity's world-model buffer does on reset).
- Fall labels y_h[n, t] = the episode ends with a fall (terminated) in one of the steps t .. t+h-1.
  Known if positive, or if no episode end and no end of the recording falls in those steps; otherwise
  (time-out or recording end within h steps) the label is censored. Noisy-action envs: always censored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .layout import WMLayout, load_layout

POSE_FIELDS = ("root_pos", "root_quat")


def quat_yaw(q: np.ndarray) -> np.ndarray:
    """Yaw of (w, x, y, z) quaternions."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def displacement(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """[N, T, 4] base (dx, dy, dz, dyaw) from t to t+1 in the yaw frame at t; the last step is 0."""
    yaw = quat_yaw(quat)
    dp = np.zeros_like(pos)
    dp[:, :-1] = pos[:, 1:] - pos[:, :-1]
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.zeros(pos.shape[:2] + (4,), np.float32)
    out[..., 0] = c * dp[..., 0] + s * dp[..., 1]
    out[..., 1] = -s * dp[..., 0] + c * dp[..., 1]
    out[..., 2] = dp[..., 2]
    out[:, :-1, 3] = wrap_angle(yaw[:, 1:] - yaw[:, :-1])
    return out


def fall_labels(terminated: np.ndarray, truncated: np.ndarray, horizons: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """terminated/truncated [N, T] -> (y, known), each [N, T, len(horizons)] bool."""
    N, T = terminated.shape
    end_idx = np.empty((N, T), np.int64)  # next step >= t in which the episode ends (T = not within the recording)
    ends_fall = np.empty((N, T), bool)
    nxt, nxt_fall = np.full(N, T), np.zeros(N, bool)
    for t in range(T - 1, -1, -1):
        done = terminated[:, t] | truncated[:, t]
        nxt = np.where(done, t, nxt)
        nxt_fall = np.where(done, terminated[:, t], nxt_fall)
        end_idx[:, t], ends_fall[:, t] = nxt, nxt_fall
    dist = end_idx - np.arange(T)[None]
    y = np.stack([ends_fall & (dist < h) for h in horizons], axis=-1)
    known = np.stack([(dist >= h) for h in horizons], axis=-1) | y
    return y, known


class WMData:
    """One recorded split, held as [N, T, ...] tensors on `device`.

    envs: optional env subset (e.g. the validation envs); all derived arrays follow that order.
    """

    def __init__(self, path: str | Path, device: str = "cuda", envs: np.ndarray | None = None,
                 layout: WMLayout | None = None, extra: bool = False):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.layout = layout or load_layout()
        T = int(self.meta["steps_written"])
        if T < 2:
            raise ValueError(f"{path}: only {T} steps written")
        info = dict(np.load(self.path / "env_info.npz"))
        n_all = len(info["policy_id"])
        self.env_ids = np.arange(n_all) if envs is None else np.asarray(envs)
        sel = self.env_ids

        def load(name: str) -> np.ndarray:  # [T, N, ...] memmap -> [N_sel, T, ...]
            a = np.load(self.path / f"{name}.npy", mmap_mode="r")[:T]
            return np.ascontiguousarray(np.swapaxes(a[:, sel], 0, 1))

        lay = self.layout
        proprio = load("proprio")
        frames = np.ascontiguousarray(proprio[..., lay.frame_idx], dtype=np.float32)
        self.command = proprio[..., 9:12].astype(np.float32) if extra else None
        del proprio
        episode_step = load("episode_step").astype(np.int64)
        terminated, truncated, push = load("terminated"), load("truncated"), load("push")
        done = terminated | truncated
        N = len(sel)
        t_idx = np.arange(T)[None].repeat(N, 0)
        trans_ok = (t_idx < T - 1) & ~done
        train_ok = trans_ok & (episode_step >= 1) & ~push
        y, known = fall_labels(terminated, truncated, lay.fall_horizons)
        noisy = info["noise_std"][sel] > 0
        known[noisy] = False
        disp = displacement(load("root_pos").astype(np.float64), load("root_quat").astype(np.float64))

        self.N, self.T, self.device = N, T, device
        dev = torch.device(device)
        self.frames = torch.from_numpy(frames).to(dev)  # [N, T, F]
        self.scan = torch.from_numpy(load("scan")).to(dev)  # [N, T, 187] float16
        self.action = torch.from_numpy(load("action").astype(np.float32)).to(dev)  # [N, T, 12]
        self.ep_start = torch.from_numpy(t_idx - episode_step).clamp(min=0).to(dev)  # [N, T]
        self.episode_step = torch.from_numpy(episode_step).to(dev)
        self.trans_ok = torch.from_numpy(trans_ok).to(dev)
        self.train_ok = torch.from_numpy(train_ok).to(dev)
        self.fall_y = torch.from_numpy(y).to(dev)
        self.fall_known = torch.from_numpy(known).to(dev)
        self.disp = torch.from_numpy(disp).to(dev)
        self.push = torch.from_numpy(push).to(dev)
        self.policy_id = torch.from_numpy(info["policy_id"][sel].astype(np.int64)).to(dev)
        self.noise_std = info["noise_std"][sel]
        self.terminated = torch.from_numpy(terminated).to(dev)
        self.truncated = torch.from_numpy(truncated).to(dev)
        self.info = {k: v[sel] if v.shape[:1] == (n_all,) else v for k, v in info.items()}
        if extra:  # analysis-only fields
            self.foot_mu = load("foot_mu").astype(np.float32)
            self.tile = load("tile").astype(np.int64)
            self.deceived = load("deceived")
            self.mu_change = load("mu_change")
            self.event = load("event")
        self._perm = torch.from_numpy(lay.hist_perm).to(dev)
        self._state_idx = torch.from_numpy(lay.state_in_frame).to(dev)
        self._hist_offs = torch.arange(lay.history_length - 1, -1, -1, device=dev)  # oldest first

    # --- assembling model inputs ----------------------------------------------------------------------
    def state(self, n: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.frames[n, t][:, self._state_idx]

    def history_frames(self, n: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """[B, H, F] frames t-H+1 .. t (oldest first), filled with the episode's first frame before its start."""
        tt = torch.maximum(t[:, None] - self._hist_offs[None], self.ep_start[n, t][:, None])
        return self.frames[n[:, None], tt]

    def onehot(self, n: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(self.policy_id[n], len(self.layout.policies)).float()

    def assemble(self, hist: torch.Tensor, action: torch.Tensor, scan: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        """wm_in [B, input_dim] from history frames [B, H, F] (oldest first) and the other blocks."""
        B = hist.shape[0]
        return torch.cat([hist.reshape(B, -1)[:, self._perm], action, scan.float(), onehot], dim=1)

    def inputs(self, n: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.assemble(self.history_frames(n, t), self.action[n, t], self.scan[n, t], self.onehot(n))

    def delta(self, n: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """s_{t+1} - s_t (only meaningful where trans_ok)."""
        return self.state(n, (t + 1).clamp(max=self.T - 1)) - self.state(n, t)

    def positions(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, t = mask.nonzero(as_tuple=True)
        return n, t


VAL_STRIDE = 23  # prime, so held-out envs are balanced over policies (i % 3), noise (i % 4) and terrain columns


def val_envs(num_envs: int) -> np.ndarray:
    """Held-out envs: every 23rd (about 4 %)."""
    return np.arange(num_envs)[np.arange(num_envs) % VAL_STRIDE == 0]


def train_envs(num_envs: int) -> np.ndarray:
    return np.arange(num_envs)[np.arange(num_envs) % VAL_STRIDE != 0]
