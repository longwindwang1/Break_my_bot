"""Terrain-risk samples: one per recorded frame, i.e. (base x, y, yaw, command, policy) -> fall within h steps.

Labels are the world model's censored fall labels (models/world_model/dataset.fall_labels). A frame is used
if it is not the first frame of an episode, its env had no action noise, its height scan was truthful, and its
label for that horizon is known. Train / validation are split by terrain tile (envs respawn on their own tile,
so an env split would leak terrain); frames on the border (tile -1) go to train.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from models.world_model.dataset import fall_labels, quat_yaw

from .patches import HeightMap, ice_channel, load_risk_layout, patch


class RiskFrames:
    def __init__(self, split_dir: str | Path, device: str = "cuda", val_frac: float = 0.1, seed: int = 0):
        d = Path(split_dir)
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text())
        T = int(self.meta["steps_written"])
        info = dict(np.load(d / "env_info.npz"))
        lay = load_risk_layout()

        def load(name):  # [T, N, ...] -> [N, T, ...]
            return np.ascontiguousarray(np.swapaxes(np.load(d / f"{name}.npy", mmap_mode="r")[:T], 0, 1))

        pos = load("root_pos")
        yaw = quat_yaw(load("root_quat").astype(np.float64)).astype(np.float32)
        cmd = load("proprio")[..., 9:12].astype(np.float32)
        es = load("episode_step").astype(np.int64)
        term, trunc = load("terminated"), load("truncated")
        deceived = load("deceived")
        tile = load("tile").astype(np.int64)
        y, known = fall_labels(term, trunc, lay.horizons)
        N = pos.shape[0]
        usable = (es >= 1) & ~deceived & (info["noise_std"][:, None] == 0)
        known &= usable[..., None]
        # tile split
        ter = self.meta["terrain"]
        n_tiles = ter["num_rows"] * ter["num_cols"]
        rng = np.random.default_rng(seed)
        val_tiles = rng.choice(n_tiles, size=max(1, int(round(val_frac * n_tiles))), replace=False)
        is_val = np.isin(tile, val_tiles)
        keep = known.any(-1)
        n_idx, t_idx = np.nonzero(keep)
        dev = torch.device(device)
        self.x = torch.from_numpy(pos[n_idx, t_idx, 0]).to(dev)
        self.y = torch.from_numpy(pos[n_idx, t_idx, 1]).to(dev)
        self.yaw = torch.from_numpy(yaw[n_idx, t_idx]).to(dev)
        self.cmd = torch.from_numpy(cmd[n_idx, t_idx]).to(dev)
        self.policy = torch.from_numpy(info["policy_id"][n_idx].astype(np.int64)).to(dev)
        self.label = torch.from_numpy(y[n_idx, t_idx]).to(dev)  # [M, n_h] bool
        self.known = torch.from_numpy(known[n_idx, t_idx]).to(dev)
        self.is_val = torch.from_numpy(is_val[n_idx, t_idx]).to(dev)
        self.tile = torch.from_numpy(tile[n_idx, t_idx]).to(dev)
        self.env = torch.from_numpy(n_idx).to(dev)
        self.step = torch.from_numpy(t_idx).to(dev)
        self.val_tiles = val_tiles
        self.n_frames = len(n_idx)
        self.maps = [HeightMap.from_split(d, device=device)]
        self.ice_maps = [HeightMap.ice_from_split(d, device=device)]
        self.map_id = torch.zeros(self.n_frames, dtype=torch.long, device=dev)
        self.dirs = [str(d)]
        self.parts_meta = [self.meta]

    @classmethod
    def merge(cls, dirs: list, device: str = "cuda", val_frac: float = 0.1, seed: int = 0) -> "RiskFrames":
        """Frames of several splits (each with its own height map); validation tiles drawn per split."""
        parts = [cls(d, device=device, val_frac=val_frac, seed=seed + k) for k, d in enumerate(dirs)]
        out = parts[0]
        if len(parts) == 1:
            return out
        for name in ("x", "y", "yaw", "cmd", "policy", "label", "known", "is_val", "tile", "env", "step"):
            setattr(out, name, torch.cat([getattr(p_, name) for p_ in parts]))
        out.map_id = torch.cat([torch.full((p_.n_frames,), k, dtype=torch.long, device=out.x.device) for k, p_ in enumerate(parts)])
        out.maps = [p_.maps[0] for p_ in parts]
        out.ice_maps = [p_.ice_maps[0] for p_ in parts]
        out.dirs = [p_.dirs[0] for p_ in parts]
        out.parts_meta = [p_.meta for p_ in parts]
        out.val_tiles = np.stack([p_.val_tiles for p_ in parts])
        out.n_frames = sum(p_.n_frames for p_ in parts)
        return out

    def patches(self, lay, idx: torch.Tensor) -> torch.Tensor:
        """Receptive-field patches [B, 1, rf, rf] for frames idx, each from its own split's height map."""
        out = torch.empty(len(idx), 1, lay.rf, lay.rf, device=self.x.device)
        mid = self.map_id[idx]
        for k, hm in enumerate(self.maps):
            sel = (mid == k).nonzero().squeeze(1)
            if len(sel):
                i = idx[sel]
                out[sel] = patch(hm, lay, self.x[i], self.y[i], self.yaw[i])
        return out

    def ice_patches(self, lay, idx: torch.Tensor, mode: str = "map") -> torch.Tensor:
        """Ice-channel patches [B, 1, rf, rf] (models.terrain_risk.patches.ice_channel) for frames idx."""
        out = torch.empty(len(idx), 1, lay.rf, lay.rf, device=self.x.device)
        mid = self.map_id[idx]
        for k, im in enumerate(self.ice_maps):
            sel = (mid == k).nonzero().squeeze(1)
            if len(sel):
                i = idx[sel]
                out[sel] = ice_channel(patch(im, lay, self.x[i], self.y[i], self.yaw[i]), mode)
        return out

    def tile_keys(self) -> np.ndarray:
        """Per frame: "<sub-terrain>:<difficulty decile>" of its tile in its own split ("border" off the tiles)."""
        keys = np.empty(self.n_frames, dtype=object)
        mid, tile = self.map_id.cpu().numpy(), self.tile.cpu().numpy()
        for k, meta in enumerate(self.parts_meta):
            ter = meta["terrain"]
            info = {r * ter["num_cols"] + c: f"{name}:{min(9, int(d * 10))}" for r, c, name, d in ter["tiles"]}
            sel = mid == k
            keys[sel] = [info.get(int(t), "border") for t in tile[sel]]
        return keys

    def indices(self, val: bool) -> torch.Tensor:
        return (self.is_val == val).nonzero().squeeze(1)

    def stats(self) -> dict:
        out = {}
        for name, v in (("train", False), ("val", True)):
            idx = self.indices(v)
            k, y = self.known[idx], self.label[idx]
            out[name] = {"frames": int(len(idx)),
                         "positives": [int((y[:, h] & k[:, h]).sum()) for h in range(y.shape[1])],
                         "known": [int(k[:, h].sum()) for h in range(y.shape[1])]}
        return out
