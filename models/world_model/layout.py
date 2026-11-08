"""World-model input/state index bookkeeping, derived from configs/robot_interface.yaml only.

A "frame" is one step of the world model's history: the history terms of world_model.input[0] (45-d:
base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, actions), taken from the policy
observation vector. The state s (33-d) is the state_terms part of a frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isaac.interface import load_interface, obs_term_slices, wm_input_blocks, wm_input_dim, wm_state_dim


def history_perm(term_dims: list[int], history_length: int) -> np.ndarray:
    """Permutation p such that frames.reshape(B, H*F)[:, p] is the term-by-term history layout
    (each term block oldest -> newest), for frames [B, H, F] ordered oldest first."""
    F = sum(term_dims)
    offsets = np.cumsum([0] + term_dims[:-1])
    return np.concatenate(
        [(h * F + o + np.arange(d)) for o, d in zip(offsets, term_dims) for h in range(history_length)]
    ).astype(np.int64)


@dataclass
class WMLayout:
    frame_idx: np.ndarray  # [F] positions in the policy observation vector that form one frame
    frame_term_dims: list[int]
    history_length: int
    state_in_frame: np.ndarray  # [S] positions of the state terms inside a frame
    hist_perm: np.ndarray  # [H*F]
    offsets: dict[str, tuple[int, int]]  # input block name -> (start, end) in wm_in
    input_dim: int
    state_dim: int
    current_state_in_input: np.ndarray  # [S] positions in wm_in of the newest frame's state terms
    policies: list[str]
    fall_horizons: list[int]

    @property
    def frame_dim(self) -> int:
        return len(self.frame_idx)

    @property
    def dyn_dim(self) -> int:
        """Inputs the dynamics trunk reads (everything except the policy one-hot)."""
        return self.offsets["policy_onehot"][0]


def load_layout(itf: dict | None = None) -> WMLayout:
    itf = itf or load_interface()
    wm = itf["world_model"]
    slices = obs_term_slices(itf)
    blocks = wm_input_blocks(itf)
    names = [b["name"] for b in blocks]
    if names != ["history", "action", "height_scan", "policy_onehot"]:
        raise ValueError(f"unexpected world_model.input blocks {names}")
    hist = blocks[0]
    terms = [t["name"] for t in hist["terms"]]
    dims = [t["dim"] for t in hist["terms"]]
    frame_idx = np.concatenate([np.arange(slices[n].start, slices[n].stop) for n in terms])
    frame_off = dict(zip(terms, np.cumsum([0] + dims[:-1])))
    state_terms = wm["state_terms"]
    state_in_frame = np.concatenate([frame_off[n] + np.arange(dims[terms.index(n)]) for n in state_terms])
    offsets, i = {}, 0
    for b in blocks:
        offsets[b["name"]] = (i, i + b["dim"])
        i += b["dim"]
    H = hist["history_length"]
    term_block = dict(zip(terms, H * np.cumsum([0] + dims[:-1])))
    current = np.concatenate(
        [term_block[n] + (H - 1) * dims[terms.index(n)] + np.arange(dims[terms.index(n)]) for n in state_terms]
    )
    layout = WMLayout(
        frame_idx=frame_idx.astype(np.int64),
        frame_term_dims=dims,
        history_length=H,
        state_in_frame=state_in_frame.astype(np.int64),
        hist_perm=history_perm(dims, H),
        offsets=offsets,
        input_dim=wm_input_dim(itf),
        state_dim=wm_state_dim(itf),
        current_state_in_input=current.astype(np.int64),
        policies=list(wm["policies"]),
        fall_horizons=list(wm["fall_horizons_steps"]),
    )
    assert layout.input_dim == wm["input_dim"] == i, (layout.input_dim, wm["input_dim"], i)
    # the training rollout and the exporter assume frame = [state terms, actions]
    assert terms == list(state_terms) + ["actions"], terms
    assert len(layout.state_in_frame) == layout.state_dim
    return layout
