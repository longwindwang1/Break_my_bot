"""Window gathering for the latent world model, on the world-model splits (models/world_model/dataset.py WMData).

Offsets are relative to the prediction step t. The encoder window ending at offset o covers the states at
o - L + 1 .. o; the predictor context is the latents at offsets -(H - 1) .. 0. Frames before the episode start
are filled with its first frame, exactly as WMData.history_frames does for the ensemble.
"""

from __future__ import annotations

import torch

from models.world_model.dataset import WMData


def gather(d: WMData, n: torch.Tensor, t: torch.Tensor, H: int, L: int, K: int) -> dict[str, torch.Tensor]:
    """Everything a K-step training rollout from (n, t) needs. Shapes for B = len(n):
    wins     [B, H + K, L, S]  raw state windows ending at offsets -(H-1) .. K (index i <-> offset i - (H - 1))
    prev_act [B, H + K, A]     action term of the frames at offsets -(H-1) .. K (frame u holds a_{u-1})
    act      [B, K, A]         a_{t+k}, k = 0 .. K-1
    scan     [B, scan_dim]     height scan at t (held during rollouts, as the ensemble does)
    valid    [B, K]            the (k+1)-step prediction is valid (train_ok along the way, as the ensemble)
    states_next [B, K, S]      s_{t+k+1} (raw)
    """
    lay = d.layout
    dev = n.device
    lo = (H - 1) + (L - 1)
    offs = torch.arange(-lo, K + 1, device=dev)
    tt = torch.maximum(t[:, None] + offs[None], d.ep_start[n, t][:, None]).clamp(max=d.T - 1)
    fr = d.frames[n[:, None], tt]  # [B, W, F]
    state_idx = torch.as_tensor(lay.state_in_frame, device=dev)
    act_idx = torch.arange(lay.state_dim, lay.frame_dim, device=dev)  # frame = state terms, then actions
    states = fr[..., state_idx]
    out = {"wins": states.unfold(1, L, 1).transpose(-1, -2),
           "prev_act": fr[:, L - 1 :][..., act_idx],
           "scan": d.scan[n, t].float(),
           "states_next": states[:, lo + 1 :]}
    if K > 0:
        tk = (t[:, None] + torch.arange(K, device=dev)[None]).clamp(max=d.T - 1)
        out["act"] = d.action[n[:, None], tk]
        out["valid"] = torch.cumprod(d.train_ok[n[:, None], tk].int(), dim=1).bool()
    return out


@torch.no_grad()
def normalizer(d: WMData, n: torch.Tensor, t: torch.Tensor, samples: int = 200_000) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per-dim mean / std of states, actions, scan and base displacement over sampled training steps."""
    idx = torch.randperm(len(n), device=n.device)[:samples]
    n, t = n[idx], t[idx]
    s = d.state(n, t)
    out = {"s": (s.mean(0), s.std(0).clamp(min=1e-3)),
           "a": (d.action[n, t].mean(0), d.action[n, t].std(0).clamp(min=1e-3)),
           "scan": (d.scan[n, t].float().mean(0), d.scan[n, t].float().std(0).clamp(min=1e-3)),
           "disp": (d.disp[n, t].mean(0), d.disp[n, t].std(0).clamp(min=1e-6))}
    return out
