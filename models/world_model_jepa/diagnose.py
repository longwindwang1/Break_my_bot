"""Diagnostics of a latent world model: what the encoder looks at, and how events move the latent residual.

  encoder sensitivity  share of the target encoder's squared gradient (in normalized-state units) per window
                       frame and per state term, on random training-like steps of the events split
  residual per event   mean r^2 over latent dims 2 s before vs 1-3 s after each event type; dims raised > 2x

  python -m models.world_model_jepa.diagnose --ckpt logs/wm/<run>/model_fall.pt [--data logs/wm_data/w4_events]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from models.world_model.dataset import WMData
from models.world_model.surprisal import state_term_slices

from .data import gather
from .model import load_model
from .predict import predict_latent


def encoder_sensitivity(m, d: WMData, samples: int = 2000, seed: int = 0) -> dict:
    n, t = d.positions(d.train_ok)
    g = torch.Generator(device=n.device).manual_seed(seed)
    sel = torch.randperm(len(n), device=n.device, generator=g)[:samples]
    win = gather(d, n[sel], t[sel], m.H, m.L, 0)["wins"][:, -1].clone().requires_grad_(True)  # [B, L, S]
    z = m.encode(win, target=True)
    J = torch.zeros_like(win[0])
    for k in range(m.d):
        gr = torch.autograd.grad(z[:, k].sum(), win, retain_graph=k < m.d - 1)[0]
        J += (gr * m.s_std).pow(2).mean(0)
    out = {"per_frame_oldest_first": (J.sum(1) / J.sum()).tolist()}
    for name, sl in state_term_slices():
        out[name] = float(J[:, sl].sum() / J.sum())
    return out


@torch.no_grad()
def residual_by_event(m, d: WMData) -> dict:
    P = predict_latent(m, d, keep_latents=True, keep_pred=True)
    mu, var = P["pred"]
    z = P["latents"]
    r2 = torch.zeros_like(z)
    r2[:, 1:] = (z[:, 1:] - mu[:, :-1]).pow(2) / (var[:, :-1] + m.var_floor)
    E = d.meta["event_step"]
    ev = torch.from_numpy(d.info["event_type"].astype(np.int64)).to(z.device)
    out = {}
    for eid, name in enumerate(d.meta["events"]):
        s = ev == eid
        before, after = r2[s][:, E - 100 : E].mean((0, 1)), r2[s][:, E + 50 : E + 150].mean((0, 1))
        ratio = after / before
        out[name] = {"mean_r2_before": float(before.mean()), "mean_r2_after_1_3s": float(after.mean()),
                     "dims_raised_2x": int((ratio > 2).sum()), "max_ratio": float(ratio.max())}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="logs/wm_data/w4_events")
    p.add_argument("--env_stride", type=int, default=4, help="use every k-th env (memory); 4 keeps all event types")
    p.add_argument("--out", default=None, help="optional JSON output path")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    m, _ = load_model(args.ckpt, args.device)
    meta = json.loads(open(f"{args.data}/meta.json").read())
    d = WMData(args.data, device=args.device, envs=np.arange(0, meta["num_envs"], args.env_stride))
    res = {"ckpt": args.ckpt, "data": args.data, "encoder_sensitivity": encoder_sensitivity(m, d), "residual_by_event": residual_by_event(m, d)}
    print(json.dumps(res, indent=1))
    if args.out:
        open(args.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
