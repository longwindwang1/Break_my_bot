"""Fit the fall head of a trained latent world model on the full-randomization split, everything else frozen.

Same procedure as models/world_model/refit_fall.py for the ensemble (fresh head, train envs, early stop on the
val-env BCE): the dynamics never saw ice or pushes, but the fall head must. The head reads the detached predictor
trunk features of the 1-step prediction plus the policy one-hot. Writes <run>/model_fall.pt.

  python -m models.world_model_jepa.refit_heads --run logs/wm/<run> --data logs/wm_data/w4_train
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from models.world_model.dataset import WMData, train_envs, val_envs
from models.world_model.ensemble import EnsembleLinear
from models.world_model.evaluate import auroc
from models.world_model.train import fall_loss

from .data import gather
from .model import LatentWM, load_model


def features(model: LatentWM, d: WMData, n: torch.Tensor, t: torch.Tensor, same_batch: bool) -> torch.Tensor:
    """Detached 1-step trunk features [M, B, feat] at (n, t); rows split over members unless same_batch."""
    M, H = model.M, model.H
    g = gather(d, n, t, H, model.L, 1)
    with torch.no_grad():
        z = model.encode(g["wins"][:, :H])
        x = model.predictor_input(z, g["prev_act"][:, :H], g["act"][:, 0], g["scan"])
        if same_batch:
            x = x.unsqueeze(0).expand(M, -1, -1)
        else:
            x = x.reshape(M, -1, x.shape[-1])
        return model.trunk(x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = args.device
    run = Path(args.run)
    model, ckpt = load_model(run / "model.pt", dev)
    for prm in model.parameters():
        prm.requires_grad_(False)
    for m in model.fall:
        if isinstance(m, EnsembleLinear):
            fresh = EnsembleLinear(m.weight.shape[0], m.weight.shape[1], m.weight.shape[2]).to(dev)
            m.weight.data.copy_(fresh.weight.data)
            m.bias.data.copy_(fresh.bias.data)
    head = list(model.fall.parameters())
    for prm in head:
        prm.requires_grad_(True)

    meta = json.loads((Path(args.data) / "meta.json").read_text())
    N = meta["num_envs"]
    tr = WMData(args.data, device=dev, envs=train_envs(N))
    va = WMData(args.data, device=dev, envs=val_envs(N))
    tr_n, tr_t = (tr.fall_known.any(-1) & (tr.episode_step >= 1)).nonzero(as_tuple=True)
    va_n, va_t = (va.fall_known.any(-1) & (va.episode_step >= 1)).nonzero(as_tuple=True)
    known = tr.fall_known[tr_n, tr_t]
    pos = (tr.fall_y[tr_n, tr_t] & known).sum(0).float()
    neg = (~tr.fall_y[tr_n, tr_t] & known).sum(0).float()
    pos_weight = (neg / pos.clamp(min=1)).clamp(1.0, 100.0)
    print(f"[fall] train steps {len(tr_n)}, positives {pos.tolist()}, pos_weight {pos_weight.tolist()}", flush=True)
    opt = torch.optim.AdamW(head, lr=args.lr, weight_decay=args.weight_decay)
    M, B = model.M, args.batch
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    @torch.no_grad()
    def validate():
        model.eval()
        bces, probs = [], []
        for chunk in torch.split(torch.arange(len(va_n), device=dev), 4_000):
            n, t = va_n[chunk], va_t[chunk]
            h = features(model, va, n, t, same_batch=True)
            fall = model.fall_logits(h, va.onehot(n).unsqueeze(0).expand(M, -1, -1))
            bces.append(fall_loss(fall.reshape(M * len(n), -1), va, n.repeat(M), t.repeat(M), pos_weight).item())
            probs.append(torch.sigmoid(fall).mean(0))
        prob = torch.cat(probs)
        y, kn = va.fall_y[va_n, va_t], va.fall_known[va_n, va_t]
        aucs = [auroc(prob[:, h][kn[:, h]].cpu().numpy(), y[:, h][kn[:, h]].cpu().numpy()) for h in range(prob.shape[1])]
        return float(np.mean(bces)), aucs

    best, bad, t0, best_state = float("inf"), 0, time.time(), None
    for it in range(args.iters):
        model.train()
        idx = torch.randint(len(tr_n), (M * B,), device=dev, generator=gen)
        n, t = tr_n[idx], tr_t[idx]
        h = features(model, tr, n, t, same_batch=False)
        fall = model.fall_logits(h, tr.onehot(n).reshape(M, B, -1))
        loss = fall_loss(fall.reshape(M * B, -1), tr, n, t, pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % args.val_every == 0:
            vb, aucs = validate()
            star = vb < best
            if star:
                best, bad = vb, 0
                best_state = {k: v.detach().clone() for k, v in model.fall.state_dict().items()}
            else:
                bad += 1
            print(f"[fall] it {it + 1} train bce {loss.item():.3f} | val bce {vb:.3f} auroc {[round(a, 3) for a in aucs]}"
                  f"{'  *' if star else ''} | {time.time() - t0:.0f} s", flush=True)
            if bad >= args.patience:
                break
    model.fall.load_state_dict(best_state)
    out = dict(ckpt)
    out["state_dict"] = model.state_dict()
    out["fall_refit"] = {"data": args.data, "best_val_bce": best, "iters": it + 1, "pos_weight": pos_weight.tolist(), "args": vars(args)}
    torch.save(out, run / "model_fall.pt")
    print(f"[fall] best val bce {best:.3f}; wrote {run / 'model_fall.pt'}", flush=True)


if __name__ == "__main__":
    main()
