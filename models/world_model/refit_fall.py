"""Refit the fall head of a trained world model on another split, with the trunk frozen (W5).

The surprisal model is trained on normal physics only, so that ice and pushes stay surprising; its fall
head must still know about falls on ice and after pushes. This script keeps the trunk and dynamics heads,
re-initializes the fall head, trains it on the full-randomization split (train envs) and stops on the
validation BCE (val envs). The result is written to <run>/model_fall.pt with the same format as model.pt.

  ~/miniforge3/envs/env_isaaclab/bin/python -m models.world_model.refit_fall --run logs/wm/<run> \
      --data logs/wm_data/w4_train
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import WMData, train_envs, val_envs
from .ensemble import EnsembleLinear, load_model
from .evaluate import auroc
from .train import fall_loss


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--patience", type=int, default=5, help="validations without improvement before stopping")
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
    for m in model.fall:  # fresh head
        if isinstance(m, EnsembleLinear):
            fresh = EnsembleLinear(m.weight.shape[0], m.weight.shape[1], m.weight.shape[2]).to(dev)
            m.weight.data.copy_(fresh.weight.data)
            m.bias.data.copy_(fresh.bias.data)
    head_params = list(model.fall.parameters())
    for prm in head_params:
        prm.requires_grad_(True)

    meta = json.loads((Path(args.data) / "meta.json").read_text())
    N = meta["num_envs"]
    tr = WMData(args.data, device=dev, envs=train_envs(N))
    va = WMData(args.data, device=dev, envs=val_envs(N))
    sel_tr = tr.fall_known.any(-1) & (tr.episode_step >= 1)
    sel_va = va.fall_known.any(-1) & (va.episode_step >= 1)
    tr_n, tr_t = sel_tr.nonzero(as_tuple=True)
    va_n, va_t = sel_va.nonzero(as_tuple=True)
    known = tr.fall_known[tr_n, tr_t]
    pos = (tr.fall_y[tr_n, tr_t] & known).sum(0).float()
    neg = (~tr.fall_y[tr_n, tr_t] & known).sum(0).float()
    pos_weight = (neg / pos.clamp(min=1)).clamp(1.0, 100.0)
    print(f"[fall] train steps {len(tr_n)}, positives {pos.tolist()}, pos_weight {pos_weight.tolist()}", flush=True)

    opt = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=args.weight_decay)
    E, B = model.E, args.batch
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    @torch.no_grad()
    def validate() -> tuple[float, list[float]]:
        model.eval()
        bces, probs = [], []
        for chunk in torch.split(torch.arange(len(va_n), device=dev), 20_000):
            n, t = va_n[chunk], va_t[chunk]
            x = va.inputs(n, t).unsqueeze(0).expand(E, -1, -1)
            _, _, fall = model(x)
            bces.append(fall_loss(fall.reshape(E * len(n), -1), va, n.repeat(E), t.repeat(E), pos_weight).item())
            probs.append(torch.sigmoid(fall).mean(0))
        prob = torch.cat(probs)
        y, kn = va.fall_y[va_n, va_t], va.fall_known[va_n, va_t]
        aucs = [auroc(prob[:, h][kn[:, h]].cpu().numpy(), y[:, h][kn[:, h]].cpu().numpy()) for h in range(prob.shape[1])]
        return float(np.mean(bces)), aucs

    best, bad, t0 = float("inf"), 0, time.time()
    best_state = None
    for it in range(args.iters):
        model.train()
        idx = torch.randint(len(tr_n), (E * B,), device=dev, generator=gen)
        n, t = tr_n[idx], tr_t[idx]
        x = tr.inputs(n, t).view(E, B, -1)
        _, _, fall = model(x)
        loss = fall_loss(fall.reshape(E * B, -1), tr, n, t, pos_weight)
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
            print(f"[fall] it {it + 1} train bce {loss.item():.3f} | val bce {vb:.3f} auroc {[round(a, 3) for a in aucs]}{'  *' if star else ''} "
                  f"| {time.time() - t0:.0f} s", flush=True)
            if bad >= args.patience:
                break
    model.fall.load_state_dict(best_state)
    out = dict(ckpt)
    out["state_dict"] = model.state_dict()
    out["fall_refit"] = {"data": args.data, "best_val_bce": best, "iters": it + 1, "pos_weight": pos_weight.tolist(),
                         "args": vars(args)}
    torch.save(out, run / "model_fall.pt")
    print(f"[fall] best val bce {best:.3f}; wrote {run / 'model_fall.pt'}", flush=True)


if __name__ == "__main__":
    main()
