"""Probes on frozen world-model representations: does the representation carry friction / ice information?

Data: the full-randomization split w4_train (foot mu re-drawn every 4-8 s, half of the changes on a single foot;
pushes), all envs: no probed representation was trained on it (dynamics: w4_train_nominal; only the fall heads,
which are not probed, saw w4_train). Steps with episode step >= 25; probe-train = even envs, probe-test = odd envs
(mu is constant for seconds per env, so a split by env is needed to avoid memorizing envs).
Representations (all at step t, from the model's own inputs at t):
  ensemble  trunk: last trunk layer, mean over members (512)
  latent    z: target-encoder latent of the window ending at t (d); trunk: predictor trunk, mean over members (512)
Targets: min foot mu (R^2), any foot mu < 0.3 (AUROC), which foot is the icy one when exactly one is (accuracy,
4 classes), base_lin_vel (R^2). Probes: linear (ridge / logistic) and a 2 x 256 MLP.

  python -m models.world_model_jepa.probes --model A=ensemble:<ckpt> --model B=latent:<ckpt> --out experiments/w7_jepa_wm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.world_model.dataset import WMData, val_envs
from models.world_model.ensemble import load_model as load_ensemble
from models.world_model.evaluate import auroc

from .data import gather
from .model import load_model as load_latent


@torch.no_grad()
def representations(kind: str, net, d: WMData, n: torch.Tensor, t: torch.Tensor, chunk: int = 8192) -> dict[str, torch.Tensor]:
    out: dict[str, list] = {}
    for c in torch.split(torch.arange(len(n), device=n.device), chunk):
        nn_, tt = n[c], t[c]
        if kind == "ensemble":
            x = d.inputs(nn_, tt)[:, : net.dyn_dim]
            h = net.trunk(((x - net.in_mean) / net.in_std).unsqueeze(0).expand(net.E, -1, -1)).mean(0)
            out.setdefault("trunk", []).append(h)
        else:
            g = gather(d, nn_, tt, net.H, net.L, 1)
            z_ctx = net.encode(g["wins"][:, : net.H])
            xin = net.predictor_input(z_ctx, g["prev_act"][:, : net.H], g["act"][:, 0], g["scan"])
            h = net.trunk(xin.unsqueeze(0).expand(net.M, -1, -1)).mean(0)
            out.setdefault("z", []).append(net.encode(g["wins"][:, net.H - 1], target=True))
            out.setdefault("trunk", []).append(h)
    return {k: torch.cat(v) for k, v in out.items()}


def r2(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(1.0 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def fit_probe(X: torch.Tensor, y: torch.Tensor, Xt: torch.Tensor, task: str, kind: str, steps: int = 3000, seed: int = 0):
    """Returns test predictions: regression values, binary logits or class logits."""
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    X, Xt = (X - mu) / sd, (Xt - mu) / sd
    if kind == "linear" and task == "reg":
        Xb = torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1).double()
        A = Xb.T @ Xb + 1e-3 * len(X) * torch.eye(Xb.shape[1], device=X.device, dtype=torch.double)  # ridge
        w = torch.linalg.solve(A, Xb.T @ y.double())
        return (torch.cat([Xt, torch.ones(len(Xt), 1, device=X.device)], 1).double() @ w).float()
    torch.manual_seed(seed)
    n_out = y.shape[1] if task == "reg" else (1 if task == "bin" else int(y.max()) + 1)
    net = nn.Linear(X.shape[1], n_out) if kind == "linear" else \
        nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, n_out))
    net = net.to(X.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    g = torch.Generator(device=X.device).manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(len(X), (4096,), device=X.device, generator=g)
        o = net(X[idx])
        if task == "reg":
            loss = F.mse_loss(o, y[idx])
        elif task == "bin":
            loss = F.binary_cross_entropy_with_logits(o[:, 0], y[idx].float())
        else:
            loss = F.cross_entropy(o, y[idx].long())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(Xt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, help="NAME=KIND:CHECKPOINT")
    p.add_argument("--data", default="logs/wm_data/w4_train")
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=400_000)
    p.add_argument("--val_envs_only", action="store_true", help="only the split's val envs (less memory, fewer envs)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = args.device
    meta = json.loads((Path(args.data) / "meta.json").read_text())
    envs = val_envs(meta["num_envs"]) if args.val_envs_only else None
    d = WMData(args.data, device=dev, envs=envs, extra=True)
    sel = (d.episode_step >= 25) & (d.trans_ok)
    n, t = sel.nonzero(as_tuple=True)
    g = torch.Generator(device=dev).manual_seed(0)
    keep = torch.randperm(len(n), device=dev, generator=g)[: args.samples]
    n, t = n[keep], t[keep]
    mu = torch.from_numpy(d.foot_mu).to(dev)[n, t]  # [B, 4]
    icy = mu < 0.3
    targets = {"mu_min": ("reg", mu.min(1).values[:, None]), "any_icy": ("bin", icy.any(1)),
               "base_lin_vel": ("reg", d.state(n, t)[:, 0:3])}
    one = icy.sum(1) == 1
    targets["icy_foot"] = ("cls", icy.float().argmax(1), one)
    test = torch.from_numpy(d.env_ids % 2 == 1).to(dev)[n]  # by original env index
    res = {"data": args.data, "frames": int(len(n)), "frames_any_icy": float(icy.any(1).float().mean()), "single_icy_frames": int(one.sum())}
    # representations first (kept on the CPU), then free the split, then fit the probes: fits in 8 GB
    reps_all = {}
    for spec in args.model:
        name, _, rest = spec.partition("=")
        kind, _, ckpt = rest.partition(":")
        net = load_ensemble(ckpt, dev)[0] if kind == "ensemble" else load_latent(ckpt, dev)[0]
        reps_all[name] = {k: v.cpu() for k, v in representations(kind, net, d, n, t).items()}
        del net
        torch.cuda.empty_cache()
    del d
    torch.cuda.empty_cache()
    for name, reps in reps_all.items():
        for rname, Xc in reps.items():
            X = Xc.to(dev)
            for tname, tgt in targets.items():
                task, y = tgt[0], tgt[1]
                m = tgt[2] if len(tgt) > 2 else torch.ones_like(test)
                tr_m, te_m = m & ~test, m & test
                for pk in ("linear", "mlp"):
                    o = fit_probe(X[tr_m], y[tr_m], X[te_m], task, pk)
                    yt = y[te_m]
                    if task == "reg":
                        v = r2(o, yt)
                    elif task == "bin":
                        v = auroc(o[:, 0].cpu().numpy(), yt.cpu().numpy())
                    else:
                        v = float((o.argmax(1) == yt).float().mean())
                    res.setdefault(name, {}).setdefault(rname, {})[f"{tname}/{pk}"] = v
                    print(f"[probe] {name} {rname:5s} {tname:12s} {pk:6s} {v:.3f}", flush=True)
            del X
            torch.cuda.empty_cache()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probes.json").write_text(json.dumps(res, indent=1))
    print(f"[probe] wrote {out / 'probes.json'}")


if __name__ == "__main__":
    main()
