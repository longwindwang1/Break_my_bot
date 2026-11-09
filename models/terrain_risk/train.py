"""Train the terrain-risk network on a geo split recorded by isaac/collect_wm_data.py (W6).

Each sample is one recorded frame: a 17 x 17 height patch centred on the base and rotated to its yaw, plus the
command and policy, labelled with the censored "fall within 25 / 50 steps" labels. Positives (fall within 50
steps) are oversampled to --pos_frac of every batch; left-right mirroring (patch flipped in y, vy and wz
negated) doubles the terrain variety. After training, Platt scaling on the validation tiles (natural class
balance) turns the logits back into calibrated probabilities. Checkpoint: best mean validation AUROC.

  ~/miniforge3/envs/env_isaaclab/bin/python -m models.terrain_risk.train --data logs/wm_data/w6_geo_train --name w6_risk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from isaac.interface import INTERFACE_PATH, REPO_ROOT

from models.world_model.evaluate import auroc, average_precision
from models.world_model.train import git_hash

from .dataset import RiskFrames
from .model import RiskNet
from .patches import cond_vector, load_risk_layout


def batch_inputs(fr: RiskFrames, lay, idx: torch.Tensor, mirror: torch.Tensor | None = None, ice_mode: str = "map"):
    """(height [B, 1, rf, rf], ice [B, 1, rf, rf], cond [B, 5]) for frames idx, optionally mirrored left-right."""
    h = fr.patches(lay, idx)
    ice = fr.ice_patches(lay, idx, ice_mode)
    cmd = fr.cmd[idx].clone()
    if mirror is not None:
        m4 = mirror[:, None, None, None]
        h = torch.where(m4, h.flip(2), h)  # rows are y: flip y -> -y
        ice = torch.where(m4, ice.flip(2), ice)
        cmd[:, 1:] = torch.where(mirror[:, None], -cmd[:, 1:], cmd[:, 1:])  # vy, wz change sign
    return h, ice, cond_vector(cmd, fr.policy[idx])


@torch.no_grad()
def predict_logits(model: RiskNet, fr: RiskFrames, lay, idx: torch.Tensor, chunk: int = 8192, ice_mode: str = "map") -> torch.Tensor:
    out = []
    for c in torch.split(idx, chunk):
        h, ice, cond = batch_inputs(fr, lay, c, ice_mode=ice_mode)
        out.append(model.logits(h, cond, ice)[:, :, 0, 0])
    return torch.cat(out)


def metrics(logits: torch.Tensor, fr: RiskFrames, idx: torch.Tensor) -> dict:
    res = {}
    for k in range(logits.shape[1]):
        m = fr.known[idx, k]
        s, y = logits[m, k].cpu().numpy(), fr.label[idx, k][m].cpu().numpy()
        res[f"h{k}"] = {"auroc": auroc(s, y), "ap": average_precision(s, y), "positives": int(y.sum()), "count": int(len(y))}
    return res


def fit_platt(logits: torch.Tensor, fr: RiskFrames, idx: torch.Tensor) -> tuple[list[float], list[float]]:
    scales, biases = [], []
    for k in range(logits.shape[1]):
        m = fr.known[idx, k]
        z, y = logits[m, k].double(), fr.label[idx, k][m].double()
        a = torch.ones(1, dtype=torch.float64, device=z.device, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, device=z.device, requires_grad=True)
        opt = torch.optim.LBFGS([a, b], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(a * z + b, y)
            loss.backward()
            return loss

        opt.step(closure)
        scales.append(float(a))
        biases.append(float(b))
    return scales, biases


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, nargs="+", help="one or more geo split dirs")
    p.add_argument("--name", default="risk")
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--pos_frac", type=float, default=0.1)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--no_ice", action="store_true", help="height-only model (phase 1)")
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--val_samples", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    lay = load_risk_layout()
    run_dir = REPO_ROOT / "logs" / "terrain_risk" / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fr = RiskFrames.merge(args.data, device=dev, seed=args.seed)
    tr_idx, va_idx = fr.indices(False), fr.indices(True)
    pos_idx = tr_idx[fr.label[tr_idx, -1] & fr.known[tr_idx, -1]]
    g = torch.Generator(device=dev).manual_seed(args.seed)
    va_sub = va_idx[torch.randperm(len(va_idx), device=dev, generator=g)[: args.val_samples]]
    print(f"[risk] {fr.stats()} | train positives (h50) {len(pos_idx)} | loaded in {time.time() - t0:.0f} s", flush=True)

    model = RiskNet(width=args.width, n_cond=len(lay.cond), n_out=len(lay.horizons), n_ice=0 if args.no_ice else 1).to(dev)
    assert model.rf == lay.rf, (model.rf, lay.rf)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    n_pos = int(args.pos_frac * args.batch)
    best, history = -1.0, []
    info = {"args": vars(args), "git": git_hash(), "data": args.data, "stats": fr.stats(),
            "val_tiles": fr.val_tiles.tolist(), "robot_interface_sha256": hashlib.sha256(INTERFACE_PATH.read_bytes()).hexdigest()}
    t0 = time.time()
    for it in range(args.iters):
        model.train()
        idx = torch.cat([pos_idx[torch.randint(len(pos_idx), (n_pos,), device=dev, generator=g)],
                         tr_idx[torch.randint(len(tr_idx), (args.batch - n_pos,), device=dev, generator=g)]])
        mirror = torch.rand(len(idx), device=dev, generator=g) < 0.5
        h, ice, cond = batch_inputs(fr, lay, idx, mirror)
        z = model.logits(h, cond, ice)[:, :, 0, 0]
        k = fr.known[idx]
        loss = (F.binary_cross_entropy_with_logits(z, fr.label[idx].float(), reduction="none") * k).sum() / k.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % args.val_every == 0 or it == args.iters - 1:
            model.eval()
            lv = predict_logits(model, fr, lay, va_sub)
            m = metrics(lv, fr, va_sub)
            score = float(np.nan_to_num(np.mean([v["auroc"] for v in m.values()]), nan=0.0))
            history.append({"iter": it + 1, "loss": loss.item(), **m})
            star = score > best
            if star:
                best = score
                torch.save({"state_dict": model.state_dict(), "model_config": model.config, "iter": it + 1, "val": m, **info},
                           run_dir / "model.pt")
            print(f"[risk] it {it + 1} loss {loss.item():.4f} | val " + "  ".join(
                f"h{lay.horizons[i]} auroc {v['auroc']:.3f} ap {v['ap']:.3f}" for i, v in enumerate(m.values()))
                + f"{'  *' if star else ''} | {time.time() - t0:.0f} s", flush=True)
            (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    # Platt scaling of the best checkpoint on the validation tiles
    ckpt = torch.load(run_dir / "model.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    lv = predict_logits(model, fr, lay, va_idx)
    scales, biases = fit_platt(lv, fr, va_idx)
    model.platt_scale.copy_(torch.tensor(scales, device=dev))
    model.platt_bias.copy_(torch.tensor(biases, device=dev))
    probs = torch.sigmoid(lv * model.platt_scale + model.platt_bias)
    calib = {f"h{lay.horizons[k]}": {"mean_prob": float(probs[fr.known[va_idx, k], k].mean()),
                                     "base_rate": float(fr.label[va_idx, k][fr.known[va_idx, k]].float().mean())}
             for k in range(len(lay.horizons))}
    ckpt.update(state_dict=model.state_dict(), platt={"scale": scales, "bias": biases, "check": calib},
                wall_s=time.time() - t0, history=history)
    torch.save(ckpt, run_dir / "model.pt")
    (run_dir / "run_info.json").write_text(json.dumps({k: v for k, v in ckpt.items() if k not in ("state_dict",)}, indent=1, default=str))
    print(f"[risk] done in {(time.time() - t0) / 60:.1f} min; best val mean AUROC {best:.3f}; platt {scales} {biases}; {calib}", flush=True)
    print(f"[risk] wrote {run_dir / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
