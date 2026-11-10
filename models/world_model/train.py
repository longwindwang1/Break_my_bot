"""Train the world-model ensemble on a split recorded by isaac/collect_wm_data.py.

Loss per sampled (env, step) window, each member with its own minibatch:
  1-step Gaussian NLL of (Delta s, base displacement), normalized units
  + ms_weight * mean over k = 2..K of MSE_k / k, MSE_k = error of the k-step open-loop state prediction in
    Delta-std units (the model's own predicted states are fed back; recorded actions; height scan held at its
    step-t value, since later scans would leak the true future base height)
  + fall_weight * BCE of the fall heads (censored labels masked, positive class re-weighted)
  + 0.01 * (sum max_logvar - sum min_logvar) / (E * out_dim)   (PETS bound penalty, scaled to our mean NLL)

Example (repo root):
  ~/miniforge3/envs/env_isaaclab/bin/python -m models.world_model.train --data logs/wm_data/w4_train --name w4_wm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from isaac.interface import INTERFACE_PATH, REPO_ROOT, load_interface

from .dataset import WMData, train_envs, val_envs
from .ensemble import WorldModel


def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.to(x.dtype)
    return (x * m).sum() / m.sum().clamp(min=1.0)


def wm_spec_sha(itf: dict) -> str:
    return hashlib.sha256(yaml.safe_dump(itf["world_model"], sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def normalizer(d: WMData, n: torch.Tensor, t: torch.Tensor, samples: int = 200_000):
    idx = torch.randperm(len(n), device=n.device)[:samples]
    n, t = n[idx], t[idx]
    x = d.inputs(n, t)[:, : d.layout.dyn_dim]
    y = torch.cat([d.delta(n, t), d.disp[n, t]], dim=1)
    return x.mean(0), x.std(0).clamp(min=1e-3), y.mean(0), y.std(0).clamp(min=1e-6)


def rollout_losses(model: WorldModel, d: WMData, n: torch.Tensor, t: torch.Tensor, K: int, same_batch: bool = False):
    """Losses for windows starting at (n, t). Shapes: n, t [E*B] (member e gets rows e*B..(e+1)*B-1),
    or [B] with same_batch=True (every member sees the same rows; for validation)."""
    E = model.E
    lay = d.layout
    S = lay.state_dim
    if same_batch:
        n, t = n.repeat(E), t.repeat(E)
    B = len(n) // E
    hist = d.history_frames(n, t)
    scan = d.scan[n, t]
    oh = d.onehot(n)
    state_idx = torch.as_tensor(lay.state_in_frame, device=n.device)
    act_idx = torch.arange(S, lay.frame_dim, device=n.device)  # frame = state terms, then actions
    s = hist[:, -1][:, state_idx]
    valid = torch.ones_like(n, dtype=torch.bool)
    out = {}
    ms = []
    for k in range(K):
        tk = (t + k).clamp(max=d.T - 1)
        valid = valid & d.train_ok[n, tk]
        a = d.action[n, tk]
        x = d.assemble(hist, a, scan, oh).view(E, B, -1)
        mean, lv, fall = model(x)
        mean, lv, fall = mean.reshape(E * B, -1), lv.reshape(E * B, -1), fall.reshape(E * B, -1)
        s_next = d.state(n, (tk + 1).clamp(max=d.T - 1))
        if k == 0:
            y = (torch.cat([s_next - s, d.disp[n, t]], dim=1) - model.out_mean) / model.out_std
            nll = ((mean - y) ** 2 * torch.exp(-lv) + lv).mean(dim=1)
            out["nll"] = masked_mean(nll, valid)
            out["nll_state"] = masked_mean(((mean - y) ** 2 * torch.exp(-lv) + lv)[:, :S].mean(dim=1), valid)
            out["mse1"] = masked_mean(((mean - y) ** 2)[:, :S].mean(dim=1), valid)
            out["fall_logits"], out["valid0"] = fall, valid
        delta = mean[:, :S] * model.out_std[:S] + model.out_mean[:S]
        s_pred = s + delta
        if k > 0:  # (k+1)-step prediction; divided by k+1 so every horizon weighs about like the 1-step MSE
            err = ((s_pred - s_next) / model.out_std[:S]) ** 2
            ms.append(masked_mean(err.mean(dim=1), valid) / (k + 1))
        frame = torch.empty_like(hist[:, -1])
        frame[:, state_idx] = s_pred
        frame[:, act_idx] = a
        hist = torch.cat([hist[:, 1:], frame[:, None]], dim=1)
        s = s_pred
    out["ms"] = torch.stack(ms).mean() if ms else torch.zeros((), device=n.device)
    out["ms_per_k"] = [m.item() for m in ms] if same_batch else None
    return out


def fall_loss(logits: torch.Tensor, d: WMData, n: torch.Tensor, t: torch.Tensor, pos_weight: torch.Tensor):
    y, known = d.fall_y[n, t].float(), d.fall_known[n, t]
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
    return masked_mean(bce, known)


def git_hash() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
        return h + ("-dirty" if dirty.strip() else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="train split dir (logs/wm_data/...)")
    p.add_argument("--name", default="wm")
    p.add_argument("--out_root", default=str(REPO_ROOT / "logs" / "wm"), help="run dirs go to <out_root>/<time>_<name>")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=float, default=20.0, help="passes over the training windows (per member)")
    p.add_argument("--max_iters", type=int, default=None, help="cap on iterations (smoke tests)")
    p.add_argument("--batch", type=int, default=1024, help="windows per member per iteration")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--K", type=int, default=5, help="rollout length of the multi-step loss")
    p.add_argument("--ms_weight", type=float, default=1.0)
    p.add_argument("--fall_weight", type=float, default=1.0)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--val_every", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = args.device
    run_dir = Path(args.out_root) / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    itf = load_interface()

    meta = json.loads((Path(args.data) / "meta.json").read_text())
    N = meta["num_envs"]
    t0 = time.time()
    tr = WMData(args.data, device=dev, envs=train_envs(N))
    va = WMData(args.data, device=dev, envs=val_envs(N))
    lay = tr.layout
    tr_n, tr_t = tr.positions(tr.train_ok)
    va_n, va_t = va.positions(va.train_ok)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    vsel = torch.randperm(len(va_n), device=dev, generator=gen)[:50_000]
    va_n, va_t = va_n[vsel], va_t[vsel]
    print(f"[wm] data loaded in {time.time() - t0:.0f} s: train windows {len(tr_n)}, val {len(va_n)}, "
          f"GPU mem {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)

    model = WorldModel(dyn_dim=lay.dyn_dim, n_policy=len(lay.policies), state_dim=lay.state_dim,
                       n_fall=len(lay.fall_horizons), E=args.members, hidden=tuple(args.hidden)).to(dev)
    in_mean, in_std, out_mean, out_std = normalizer(tr, tr_n, tr_t)
    model.in_mean.copy_(in_mean)
    model.in_std.copy_(in_std)
    model.out_mean.copy_(out_mean)
    model.out_std.copy_(out_std)

    known = tr.fall_known[tr_n, tr_t]
    pos = (tr.fall_y[tr_n, tr_t] & known).sum(0).float()
    neg = (~tr.fall_y[tr_n, tr_t] & known).sum(0).float()
    pos_weight = (neg / pos.clamp(min=1)).clamp(1.0, 100.0)
    print(f"[wm] fall positives per horizon {pos.tolist()} of {(pos + neg).tolist()}, pos_weight {pos_weight.tolist()}")

    iters_per_epoch = max(1, len(tr_n) // args.batch)
    total = int(args.epochs * iters_per_epoch)
    if args.max_iters:
        total = min(total, args.max_iters)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / max(1, warmup)) * 0.5 * (1 + np.cos(np.pi * min(1.0, i / max(1, total)))))

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(run_dir)
    E, B = args.members, args.batch
    best, history = float("inf"), []
    info = {"args": vars(args), "git": git_hash(), "data": str(args.data), "data_meta_git": meta.get("git"),
            "train_windows": len(tr_n), "iters": total, "iters_per_epoch": iters_per_epoch,
            "pos_weight": pos_weight.tolist(), "wm_spec_sha256": wm_spec_sha(itf),
            "robot_interface_sha256": hashlib.sha256(INTERFACE_PATH.read_bytes()).hexdigest()}

    def save(path: Path, it: int, val: dict):
        torch.save({"state_dict": model.state_dict(), "model_config": model.config, "iter": it, "val": val, **info}, path)

    print(f"[wm] {total} iterations ({iters_per_epoch} per epoch), run dir {run_dir}", flush=True)
    t0 = time.time()
    for it in range(total):
        model.train()
        idx = torch.randint(len(tr_n), (E * B,), device=dev, generator=gen)
        n, t = tr_n[idx], tr_t[idx]
        o = rollout_losses(model, tr, n, t, args.K)
        lf = fall_loss(o["fall_logits"], tr, n, t, pos_weight)
        # PETS penalty (0.01 per bound) relative to an NLL summed over dims and members; ours is a mean, so scale it
        bound = 0.01 * (model.max_logvar.sum() - model.min_logvar.sum()) / (model.E * model.out_dim)
        loss = o["nll"] + args.ms_weight * o["ms"] + args.fall_weight * lf + bound
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()

        if it % 100 == 0:
            for k, v in {"nll": o["nll"], "ms": o["ms"], "fall_bce": lf, "mse1": o["mse1"], "loss": loss}.items():
                writer.add_scalar(f"train/{k}", v.item(), it)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], it)
        if (it + 1) % args.val_every == 0 or it == total - 1:
            model.eval()
            with torch.no_grad():
                vals = []
                for chunk in torch.split(torch.arange(len(va_n), device=dev), 10_000):
                    vo = rollout_losses(model, va, va_n[chunk], va_t[chunk], args.K, same_batch=True)
                    vf = fall_loss(vo["fall_logits"], va, va_n[chunk].repeat(E), va_t[chunk].repeat(E), pos_weight)
                    vals.append([vo["nll"].item(), vo["nll_state"].item(), vo["mse1"].item(), vo["ms"].item(), vf.item()] + vo["ms_per_k"])
                v = np.mean(vals, axis=0)
            val = {"nll": v[0], "nll_state": v[1], "mse1": v[2], "ms": v[3], "fall_bce": v[4],
                   "ms_per_k": v[5:].tolist(), "iter": it + 1, "elapsed_s": time.time() - t0}
            history.append(val)
            for k in ("nll", "nll_state", "mse1", "ms", "fall_bce"):
                writer.add_scalar(f"val/{k}", val[k], it)
            score = val["nll"] + val["fall_bce"]
            if score < best:
                best = score
                save(run_dir / "model.pt", it + 1, val)
            save(run_dir / "last.pt", it + 1, val)
            rate = (it + 1) / (time.time() - t0)
            print(f"[wm] it {it + 1}/{total}  train nll {o['nll'].item():.3f} ms {o['ms'].item():.3f} fall {lf.item():.3f} | "
                  f"val nll {val['nll']:.3f} mse1 {val['mse1']:.3f} ms {val['ms']:.3f} fall {val['fall_bce']:.3f}"
                  f"{'  *' if score <= best else ''} | {rate:.1f} it/s, eta {(total - it - 1) / rate / 60:.1f} min", flush=True)
            (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    info["wall_s"] = time.time() - t0
    info["best_val"] = min(history, key=lambda h: h["nll"] + h["fall_bce"])
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=1))
    writer.close()
    print(f"[wm] done in {info['wall_s'] / 60:.1f} min; best {run_dir / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
