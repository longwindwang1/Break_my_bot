"""Train the latent (JEPA-style) world model on a world-model split (models/world_model_jepa/model.py).

Loss per sampled (env, step) window, each of the M predictor members with its own minibatch (as the ensemble):
  w_pred * [ MSE_1 + ms_weight * mean_{k=2..K} MSE_k / k ]
      MSE_k = mean over latent dims of (ẑ_{t+k} - sg(Ē(window ending t+k)))^2, ẑ rolled out in latent space with the
      recorded actions and the scan held at t (as the ensemble's multi-step loss)
  + Gaussian NLL of the 1-step target under (mean detached, logvar head on detached features) + PETS bound penalty
  + MSE of the displacement head (normalized units; detached features)
  + reg vicreg: w_var * variance hinge + w_cov * covariance penalty on E's outputs (current window)
    reg ema:    nothing (EMA target + output LayerNorm)
    reg recon:  w_rec * (MSE of decoding the current window from z_t + MSE of decoding s_{t+k} from ẑ_{t+k})
The target encoder is an EMA of the encoder (tau cosine from tau0 to 1). After training, the predictive-variance
floor and the residual whitening matrix are estimated (on train and val envs) and stored in the checkpoint.

Runs on any CUDA machine with torch + numpy + pyyaml (no Isaac Sim); see cloud/README.md.
  python -m models.world_model_jepa.train --data logs/wm_data/w4_train_nominal --name w7_jepa --reg vicreg --d 128 --L 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from isaac.interface import INTERFACE_PATH, REPO_ROOT, load_interface
from models.world_model.dataset import WMData, train_envs, val_envs
from models.world_model.train import git_hash, masked_mean, wm_spec_sha

from .data import gather, normalizer
from .model import LatentWM, cosine_tau, effective_rank, vicreg_terms


def rollout(model: LatentWM, d: WMData, n: torch.Tensor, t: torch.Tensor, K: int, M: int, args, same_batch: bool = False) -> dict:
    """Losses and diagnostics for windows (n, t) [M*B] (member m gets rows m*B..), or [B] with same_batch."""
    if same_batch:
        n, t = n.repeat(M), t.repeat(M)
    B = len(n) // M
    H, L = model.H, model.L
    g = gather(d, n, t, H, L, K)
    wins = g["wins"]
    z_ctx = model.encode(wins[:, :H])  # [MB, H, d]
    with torch.no_grad():
        z_tg = model.encode(wins[:, H - 1 :], target=True)  # offsets 0..K
    v = lambda x: x.reshape(M, B, *x.shape[1:])  # noqa: E731
    z_hist, z_tg_m = v(z_ctx), v(z_tg)
    prev_act, act, scan, valid = v(g["prev_act"]), v(g["act"]), v(g["scan"]), v(g["valid"])
    out, ms, rec_next = {}, [], []
    for k in range(K):
        x = model.predictor_input(z_hist, prev_act[:, :, k : k + H], act[:, :, k], scan)
        mean, lv, h = model.step(x, z_hist[:, :, -1])
        target = z_tg_m[:, :, k + 1]
        vk = valid[:, :, k]
        mse = ((mean - target) ** 2).mean(-1)
        if k == 0:
            out["mse1"] = masked_mean(mse, vk)
            hold = ((z_tg_m[:, :, 0] - target) ** 2).mean(-1)
            out["hold1"] = masked_mean(hold, vk)
            out["nll"] = masked_mean((((mean.detach() - target) ** 2) * torch.exp(-lv) + lv).mean(-1), vk)
            dn = v((d.disp[n, t] - model.disp_mean) / model.disp_std)
            out["disp"] = masked_mean(((model.disp(h) - dn) ** 2).mean(-1), vk)
            tv = target.reshape(-1, target.shape[-1])
            out["norm_err"] = masked_mean(mse, vk) / tv.var(0).mean().clamp(min=1e-12)
            out["valid0"] = vk
        else:
            ms.append(masked_mean(mse, vk) / (k + 1))
        if model.reg == "recon":
            sn = model.norm_states(v(g["states_next"])[:, :, k])
            rec_next.append(masked_mean(((model.decoder_next(mean) - sn) ** 2).mean(-1), vk))
        z_hist = torch.cat([z_hist[:, :, 1:], mean[:, :, None]], dim=2)
    out["ms"] = torch.stack(ms).mean() if ms else torch.zeros((), device=n.device)
    out["ms_per_k"] = [m.item() for m in ms] if same_batch else None
    out["pred"] = out["mse1"] + args.ms_weight * out["ms"]
    z_now = z_ctx[:, -1]
    out["z_now"] = z_now
    if model.reg == "vicreg":
        out["var"], out["cov"] = vicreg_terms(z_now)
    elif model.reg == "recon":
        rec = ((model.decoder(z_now) - model.norm_states(wins[:, H - 1]).flatten(-2)) ** 2).mean(-1).mean()
        out["rec"] = rec + torch.stack(rec_next).mean()
    return out


def total_loss(model: LatentWM, o: dict, args) -> torch.Tensor:
    bound = 0.01 * (model.max_logvar.sum() - model.min_logvar.sum()) / (model.M * model.d)
    loss = args.w_pred * o["pred"] + o["nll"] + o["disp"] + bound
    if model.reg == "vicreg":
        loss = loss + args.w_var * o["var"] + args.w_cov * o["cov"]
    elif model.reg == "recon":
        loss = loss + args.w_rec * o["rec"]
    return loss


@torch.no_grad()
def residual_stats(model: LatentWM, tr: WMData, tr_n, tr_t, va: WMData, va_n, va_t, samples: int = 200_000, chunk: int = 4_000):
    """var_floor = (0.1 * std of the 1-step target-latent delta)^2 on train windows; whitening matrix W with
    W C W^T = I for C = covariance of the standardized 1-step residual on val windows (Mahalanobis)."""
    M, H, L = model.M, model.H, model.L
    g = torch.Generator(device=tr_n.device).manual_seed(1)
    sel = torch.randperm(len(tr_n), device=tr_n.device, generator=g)[:samples]
    deltas = []
    for c in torch.split(sel, chunk):
        w = gather(tr, tr_n[c], tr_t[c], H, L, 1)["wins"]
        z = model.encode(w[:, H - 1 :], target=True)
        deltas.append(z[:, 1] - z[:, 0])
    floor = (0.1 * torch.cat(deltas).std(0)) ** 2
    model.var_floor.copy_(floor)
    sel = torch.randperm(len(va_n), device=va_n.device, generator=g)[:samples]
    res = []
    for c in torch.split(sel, chunk):
        n, t = va_n[c], va_t[c]
        gg = gather(va, n, t, H, L, 1)
        z_ctx = model.encode(gg["wins"][:, :H])
        z_tg = model.encode(gg["wins"][:, H:H + 1], target=True)[:, 0]
        x = model.predictor_input(z_ctx, gg["prev_act"][:, :H], gg["act"][:, 0], gg["scan"])
        mean, lv, _ = model.step(x.unsqueeze(0).expand(M, -1, -1), z_ctx[:, -1].unsqueeze(0).expand(M, -1, -1))
        mu, var = LatentWM.moments(mean, lv)
        res.append((z_tg - mu) / torch.sqrt(var + floor))
    r = torch.cat(res).double()
    C = torch.cov(r.T)
    Lc = torch.linalg.cholesky(C + 1e-6 * torch.eye(C.shape[0], dtype=C.dtype, device=C.device))
    model.whiten.copy_(torch.linalg.inv(Lc).float())  # u = W r has identity covariance
    R = torch.corrcoef(r.T)
    return {"z_std_val": float(r.std(0).mean()), "floor_mean": float(floor.mean()),
            "corr_offdiag_abs_mean": float((R - torch.diag(torch.diag(R))).abs().sum() / (R.shape[0] * (R.shape[0] - 1)))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="train split dir (logs/wm_data/...)")
    p.add_argument("--name", default="jepa")
    p.add_argument("--out_root", default=str(REPO_ROOT / "logs" / "wm"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=float, default=20.0, help="passes over the training windows (per member)")
    p.add_argument("--max_iters", type=int, default=None, help="cap on iterations (smoke tests)")
    p.add_argument("--batch", type=int, default=1024, help="windows per member per iteration")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--ms_weight", type=float, default=1.0)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--d", type=int, default=128, help="latent dim")
    p.add_argument("--L", type=int, default=3, help="states per encoder window")
    p.add_argument("--reg", default="vicreg", choices=["vicreg", "ema", "recon"])
    p.add_argument("--w_pred", type=float, default=25.0)
    p.add_argument("--w_var", type=float, default=25.0)
    p.add_argument("--w_cov", type=float, default=1.0)
    p.add_argument("--w_rec", type=float, default=1.0)
    p.add_argument("--tau0", type=float, default=0.996)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--enc_hidden", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--val_every", type=int, default=2000)
    p.add_argument("--val_chunk", type=int, default=2000, help="val windows per chunk (x members); lower if GPU memory is short")
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
    va_n50, va_t50 = va_n[vsel], va_t[vsel]
    print(f"[jepa] data loaded in {time.time() - t0:.0f} s: train windows {len(tr_n)}, val {len(va_n50)}, "
          f"GPU mem {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)

    M = args.members
    model = LatentWM(state_dim=lay.state_dim, act_dim=lay.frame_dim - lay.state_dim, scan_dim=tr.scan.shape[-1],
                     n_policy=len(lay.policies), H=lay.history_length, L=args.L, d=args.d, M=M, reg=args.reg,
                     enc_hidden=tuple(args.enc_hidden), hidden=tuple(args.hidden), n_fall=len(lay.fall_horizons)).to(dev)
    for name, (m, s) in normalizer(tr, tr_n, tr_t).items():
        getattr(model, f"{name}_mean").copy_(m)
        getattr(model, f"{name}_std").copy_(s)
    model.target.load_state_dict(model.encoder.state_dict())

    params = [p_ for n_, p_ in model.named_parameters() if p_.requires_grad and not n_.startswith("fall.")]
    iters_per_epoch = max(1, len(tr_n) // args.batch)
    total = int(args.epochs * iters_per_epoch)
    if args.max_iters:
        total = min(total, args.max_iters)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warmup = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / max(1, warmup)) * 0.5 * (1 + np.cos(np.pi * min(1.0, i / max(1, total)))))

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(run_dir)
    B = args.batch
    history = []
    info = {"args": vars(args), "git": git_hash(), "data": str(args.data), "data_meta_git": meta.get("git"),
            "train_windows": len(tr_n), "iters": total, "iters_per_epoch": iters_per_epoch, "kind": "latent_wm",
            "wm_spec_sha256": wm_spec_sha(itf), "robot_interface_sha256": hashlib.sha256(INTERFACE_PATH.read_bytes()).hexdigest()}

    def save(path: Path, it: int, val: dict):
        torch.save({"state_dict": model.state_dict(), "model_config": model.config, "iter": it, "val": val, **info}, path)

    print(f"[jepa] {args.reg} d {args.d} L {args.L}: {total} iterations ({iters_per_epoch} per epoch), run dir {run_dir}", flush=True)
    t0 = time.time()
    val = {}
    for it in range(total):
        model.train()
        idx = torch.randint(len(tr_n), (M * B,), device=dev, generator=gen)
        o = rollout(model, tr, tr_n[idx], tr_t[idx], args.K, M, args)
        loss = total_loss(model, o, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        sched.step()
        model.ema_update(cosine_tau(it, total, args.tau0))

        if it % 100 == 0:
            logs = {k: o[k] for k in ("pred", "mse1", "hold1", "ms", "nll", "disp", "norm_err", "var", "cov", "rec") if k in o}
            for k, x in {**logs, "loss": loss}.items():
                writer.add_scalar(f"train/{k}", x.item(), it)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], it)
        if (it + 1) % args.val_every == 0 or it == total - 1:
            model.eval()
            with torch.no_grad():
                acc, zs = [], []
                for chunk in torch.split(torch.arange(len(va_n50), device=dev), args.val_chunk):
                    vo = rollout(model, va, va_n50[chunk], va_t50[chunk], args.K, M, args, same_batch=True)
                    acc.append([vo[k].item() for k in ("mse1", "hold1", "ms", "nll", "disp", "norm_err")] + vo["ms_per_k"])
                    zs.append(vo["z_now"][: len(chunk)])
                a = np.mean(acc, axis=0)
                z = torch.cat(zs)
                zstd = z.std(0)
            val = {"mse1": a[0], "hold1": a[1], "pred_over_hold": a[0] / max(a[1], 1e-12), "ms": a[2], "nll": a[3],
                   "disp": a[4], "norm_err": a[5], "ms_per_k": a[6:].tolist(), "eff_rank": effective_rank(z[:20_000]),
                   "z_std_min": float(zstd.min()), "z_std_mean": float(zstd.mean()), "iter": it + 1,
                   "elapsed_s": time.time() - t0}
            history.append(val)
            for k in ("mse1", "pred_over_hold", "ms", "nll", "disp", "norm_err", "eff_rank", "z_std_min", "z_std_mean"):
                writer.add_scalar(f"val/{k}", val[k], it)
            save(run_dir / "last.pt", it + 1, val)
            rate = (it + 1) / (time.time() - t0)
            print(f"[jepa] it {it + 1}/{total}  train pred {o['pred'].item():.4f} | val mse1/hold {val['pred_over_hold']:.3f} "
                  f"norm_err {val['norm_err']:.4f} nll {val['nll']:.3f} disp {val['disp']:.3f} rank {val['eff_rank']:.1f}/{args.d} "
                  f"z std min {val['z_std_min']:.3f} | {rate:.1f} it/s, eta {(total - it - 1) / rate / 60:.1f} min", flush=True)
            (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    model.eval()
    del o, loss
    opt.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    info["residual"] = residual_stats(model, tr, tr_n, tr_t, va, va_n, va_t)
    info["wall_s"] = time.time() - t0
    info["final_val"] = val
    save(run_dir / "model.pt", total, val)
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=1))
    writer.close()
    print(f"[jepa] done in {info['wall_s'] / 60:.1f} min; {run_dir / 'model.pt'}  residual {info['residual']}", flush=True)


if __name__ == "__main__":
    main()
