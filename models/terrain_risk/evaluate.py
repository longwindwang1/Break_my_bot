"""Evaluate the terrain-risk model on held-out terrain (geo_test) and calibrate the slowdown trigger (W6).

  per-frame  AUROC / AP per horizon and policy, with env-cluster bootstrap CIs, against baselines:
             patch height range, steepest neighbour step (both from the same 17 x 17 patch), tile type x difficulty
             fall rate (from geo_train), and the proprioceptive world model's fall head
  lead time  "risk ahead" (max over a corridor along the command, from the egocentric window) vs the world-model
             fall head, same 5 s window before each fall and the same 5 % false-positive rate (frames with no fall
             within 3 s); lead = length of the above-threshold run that ends at the fall
  trigger    risk-ahead thresholds per policy at which 10 % / 20 % of frames trigger, on geo_train's held-out tiles
  heatmap    top-down risk of part of the test terrain with the falls on it

  ~/miniforge3/envs/env_isaaclab/bin/python -m models.terrain_risk.evaluate --run logs/terrain_risk/<run> \
      --train logs/wm_data/w6_geo_train --test logs/wm_data/w6_geo_test --out experiments/w6_terrain_risk \
      --wm_run logs/wm/2026-11-07_22-45-27_w4_wm_nominal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.world_model.evaluate import auroc, average_precision

from isaac.interface import load_interface

from .dataset import RiskFrames
from .model import RiskNet, load_risk_model
from .patches import HeightMap, cond_vector, load_risk_layout, patch, window
from .train import batch_inputs

_SLOW = load_interface()["terrain_risk"]["slowdown"]
CORRIDOR_LEN, CORRIDOR_HALF_WIDTH = _SLOW["corridor"]["length"], _SLOW["corridor"]["half_width"]


def corridor_mask(cmd: torch.Tensor, lay) -> torch.Tensor:
    """[B, ny_out, nx_out] cells within CORRIDOR_HALF_WIDTH of the segment from the base along the command
    direction, CORRIDOR_LEN long (cells around the base if the command is ~0)."""
    nxo, nyo = lay.nx - 2 * lay.half, lay.ny - 2 * lay.half
    dev = cmd.device
    cx = (lay.x0 + lay.res * (torch.arange(nxo, device=dev) + lay.half))[None, None, :]
    cy = (lay.y0 + lay.res * (torch.arange(nyo, device=dev) + lay.half))[None, :, None]
    v = cmd[:, :2]
    speed = v.norm(dim=1, keepdim=True)
    d = torch.where(speed > 0.1, v / speed.clamp(min=1e-6), torch.tensor([[1.0, 0.0]], device=dev))
    dx, dy = d[:, 0, None, None], d[:, 1, None, None]
    along = cx * dx + cy * dy
    across = (-cx * dy + cy * dx).abs()
    m = (along >= 0) & (along <= CORRIDOR_LEN) & (across <= CORRIDOR_HALF_WIDTH)
    still = (speed[:, 0] <= 0.1)[:, None, None] & ((cx**2 + cy**2) <= 0.4**2)
    return torch.where((speed[:, 0] > 0.1)[:, None, None], m, still)


@torch.no_grad()
def risk_ahead(model: RiskNet, hm: HeightMap, lay, x, y, yaw, cmd, policy, horizon: int = 0, chunk: int = 512) -> torch.Tensor:
    out = []
    for s in range(0, len(x), chunk):
        e = s + chunk
        r = model(window(hm, lay, x[s:e], y[s:e], yaw[s:e]), cond_vector(cmd[s:e], policy[s:e]))[:, horizon]
        m = corridor_mask(cmd[s:e], lay)
        out.append(torch.where(m, r, torch.zeros_like(r)).amax(dim=(1, 2)))
    return torch.cat(out)


def boot_auroc(score: np.ndarray, label: np.ndarray, cluster: np.ndarray, n: int = 200, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ids = np.unique(cluster)
    groups = {c: np.nonzero(cluster == c)[0] for c in ids}
    vals = []
    for _ in range(n):
        pick = np.concatenate([groups[c] for c in rng.choice(ids, len(ids))])
        vals.append(auroc(score[pick], label[pick]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


@torch.no_grad()
def baselines(fr: RiskFrames, hm: HeightMap, lay, idx: torch.Tensor) -> dict[str, torch.Tensor]:
    rng_, step_ = [], []
    for c in torch.split(idx, 8192):
        h = patch(hm, lay, fr.x[c], fr.y[c], fr.yaw[c])[:, 0]
        rng_.append(h.amax(dim=(1, 2)) - h.amin(dim=(1, 2)))
        dx = (h[:, :, 1:] - h[:, :, :-1]).abs().amax(dim=(1, 2))
        dy = (h[:, 1:, :] - h[:, :-1, :]).abs().amax(dim=(1, 2))
        step_.append(torch.maximum(dx, dy))
    return {"height_range": torch.cat(rng_), "max_step": torch.cat(step_)}


def tile_rate_table(train: RiskFrames, k: int) -> dict:
    """Fall rate per (tile type, difficulty decile) on the training frames (label k)."""
    kn = train.known[:, k].cpu().numpy()
    y = train.label[:, k].cpu().numpy()
    keys = train.tile_keys()
    table = {}
    for key in np.unique(keys[kn]):
        m = kn & (keys == key)
        table[key] = (y[m].sum() + 1) / (m.sum() + 2)
    return table


def tile_scores(fr: RiskFrames, table: dict, idx: torch.Tensor) -> np.ndarray:
    default = float(np.mean(list(table.values())))
    keys = fr.tile_keys()[idx.cpu().numpy()]
    return np.array([table.get(k, default) for k in keys])


@torch.no_grad()
def wm_fall_probs(wm_run: Path, split_dir: Path, device: str) -> np.ndarray:
    """World-model fall probabilities [N, T, 3] on a recorded split (proprioceptive baseline)."""
    from models.world_model.dataset import WMData
    from models.world_model.ensemble import load_model
    from models.world_model.evaluate import predict_split

    ckpt = wm_run / "model_fall.pt" if (wm_run / "model_fall.pt").exists() else wm_run / "model.pt"
    model, _ = load_model(ckpt, device)
    d = WMData(split_dir, device=device)
    P = predict_split(model, d)
    out = P["fall"].cpu().numpy()
    del d, P
    torch.cuda.empty_cache()
    return out


def lead_times(score_nt: np.ndarray, term: np.ndarray, es: np.ndarray, usable: np.ndarray, thr: float,
               window_steps: int = 250) -> np.ndarray:
    """For every fall (terminated at e with at least window_steps of episode before it): length of the
    above-threshold run of score_nt[n, e-window_steps+1 .. e] that ends at e (0 if not above at e)."""
    leads = []
    for n, e in zip(*np.nonzero(term.T)):  # term [T, N] -> iterate (n, e) via transpose
        if es[e, n] < window_steps or not usable[n]:
            continue
        s = score_nt[n, e - window_steps + 1 : e + 1] > thr
        run = 0
        for v in s[::-1]:
            if not v:
                break
            run += 1
        leads.append(run)
    return np.array(leads)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--train", required=True, nargs="+", help="training split dirs (tile table, trigger thresholds)")
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--wm_run", default=None, help="world-model run for the proprioceptive baseline")
    p.add_argument("--device", default="cuda")
    p.add_argument("--frame_stride", type=int, default=5, help="evaluate every n-th frame (neighbours are near-duplicates)")
    args = p.parse_args()
    dev = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lay = load_risk_layout()
    model, ckpt = load_risk_model(Path(args.run) / "model.pt", dev)
    results = {"run": args.run, "checkpoint_iter": ckpt["iter"], "platt": ckpt.get("platt"), "test": args.test, "train": args.train}

    # --- per-frame classification on held-out terrain -----------------------------------------------------
    te = RiskFrames(args.test, device=dev, val_frac=0.0)
    hm_te = te.maps[0]
    idx = torch.arange(0, te.n_frames, args.frame_stride, device=dev)
    probs = []
    with torch.no_grad():
        for c in torch.split(idx, 8192):
            h, cond = batch_inputs(te, lay, c)
            probs.append(model(h, cond)[:, :, 0, 0])
    probs = torch.cat(probs).cpu().numpy()
    base = {k: v.cpu().numpy() for k, v in baselines(te, hm_te, lay, idx).items()}
    tr = RiskFrames.merge(args.train, device=dev)
    wm = None
    if args.wm_run:
        wm_all = wm_fall_probs(Path(args.wm_run), Path(args.test), dev)  # [N, T, 3]; horizons 10, 25, 50
        wm = wm_all[te.env[idx].cpu().numpy(), te.step[idx].cpu().numpy()]
    cls = {}
    env = te.env[idx].cpu().numpy()
    pol = te.policy[idx].cpu().numpy()
    for k, hz in enumerate(lay.horizons):
        kn = te.known[idx, k].cpu().numpy()
        y = te.label[idx, k].cpu().numpy()
        table = tile_rate_table(tr, k)
        scores = {"risk_model": probs[:, k], **base, "tile_type_x_difficulty": tile_scores(te, table, idx)}
        if wm is not None:
            scores["world_model_fall_head"] = wm[:, {25: 1, 50: 2}[hz]]
        for pname, pm in (("all", np.ones_like(kn)), ("v1", pol == 0), ("v2", pol == 1)):
            m = kn & pm
            row = {"frames": int(m.sum()), "positives": int(y[m].sum())}
            for sname, s in scores.items():
                row[sname] = {"auroc": auroc(s[m], y[m]), "ap": average_precision(s[m], y[m])}
                if pname == "all" and sname in ("risk_model", "max_step", "tile_type_x_difficulty"):
                    row[sname]["auroc_ci95"] = boot_auroc(s[m], y[m], env[m])
            cls[f"h{hz}_{pname}"] = row
        print(f"[eval] h{hz}: " + "  ".join(f"{s} {cls[f'h{hz}_all'][s]['auroc']:.3f}" for s in scores), flush=True)
    results["classification"] = cls
    results["calibration_check"] = {f"h{hz}": {"mean_prob": float(probs[te.known[idx, k].cpu().numpy(), k].mean()),
                                               "base_rate": float(te.label[idx, k][te.known[idx, k]].float().mean())}
                                    for k, hz in enumerate(lay.horizons)}

    # --- trigger thresholds on geo_train held-out tiles --------------------------------------------------------
    va = tr.indices(True)
    va = va[:: max(1, args.frame_stride)]
    ra_parts, pol_parts = [], []
    for k, hm_k in enumerate(tr.maps):
        v = va[tr.map_id[va] == k]
        ra_parts.append(risk_ahead(model, hm_k, lay, tr.x[v], tr.y[v], tr.yaw[v], tr.cmd[v], tr.policy[v]).cpu().numpy())
        pol_parts.append(tr.policy[v].cpu().numpy())
    ra_va, pol_va = np.concatenate(ra_parts), np.concatenate(pol_parts)
    trig = {}
    for pi, pname in enumerate(("v1", "v2")):
        r = ra_va[pol_va == pi]
        trig[pname] = {"trigger_10pct": float(np.quantile(r, 0.90)), "trigger_20pct": float(np.quantile(r, 0.80))}
    results["trigger_thresholds"] = trig
    print(f"[eval] trigger thresholds (risk ahead, h25): {trig}", flush=True)

    # --- lead time on geo_test: risk ahead vs world-model fall head -------------------------------------------
    T = int(te.meta["steps_written"])
    d = Path(args.test)
    load = lambda n: np.load(d / f"{n}.npy", mmap_mode="r")[:T]  # noqa: E731
    term, trunc, es = np.asarray(load("terminated")), np.asarray(load("truncated")), np.asarray(load("episode_step"))
    pos, quat, prop = load("root_pos"), load("root_quat"), load("proprio")
    from models.world_model.dataset import quat_yaw

    N = term.shape[1]
    info = dict(np.load(d / "env_info.npz"))
    usable_env = info["noise_std"] == 0
    # risk ahead for every frame in the 5 s before each fall, plus a random sample of other frames
    want = np.zeros((T, N), bool)
    for e, n in zip(*np.nonzero(term)):
        want[max(0, e - 249) : e + 1, n] = True
    rng = np.random.default_rng(0)
    want |= rng.random((T, N)) < 0.02
    tt, nn = np.nonzero(want)
    yaw = quat_yaw(np.asarray(quat)[tt, nn].astype(np.float64)).astype(np.float32)
    tens = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=dev)  # noqa: E731
    ra = risk_ahead(model, hm_te, lay, tens(pos[tt, nn, 0]), tens(pos[tt, nn, 1]), tens(yaw), tens(prop[tt, nn, 9:12]),
                    tens(info["policy_id"][nn].astype(np.int64))).cpu().numpy()
    score_ra = np.full((N, T), -np.inf, np.float32)
    score_ra[nn, tt] = ra
    from models.world_model.dataset import fall_labels

    y3s, k3s = fall_labels(term.T, trunc.T, [150])
    neg = (k3s[..., 0] & ~y3s[..., 0] & want.T & (es.T >= 1))
    lead = {}
    series = {"risk_ahead": score_ra}
    if args.wm_run:
        series["world_model_fall_head"] = wm_all[..., 2]
    for name, sc in series.items():
        thr = float(np.quantile(sc[neg], 0.95))
        L = lead_times(sc, term, es, usable_env, thr)
        lead[name] = {"threshold": thr, "falls": int(len(L)), "warned_at_fall": float((L > 0).mean()),
                      "lead_ge_0.5s": float((L >= 25).mean()), "lead_ge_1s": float((L >= 50).mean()),
                      "lead_ge_2s": float((L >= 100).mean()), "median_lead_steps": float(np.median(L))}
    results["lead_time"] = lead
    print(f"[eval] lead time: {json.dumps(lead)}", flush=True)

    # --- heatmap of a test region ---------------------------------------------------------------------------
    heat_png(model, hm_te, lay, te, term, pos, out / "heatmap_test.png")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"[eval] wrote {out}/results.json", flush=True)


@torch.no_grad()
def heat_png(model, hm, lay, fr, term, pos, path, x0=-40.0, y0=-24.0, w=24.0, hgt=16.0):
    """Risk (h25) over a region for heading +x, command (0.6, 0, 0), policy v2, with recorded falls overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = lay.res
    nx, ny = int(w / res) + 16, int(hgt / res) + 16
    origin = torch.tensor([x0 - res * 8], device=hm.h.device), torch.tensor([y0 - res * 8], device=hm.h.device)
    from .patches import sample_grid

    h = sample_grid(hm, origin[0], origin[1], torch.zeros(1, device=hm.h.device), nx, ny, 0.0, 0.0, res)
    cond = cond_vector(torch.tensor([[0.6, 0.0, 0.0]], device=hm.h.device), torch.tensor([1], device=hm.h.device))
    risk = model(h, cond)[0, 0].cpu().numpy()  # [ny-16, nx-16]
    hv = h[0, 0, 8:-8, 8:-8].cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ext = [x0, x0 + w, y0, y0 + hgt]
    ax[0].imshow(hv, origin="lower", extent=ext, cmap="gray")
    ax[0].set_title("terrain height")
    im = ax[1].imshow(risk, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=max(0.05, float(np.quantile(risk, 0.995))))
    fig.colorbar(im, ax=ax[1], label="P(fall within 0.5 s), heading +x, v = 0.6 m/s, v2")
    e, n = np.nonzero(term)
    fx, fy = np.asarray(pos)[e, n, 0], np.asarray(pos)[e, n, 1]
    sel = (fx >= x0) & (fx <= x0 + w) & (fy >= y0) & (fy <= y0 + hgt)
    for a in ax:
        a.scatter(fx[sel], fy[sel], s=10, c="cyan", marker="x", label="recorded falls (any heading)")
        a.set_xlabel("x (m)")
        a.set_ylabel("y (m)")
    ax[1].legend(loc="upper right", fontsize=7)
    ax[1].set_title("terrain-risk model")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
