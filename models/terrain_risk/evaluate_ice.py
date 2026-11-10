"""Evaluate the ice-aware terrain-risk model (phase 2) on held-out terrain with visible and hidden ice.

  fall classes  every fall in the test split is classed by the ice under the feet in the second before it
                (foot_ice codes): visible ice / hidden ice / none (geometry); if both, the more recent one. A positive
                frame takes the class of the fall it precedes.
  per-frame     AUROC per fall class (that class's positives vs all negatives), h25 / h50, for the ice model as the
                game runs it (visible ice shown), with all ice hidden, with all ice shown (oracle), the phase-1
                height-only model and the world-model fall head; plus geo_test (no ice) against phase 1.
  lead time     per fall class, as in phase 1 (5 s window, 5 % FPR on frames with no fall within 3 s): risk ahead
                (ice model / phase-1 model) and the world-model fall head; and the world model's surprisal CUSUM
                (robot_interface.yaml world_model.surprisal): falls with an alarm within 0.5 / 1 / 2 s before, ice
                contacts (first foot on a patch after >= 1 s off ice) with an alarm within 1 s, alarms per 10 min
                off ice
  trigger       risk-ahead thresholds (10 % / 20 % of frames) on the training splits' held-out tiles
  heatmap       a test region: height with the ice, risk with visible ice shown, risk with all ice hidden; falls by class

  ~/miniforge3/envs/env_isaaclab/bin/python -m models.terrain_risk.evaluate_ice --run logs/terrain_risk/<ice run> \
      --phase1_run logs/terrain_risk/2026-11-08_01-11-04_w6_risk3 --test logs/wm_data/w6_ice_test \
      --geo_test logs/wm_data/w6_geo_test --train <6 training dirs> --wm_run logs/wm/<wm run> --out experiments/w6_ice_risk
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from isaac.interface import load_interface
from models.world_model.dataset import fall_labels, quat_yaw
from models.world_model.evaluate import auroc, average_precision

from .dataset import RiskFrames
from .evaluate import boot_auroc, risk_ahead, wm_fall_probs
from .model import load_risk_model
from .patches import HeightMap, cond_vector, ice_channel, load_risk_layout, sample_grid
from .train import batch_inputs

CLASSES = ("none", "visible", "hidden")
ICE_WINDOW = 50  # steps (1 s) before a fall in which a foot on ice makes it an ice fall


def fall_classes(foot_ice: np.ndarray, term: np.ndarray, es: np.ndarray, window: int = ICE_WINDOW) -> np.ndarray:
    """[T, N] int8: class of the fall at (e, n) (0 none / 1 visible / 2 hidden), -1 where no fall."""
    T, N = term.shape
    out = np.full((T, N), -1, np.int8)
    for e, n in zip(*np.nonzero(term)):
        win = foot_ice[e - min(int(es[e, n]), window) : e + 1, n]  # [w, 4]
        last = [np.nonzero((win == c).any(axis=1))[0] for c in (1, 2)]
        last = [int(v.max()) if len(v) else -1 for v in last]
        out[e, n] = 0 if max(last) < 0 else (1 if last[0] >= last[1] else 2)
    return out


def next_fall_class(cls_tn: np.ndarray, term: np.ndarray, trunc: np.ndarray) -> np.ndarray:
    """[T, N]: class of the next fall in the same episode at or after step t (-1 if the episode times out)."""
    T, N = term.shape
    out = np.full((T, N), -1, np.int8)
    cur = np.full(N, -1, np.int8)
    for t in range(T - 1, -1, -1):
        cur = np.where(term[t], cls_tn[t], np.where(trunc[t], -1, cur))
        out[t] = cur
    return out


def ice_alarm_metrics(al: np.ndarray, term: np.ndarray, trunc: np.ndarray, es: np.ndarray, foot_ice: np.ndarray,
                      details: bool = False) -> dict:
    """Surprisal alarms [N, T] (bool) on an ice split ([T, N] term / trunc / episode_step, foot_ice [T, N, 4]):
    falls (after >= 5 s in the episode) with an alarm within 0.5 / 1 / 2 s before, per fall class; ice contacts
    (first foot on a patch after >= 1 s off ice) with an alarm within 1 s; alarms per 10 min off ice, not within
    3 s before a fall. details: also per-fall arrays (env, step, class, hit within 1 s) for paired comparisons."""
    T, N = term.shape
    cls_tn = fall_classes(foot_ice, term, es)
    falls = [(n, e) for n, e in zip(*np.nonzero(term.T)) if es[e, n] >= 250]
    fc = np.array([cls_tn[e, n] for n, e in falls])
    y3s, k3s = fall_labels(term.T, trunc.T, [150])
    cs = np.concatenate([np.zeros((N, 1), np.int64), np.cumsum(al, axis=1)], axis=1)  # alarms in [a, b) = cs[b] - cs[a]
    wm_res = {}
    for wname, W in (("0.5s", 25), ("1s", 50), ("2s", 100)):
        hit = np.array([cs[n, e + 1] - cs[n, max(0, e + 1 - W)] > 0 for n, e in falls], bool)
        wm_res[f"falls_with_alarm_within_{wname}"] = {c: float(hit[fc == ci].mean()) if (fc == ci).any() else None
                                                      for ci, c in enumerate(CLASSES)}
    # ice contacts: first step with a foot on a patch after >= 1 s off ice, within one episode
    on = foot_ice.max(axis=2)  # [T, N]; 2 if any foot on hidden ice
    start = np.zeros((T, N), bool)  # first foot on a patch after >= 1 s off ice, within the episode
    off = np.full(N, 10**6)
    for t in range(T):
        off = np.where(es[t] == 0, 10**6, off)
        start[t] = (on[t] > 0) & (off >= ICE_WINDOW) & (es[t] >= 25)
        off = np.where(on[t] > 0, 0, off + 1)
    start[T - ICE_WINDOW :] = False
    for c in (1, 2):
        tt_, nn_ = np.nonzero(start & (on == c))
        hit = cs[nn_, tt_ + ICE_WINDOW] - cs[nn_, tt_] > 0
        wm_res[f"{CLASSES[c]}_ice_contacts"] = {"count": int(len(tt_)), "alarm_within_1s": float(hit.mean()) if len(hit) else None}
    near_ice = np.zeros((T, N), bool)  # on ice now or within the last second
    last_on = np.full(N, -10**6)
    for t in range(T):
        last_on = np.where(on[t] > 0, t, np.where(es[t] == 0, -10**6, last_on))
        near_ice[t] = t - last_on <= ICE_WINDOW
    quiet = ~near_ice.T & ~(y3s[..., 0] & k3s[..., 0]) & (es.T >= 25)
    wm_res["alarms_per_10min_off_ice_no_fall"] = float(al[quiet].sum() / quiet.sum() * 30000)
    if details:
        wm_res["_falls"] = {"env": np.array([n for n, _ in falls]), "step": np.array([e for _, e in falls]), "cls": fc,
                            "hit_1s": np.array([cs[n, e + 1] - cs[n, max(0, e + 1 - 50)] > 0 for n, e in falls], bool)}
    return wm_res


def lead_runs(score_nt: np.ndarray, falls: list[tuple[int, int]], thr: float) -> np.ndarray:
    leads = []
    for n, e in falls:
        s = score_nt[n, e - 249 : e + 1] > thr
        run = 0
        for v in s[::-1]:
            if not v:
                break
            run += 1
        leads.append(run)
    return np.array(leads)


@torch.no_grad()
def frame_probs(model, fr: RiskFrames, lay, idx: torch.Tensor, ice_mode: str) -> np.ndarray:
    out = []
    for c in torch.split(idx, 8192):
        h, ice, cond = batch_inputs(fr, lay, c, ice_mode=ice_mode)
        out.append(model(h, cond, ice)[:, :, 0, 0])
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def wm_alarms(wm_run: Path, split_dir: Path, device: str) -> np.ndarray:
    """World-model surprisal CUSUM alarms [N, T] with the deployed parameters (robot_interface.yaml)."""
    from models.world_model.dataset import WMData
    from models.world_model.ensemble import load_model
    from models.world_model.evaluate import Scorer, predict_split
    from models.world_model.surprisal import cusum, spike_stat

    sp = load_interface()["world_model"]["surprisal"]
    ckpt = wm_run / "model_fall.pt" if (wm_run / "model_fall.pt").exists() else wm_run / "model.pt"
    model, _ = load_model(ckpt, device)
    d = WMData(split_dir, device=device, extra=True)
    P = predict_split(model, d)
    sc = Scorer(d, P, torch.tensor(sp["var_floor"], device=device), sp["warmup_steps"])
    S = sc.raw(torch.tensor(sp["weights"], device=device), sp["clip_z2"], sp["statistic"])
    zs = spike_stat(sc.z)
    alarm = torch.zeros_like(sc.scored)
    for pi, pname in enumerate(load_interface()["world_model"]["policies"]):
        pp = sp["per_policy"][pname]
        sel = d.policy_id == pi
        x = (S[sel] - pp["m0"]) / pp["s0"]
        alarm[sel], _ = cusum(x, sc.scored[sel], sc.reset[sel], pp["k"], pp["h"], zs[sel] > pp["spike_z"])
    out = alarm.cpu().numpy()
    del d, P, sc, S
    torch.cuda.empty_cache()
    return out


def trigger_thresholds(model, lay, train_dirs, dev, stride: int) -> dict:
    """Risk-ahead values exceeded on 10 % / 20 % of the training splits' held-out-tile frames, per policy:
    "trigger_thresholds" for the total risk (all splits) and "trigger_thresholds_ice_excess" for the risk the ice
    adds (ice splits only; zero without ice)."""
    tr = RiskFrames.merge(train_dirs, device=dev)
    va = tr.indices(True)[:: max(1, stride)]
    parts = {"total": [], "excess": [], "pol": [], "pol_ice": []}
    for k, (hm_k, im_k) in enumerate(zip(tr.maps, tr.ice_maps)):
        v = va[tr.map_id[va] == k]
        a = (hm_k, lay, tr.x[v], tr.y[v], tr.yaw[v], tr.cmd[v], tr.policy[v])
        parts["total"].append(risk_ahead(model, *a, ice_map=im_k, ice_mode="map").cpu().numpy())
        parts["pol"].append(tr.policy[v].cpu().numpy())
        if im_k.h.numel() > 1:  # an ice split
            parts["excess"].append(risk_ahead(model, *a, ice_map=im_k, ice_mode="map", ice_excess=True).cpu().numpy())
            parts["pol_ice"].append(tr.policy[v].cpu().numpy())
    out = {}
    for key, val, pol in (("trigger_thresholds", "total", "pol"), ("trigger_thresholds_ice_excess", "excess", "pol_ice")):
        r, pp = np.concatenate(parts[val]), np.concatenate(parts[pol])
        out[key] = {pname: {"trigger_10pct": float(np.quantile(r[pp == pi], 0.90)), "trigger_20pct": float(np.quantile(r[pp == pi], 0.80))}
                    for pi, pname in enumerate(("v1", "v2"))}
    del tr
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--phase1_run", required=True)
    p.add_argument("--test", required=True, help="ice_test split dir")
    p.add_argument("--geo_test", default=None)
    p.add_argument("--train", nargs="+", required=True, help="training split dirs (trigger thresholds)")
    p.add_argument("--wm_run", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--frame_stride", type=int, default=5)
    p.add_argument("--heatmap_only", action="store_true", help="only redraw heatmap_ice_test.png")
    p.add_argument("--thresholds_only", action="store_true", help="only (re)compute the trigger thresholds into results.json")
    args = p.parse_args()
    dev = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lay = load_risk_layout()
    model, ckpt = load_risk_model(Path(args.run) / "model.pt", dev)
    p1, _ = load_risk_model(Path(args.phase1_run) / "model.pt", dev)
    assert model.has_ice and not p1.has_ice
    results = {"run": args.run, "checkpoint_iter": ckpt["iter"], "platt": ckpt.get("platt"), "phase1_run": args.phase1_run,
               "test": args.test, "train": args.train}

    if args.thresholds_only:
        old = json.loads((out / "results.json").read_text())
        old.update(trigger_thresholds(model, lay, args.train, dev, args.frame_stride))
        (out / "results.json").write_text(json.dumps(old, indent=1, default=float))
        print(f"[ice] trigger thresholds: {json.dumps({k: old[k] for k in ('trigger_thresholds', 'trigger_thresholds_ice_excess')})}", flush=True)
        return

    # --- raw test arrays and fall classes -------------------------------------------------------------------
    d = Path(args.test)
    te = RiskFrames(d, device=dev, val_frac=0.0)
    T = int(te.meta["steps_written"])
    load = lambda n: np.asarray(np.load(d / f"{n}.npy", mmap_mode="r")[:T])  # noqa: E731
    term, trunc, es, foot_ice = load("terminated"), load("truncated"), load("episode_step"), load("foot_ice")
    N = term.shape[1]
    cls_tn = fall_classes(foot_ice, term, es)
    nxt = next_fall_class(cls_tn, term, trunc)
    info = dict(np.load(d / "env_info.npz"))
    if args.heatmap_only:
        heat_png(model, te.maps[0], te.ice_maps[0], lay, load("root_pos"), term, cls_tn, info["policy_id"], out / "heatmap_ice_test.png")
        return
    counts = {c: int((cls_tn == k).sum()) for k, c in enumerate(CLASSES)}
    for pi, pname in enumerate(("v1", "v2")):
        m = info["policy_id"] == pi
        counts[pname] = {c: int((cls_tn[:, m] == k).sum()) for k, c in enumerate(CLASSES)}
    results["falls"] = counts
    results["ice_map"] = te.meta.get("ice_map")
    print(f"[ice] falls by class {counts}", flush=True)

    # --- per-frame -------------------------------------------------------------------------------------------
    idx = torch.arange(0, te.n_frames, args.frame_stride, device=dev)
    scores = {f"ice_model_{m}": frame_probs(model, te, lay, idx, m) for m in ("map", "none", "all")}
    scores["phase1_model"] = frame_probs(p1, te, lay, idx, "none")
    wm_all = None
    if args.wm_run:
        wm_all = wm_fall_probs(Path(args.wm_run), d, dev)  # [N, T, 3]; horizons 10, 25, 50
    e_np, s_np = te.env[idx].cpu().numpy(), te.step[idx].cpu().numpy()
    fcls = nxt[s_np, e_np]
    pol = te.policy[idx].cpu().numpy()
    per = {}
    for k, hz in enumerate(lay.horizons):
        kn = te.known[idx, k].cpu().numpy()
        y = te.label[idx, k].cpu().numpy()
        sc = {n_: v[:, k] for n_, v in scores.items()}
        if wm_all is not None:
            sc["world_model_fall_head"] = wm_all[e_np, s_np, {25: 1, 50: 2}[hz]]
        for pname, pm in (("all", np.ones_like(kn)), ("v1", pol == 0), ("v2", pol == 1)):
            for ci, cname in [(-9, "any"), *enumerate(CLASSES)]:
                pos = y & ((fcls == ci) if ci >= 0 else True)
                m = kn & pm & (~y | pos)
                row = {"frames": int(m.sum()), "positives": int(pos[m].sum())}
                for sname, s in sc.items():
                    row[sname] = {"auroc": auroc(s[m], y[m]), "ap": average_precision(s[m], y[m])}
                    if pname == "all" and sname in ("ice_model_map", "phase1_model") and row["positives"] >= 20:
                        row[sname]["auroc_ci95"] = boot_auroc(s[m], y[m], e_np[m], n=100)
                per[f"h{hz}_{pname}_{cname}"] = row
            if pname == "all":
                print(f"[ice] h{hz} AUROC by fall class: " + " | ".join(
                    f"{c}: " + " ".join(f"{s_} {per[f'h{hz}_all_{c}'][s_]['auroc']:.3f}" for s_ in sc)
                    for c in ("any", *CLASSES)), flush=True)
    results["per_frame"] = per

    if args.geo_test:
        g = RiskFrames(args.geo_test, device=dev, val_frac=0.0)
        gi = torch.arange(0, g.n_frames, args.frame_stride, device=dev)
        geo = {}
        for name, mdl in (("ice_model", model), ("phase1_model", p1)):
            pr = frame_probs(mdl, g, lay, gi, "map")
            for k, hz in enumerate(lay.horizons):
                kn = g.known[gi, k].cpu().numpy()
                y = g.label[gi, k].cpu().numpy()
                geo[f"h{hz}_{name}"] = {"auroc": auroc(pr[kn, k], y[kn]), "ap": average_precision(pr[kn, k], y[kn]),
                                        "auroc_ci95": boot_auroc(pr[kn, k], y[kn], g.env[gi].cpu().numpy()[kn], n=100)}
        results["geo_test"] = geo
        print(f"[ice] geo_test (no ice): {json.dumps(geo)}", flush=True)
        del g
        torch.cuda.empty_cache()

    # --- lead time per fall class ------------------------------------------------------------------------------
    pos, quat, prop = load("root_pos"), load("root_quat"), load("proprio")
    want = np.zeros((T, N), bool)
    for e, n in zip(*np.nonzero(term)):
        want[max(0, e - 249) : e + 1, n] = True
    want |= np.random.default_rng(0).random((T, N)) < 0.02
    tt, nn = np.nonzero(want)
    yaw = quat_yaw(quat[tt, nn].astype(np.float64)).astype(np.float32)
    tens = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=dev)  # noqa: E731
    args_ra = (tens(pos[tt, nn, 0]), tens(pos[tt, nn, 1]), tens(yaw), tens(prop[tt, nn, 9:12]), tens(info["policy_id"][nn].astype(np.int64)))
    series = {}
    for name, mdl, mode in (("risk_ahead_ice_model", model, "map"), ("risk_ahead_ice_model_all_hidden", model, "none"),
                            ("risk_ahead_phase1", p1, "none")):
        ra = risk_ahead(mdl, te.maps[0], lay, *args_ra, ice_map=te.ice_maps[0], ice_mode=mode).cpu().numpy()
        sc_ = np.full((N, T), -np.inf, np.float32)
        sc_[nn, tt] = ra
        series[name] = sc_
    if wm_all is not None:
        series["world_model_fall_head"] = wm_all[..., 2]
    y3s, k3s = fall_labels(term.T, trunc.T, [150])
    neg = k3s[..., 0] & ~y3s[..., 0] & want.T & (es.T >= 1)
    falls = [(n, e) for n, e in zip(*np.nonzero(term.T)) if es[e, n] >= 250]
    fc = np.array([cls_tn[e, n] for n, e in falls])
    lead = {}
    for name, sc_ in series.items():
        if not neg.any() or not falls:  # short (smoke) splits: no 3 s negatives / no falls after 5 s
            break
        thr = float(np.quantile(sc_[neg], 0.95))
        L = lead_runs(sc_, falls, thr)
        lead[name] = {"threshold": thr}
        for ci, cname in [(-9, "any"), *enumerate(CLASSES)]:
            sel = (fc == ci) if ci >= 0 else np.ones(len(fc), bool)
            Ls = L[sel]
            lead[name][cname] = {"falls": int(sel.sum()), "warned_at_fall": float((Ls > 0).mean()) if len(Ls) else None,
                                 "lead_ge_0.5s": float((Ls >= 25).mean()) if len(Ls) else None,
                                 "lead_ge_1s": float((Ls >= 50).mean()) if len(Ls) else None}
    print(f"[ice] lead time: {json.dumps(lead)}", flush=True)
    results["lead_time"] = lead

    if args.wm_run:
        al = wm_alarms(Path(args.wm_run), d, dev)  # [N, T]
        wm_res = ice_alarm_metrics(al, term, trunc, es, foot_ice)
        results["world_model_surprisal"] = wm_res
        print(f"[ice] world-model surprisal: {json.dumps(wm_res)}", flush=True)

    # --- trigger thresholds on the training splits' held-out tiles --------------------------------------------
    results.update(trigger_thresholds(model, lay, args.train, dev, args.frame_stride))
    print(f"[ice] trigger thresholds: {results['trigger_thresholds']} ice excess {results['trigger_thresholds_ice_excess']}", flush=True)

    heat_png(model, te.maps[0], te.ice_maps[0], lay, pos, term, cls_tn, info["policy_id"], out / "heatmap_ice_test.png")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"[ice] wrote {out}/results.json", flush=True)


@torch.no_grad()
def heat_png(model, hm: HeightMap, im: HeightMap, lay, pos, term, cls_tn, policy_id, path, w=24.0, hgt=16.0):
    """The 24 x 16 m tile-aligned region with the most ice falls. One row per policy: height with the ice and that
    policy's falls by class, risk with the visible ice shown (as in the game), risk with all ice hidden (heading +x,
    v = 0.6 m/s; colour scale per row)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e, n = np.nonzero(term)
    fx, fy, fc, fp = np.asarray(pos)[e, n, 0], np.asarray(pos)[e, n, 1], cls_tn[e, n], policy_id[n]
    best, x0, y0 = -1, -40.0, -24.0
    for cx in np.arange(-40.0, 40.0 - w + 1, 8.0):
        for cy in np.arange(-80.0, 80.0 - hgt + 1, 8.0):
            k = int(((fx >= cx) & (fx <= cx + w) & (fy >= cy) & (fy <= cy + hgt) & (fc > 0)).sum())
            if k > best:
                best, x0, y0 = k, float(cx), float(cy)
    res, dev = lay.res, hm.h.device
    nx, ny = int(w / res) + 16, int(hgt / res) + 16
    ox, oy, z = torch.tensor([x0 - res * 8], device=dev), torch.tensor([y0 - res * 8], device=dev), torch.zeros(1, device=dev)
    h = sample_grid(hm, ox, oy, z, nx, ny, 0.0, 0.0, res)
    h = h - h.mean()
    codes = sample_grid(im, ox, oy, z, nx, ny, 0.0, 0.0, res)
    hv, cv = h[0, 0, 8:-8, 8:-8].cpu().numpy(), codes[0, 0, 8:-8, 8:-8].cpu().numpy()
    ext = [x0, x0 + w, y0, y0 + hgt]
    sel = (fx >= x0) & (fx <= x0 + w) & (fy >= y0) & (fy <= y0 + hgt)
    ice_rgba = np.zeros((*cv.shape, 4))
    ice_rgba[cv == 1] = (0.2, 0.6, 1.0, 0.55)
    ice_rgba[cv == 2] = (1.0, 0.3, 0.3, 0.55)
    fig, ax = plt.subplots(2, 3, figsize=(19, 9.6))
    for row, (pi, pname) in enumerate(((1, "v2"), (0, "v1"))):
        cond = cond_vector(torch.tensor([[0.6, 0.0, 0.0]], device=dev), torch.tensor([pi], device=dev))
        r_vis = model(h, cond, ice_channel(codes, "map"))[0, 0].cpu().numpy()
        r_none = model(h, cond, ice_channel(codes, "none"))[0, 0].cpu().numpy()
        vmax = max(0.01, float(np.quantile(np.concatenate([r_vis.ravel(), r_none.ravel()]), 0.995)))
        a = ax[row]
        a[0].imshow(hv, origin="lower", extent=ext, cmap="gray")
        a[0].imshow(ice_rgba, origin="lower", extent=ext)
        a[0].set_title(f"{pname}: height; ice blue = visible, red = hidden; {pname} falls")
        for ax_, r, title in ((a[1], r_vis, "risk, visible ice shown (as in the game)"), (a[2], r_none, "risk, all ice hidden")):
            im_ = ax_.imshow(r, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=vmax)
            ax_.set_title(f"{pname}: {title}")
        fig.colorbar(im_, ax=list(a[1:]), label=f"P(fall within 0.5 s), heading +x, 0.6 m/s, {pname}", fraction=0.025)
        for ax_ in a:
            for ci, (c, mk) in enumerate((("white", "x"), ("deepskyblue", "o"), ("red", "o"))):
                s_ = sel & (fc == ci) & (fp == pi)
                ax_.scatter(fx[s_], fy[s_], s=16, c=c, marker=mk, edgecolors="k", linewidths=0.4,
                            label=f"fall on {CLASSES[ci]} ice" if ci else "fall, no ice")
            ax_.set_xlabel("x (m)")
            ax_.set_ylabel("y (m)")
        a[1].legend(loc="upper right", fontsize=7)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
