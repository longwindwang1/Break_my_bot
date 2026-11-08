"""Evaluate a trained world model and calibrate surprisal + CUSUM (W5).

Sections (all written to <out>/results.json, figures to <out>/*.png):
  one_step     calibration (z std, 1/2-sigma coverage), RMSE and NLL per state term; by policy, friction, terrain
  multi_step   open-loop rollouts with recorded actions and recorded scans (optimistic: the policy's real
               reactions are replayed), state RMSE and base position error at 1..50 steps
  fall         AUROC / AP of the fall heads vs a tilt baseline, per policy; warning lead time before falls
  surprisal    grid over statistic x weighting x clip x k; h from a false-alarm target on half of the clean envs,
               checked on the other half; detection rate and delay per event type; deceived pits
The chosen configuration is written as a robot_interface.yaml snippet to <out>/calibration.yaml.

  ~/miniforge3/envs/env_isaaclab/bin/python -m models.world_model.evaluate --run logs/wm/<run> \
      --data logs/wm_data --out experiments/w5_surprisal
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from isaac.interface import load_interface

from .dataset import WMData, val_envs
from .ensemble import WorldModel, load_model
from .surprisal import SPIKE_DIMS, cusum, cusum_alarm_counts, score, spike_stat, state_term_slices, weights, zscores

KEY_EVENTS = ["mu_0.1", "mu_0.05", "foot_0.05", "push_0.5", "push_1.0"]


# --- helpers ------------------------------------------------------------------------------------------
def auroc(score_: np.ndarray, label: np.ndarray) -> float:
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score_, kind="mergesort")
    s = score_[order]
    ranks = np.empty(len(s))
    _, start, counts = np.unique(s, return_index=True, return_counts=True)
    avg = start + (counts - 1) / 2.0 + 1
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(score_: np.ndarray, label: np.ndarray) -> float:
    if label.sum() == 0:
        return float("nan")
    order = np.argsort(-score_, kind="mergesort")
    lab = label[order]
    tp = np.cumsum(lab)
    precision = tp / np.arange(1, len(lab) + 1)
    return float(precision[lab].mean())


@torch.no_grad()
def predict_split(model: WorldModel, d: WMData) -> dict[str, torch.Tensor]:
    """Ensemble prediction at every (env, step): delta_mean/var [N, T, out_dim], disagreement [N, T], fall [N, T, 3]."""
    N, T = d.N, d.T
    S = d.layout.state_dim
    n = torch.arange(N, device=d.frames.device)
    out = {"delta_mean": torch.empty(N, T, model.out_dim, device=n.device),
           "delta_var": torch.empty(N, T, model.out_dim, device=n.device),
           "disagreement": torch.empty(N, T, device=n.device),
           "fall": torch.empty(N, T, len(d.layout.fall_horizons), device=n.device)}
    for t in range(T):
        p = model.predictive(d.inputs(n, torch.full_like(n, t)))
        out["delta_mean"][:, t] = p["delta_mean"]
        out["delta_var"][:, t] = p["delta_var"]
        out["disagreement"][:, t] = (p["epistemic"][:, :S] / model.out_std[:S] ** 2).mean(1)
        out["fall"][:, t] = p["fall_prob"]
    return out


def true_delta(d: WMData) -> torch.Tensor:
    s = d.frames[..., torch.as_tensor(d.layout.state_in_frame, device=d.frames.device)]
    out = torch.zeros_like(s)
    out[:, :-1] = s[:, 1:] - s[:, :-1]
    return out


def tile_names(d: WMData) -> np.ndarray:
    ter = d.meta["terrain"]
    names = np.array(["border"] + [""] * (ter["num_rows"] * ter["num_cols"]), dtype=object)
    for r, c, name, _ in ter["tiles"]:
        names[1 + r * ter["num_cols"] + c] = name
    return names[d.tile + 1]  # tile -1 -> "border"


# --- 1. one-step calibration --------------------------------------------------------------------------
def one_step_stats(y: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, model: WorldModel) -> dict:
    """y, mu, var: [M, S] selected transitions (raw units)."""
    S = y.shape[1]
    z = (y - mu) / var.sqrt()
    res = {"count": int(len(y))}
    for name, sl in state_term_slices() + [("all", slice(0, S))]:
        zz, err, v = z[:, sl], (y - mu)[:, sl], var[:, sl]
        res[name] = {
            "rmse": float(err.pow(2).mean().sqrt()),
            "rmse_zero_delta": float(y[:, sl].pow(2).mean().sqrt()),
            "z_std": float(zz.std()),
            "cov_1sigma": float((zz.abs() < 1).float().mean()),
            "cov_2sigma": float((zz.abs() < 2).float().mean()),
            "nll_norm": float((0.5 * (zz**2 + torch.log(2 * np.pi * v / model.out_std[sl] ** 2))).mean()),
            "sigma_over_delta_std": float((v.sqrt() / model.out_std[sl]).flatten()[:4_000_000].median()),
        }
    return res


def one_step_breakdown(d: WMData, P: dict, base_mask: torch.Tensor, model: WorldModel) -> dict:
    """1-step statistics over base_mask, overall and by policy, friction and terrain type."""
    S = d.layout.state_dim
    m = base_mask
    y = true_delta(d)[m]
    mu = P["delta_mean"][..., :S][m]
    var = P["delta_var"][..., :S][m]
    groups = {"all": None}
    pid = d.policy_id[:, None].expand(-1, d.T)[m]
    for i, name in enumerate(d.layout.policies):
        groups[f"policy_{name}"] = pid == i
    mu_min = torch.from_numpy(d.foot_mu.min(-1)).to(m.device)[m]
    groups["friction_ice(<0.3)"] = mu_min < 0.3
    groups["friction_normal(>=0.6)"] = mu_min >= 0.6
    names = tile_names(d)[m.cpu().numpy()]
    for tname in sorted(set(names.tolist()) - {"border", ""}):
        groups[f"terrain_{tname}"] = torch.from_numpy(names == tname).to(m.device)
    out = {}
    for g, sel in groups.items():
        if sel is None:
            out[g] = one_step_stats(y, mu, var, model)
        elif sel.any():
            out[g] = one_step_stats(y[sel], mu[sel], var[sel], model)
    return out


# --- 2. open-loop multi-step ----------------------------------------------------------------------------
@torch.no_grad()
def multi_step(model: WorldModel, d: WMData, horizons=(1, 5, 10, 25, 50), windows: int = 4096, seed: int = 0) -> dict:
    lay = d.layout
    S, K = lay.state_dim, max(horizons)
    ok = d.train_ok.int()
    run = torch.zeros_like(ok)  # consecutive train_ok steps starting at t
    for t in range(d.T - 1, -1, -1):
        run[:, t] = ok[:, t] * (1 + (run[:, t + 1] if t + 1 < d.T else 0))
    n, t = (run >= K).nonzero(as_tuple=True)
    g = torch.Generator(device=n.device).manual_seed(seed)
    sel = torch.randperm(len(n), generator=g, device=n.device)[:windows]
    n, t = n[sel], t[sel]
    state_idx = torch.as_tensor(lay.state_in_frame, device=n.device)
    hist = d.history_frames(n, t)
    oh = d.onehot(n)
    s = hist[:, -1][:, state_idx]
    s0 = s.clone()
    pos, yaw = torch.zeros(len(n), 3, device=n.device), torch.zeros(len(n), device=n.device)
    tpos, tyaw = torch.zeros_like(pos), torch.zeros_like(yaw)
    res = {h: {} for h in horizons}
    for k in range(K):
        tk = t + k
        a = d.action[n, tk]
        p = model.predictive(d.assemble(hist, a, d.scan[n, tk], oh))
        s = s + p["delta_mean"][:, :S]
        dd, td = p["delta_mean"][:, S:], d.disp[n, tk]
        for (P_, Y_, D_) in ((pos, yaw, dd), (tpos, tyaw, td)):
            c, si = torch.cos(Y_), torch.sin(Y_)
            P_[:, 0] += c * D_[:, 0] - si * D_[:, 1]
            P_[:, 1] += si * D_[:, 0] + c * D_[:, 1]
            P_[:, 2] += D_[:, 2]
            Y_ += D_[:, 3]
        frame = torch.empty_like(hist[:, -1])
        frame[:, state_idx] = s
        frame[:, S:] = a
        hist = torch.cat([hist[:, 1:], frame[:, None]], dim=1)
        if k + 1 in res:
            s_true = d.state(n, tk + 1)
            r = res[k + 1]
            for name, sl in state_term_slices():
                r[f"{name}_rmse"] = float((s - s_true)[:, sl].pow(2).mean().sqrt())
                r[f"{name}_rmse_hold"] = float((s0 - s_true)[:, sl].pow(2).mean().sqrt())  # baseline: s stays s_t
            r["pos_xy_err_m"] = float((pos[:, :2] - tpos[:, :2]).norm(dim=1).mean())
            r["pos_xy_err_m_p90"] = float((pos[:, :2] - tpos[:, :2]).norm(dim=1).quantile(0.9))
            r["true_xy_dist_m"] = float(tpos[:, :2].norm(dim=1).mean())
            r["z_err_m"] = float((pos[:, 2] - tpos[:, 2]).abs().mean())
            r["yaw_err_rad"] = float((yaw - tyaw).abs().mean())
    return {"windows": len(n), "horizons": {str(h): v for h, v in res.items()}}


# --- 3. fall prediction ----------------------------------------------------------------------------------
def fall_eval(d: WMData, P: dict) -> dict:
    lay = d.layout
    known = d.fall_known & (d.episode_step >= 1)[..., None]
    tilt = d.frames[..., 6:8].norm(dim=-1)  # |projected gravity xy|; frame[6:9] = projected_gravity
    assert d.layout.state_in_frame[6] == 6
    out = {}
    pid = d.policy_id[:, None].expand(-1, d.T)
    for hi, h in enumerate(lay.fall_horizons):
        for pi, pname in [(-1, "all")] + list(enumerate(lay.policies)):
            m = known[..., hi] & ((pid == pi) if pi >= 0 else torch.ones_like(pid, dtype=torch.bool))
            y = d.fall_y[..., hi][m].cpu().numpy()
            p = P["fall"][..., hi][m].cpu().numpy()
            b = tilt[m].cpu().numpy()
            out[f"h{h}_{pname}"] = {"positives": int(y.sum()), "count": int(len(y)),
                                    "auroc": auroc(p, y), "ap": average_precision(p, y),
                                    "auroc_tilt": auroc(b, y), "ap_tilt": average_precision(b, y),
                                    "base_rate": float(y.mean()) if len(y) else float("nan")}
    # lead time with the longest horizon, threshold at 1 % false positives
    hi = len(lay.fall_horizons) - 1
    H = lay.fall_horizons[hi]
    neg = known[..., hi] & ~d.fall_y[..., hi]
    thr = float(torch.quantile(P["fall"][..., hi][neg][:2_000_000].float(), 0.99))
    leads = []
    fn, fe = d.terminated.nonzero(as_tuple=True)
    for n_, e in zip(fn.tolist(), fe.tolist()):
        if d.episode_step[n_, e] < H or d.noise_std[n_] > 0:
            continue
        w = P["fall"][n_, e - H + 1 : e + 1, hi] > thr
        leads.append(int(H - w.float().argmax()) if w.any() else 0)
    leads = np.array(leads)
    out["lead_time"] = {"threshold_p": thr, "falls": int(len(leads)), "warned": float((leads > 0).mean()) if len(leads) else float("nan"),
                        "warned_ge_10_steps": float((leads >= 10).mean()) if len(leads) else float("nan"),
                        "median_lead_steps_if_warned": float(np.median(leads[leads > 0])) if (leads > 0).any() else float("nan")}
    return out


# --- 4. surprisal + CUSUM ------------------------------------------------------------------------------
class Scorer:
    """Per-frame standardized-score inputs for one split: z [N, T, S] (frame u vs the prediction from u-1)."""

    def __init__(self, d: WMData, P: dict, var_floor: torch.Tensor, warmup: int):
        S = d.layout.state_dim
        s = d.frames[..., torch.as_tensor(d.layout.state_in_frame, device=d.frames.device)]
        mu = torch.zeros_like(s)
        var = torch.ones_like(s)
        mu[:, 1:] = s[:, :-1] + P["delta_mean"][:, :-1, :S]
        var[:, 1:] = P["delta_var"][:, :-1, :S]
        self.z = zscores(s, mu, var, var_floor)
        self.var = var
        self.var_floor = var_floor
        scored = torch.zeros_like(d.trans_ok)
        scored[:, 1:] = d.trans_ok[:, :-1]
        self.scored = scored & (d.episode_step >= warmup)
        self.reset = d.episode_step == 0
        self.prefall = d.fall_y[..., -1]  # an alarm within fall_horizons[-1] steps before a fall is not false
        self.policy_id = d.policy_id

    def raw(self, w: torch.Tensor, clip: float, stat: str) -> torch.Tensor:
        return score(self.z, self.var, self.var_floor, w, clip, stat)


def calibrate(clean: Scorer, events: Scorer, ev_data: WMData, pits: Scorer | None, pits_data: WMData | None,
              target_far: float, policies: list[str], event_step: int, grid: dict) -> dict:
    """Per policy: spike threshold at half the false-alarm budget (clean half A), then for every grid
    configuration the smallest CUSUM h whose combined false-alarm rate (with the spike detector) meets the
    budget on half A; verified on half B; detection on the events and pits splits."""
    dev = clean.z.device
    hs = torch.logspace(np.log10(0.2), np.log10(400.0), 160, device=dev)
    N = clean.z.shape[0]
    half_a = torch.arange(N, device=dev) % 2 == 0
    ev_names = ev_data.meta["events"]
    ev_type = torch.from_numpy(ev_data.info["event_type"].astype(np.int64)).to(dev)
    reset_before = torch.from_numpy(ev_data.info["reset_before_event"]).to(dev)
    E = event_step
    sp_clean, sp_ev = spike_stat(clean.z), spike_stat(events.z)
    sp_pits = spike_stat(pits.z) if pits is not None else None
    spike_thr = {}
    for pi, pname in enumerate(policies):
        pa = half_a & (clean.policy_id == pi)
        v = sp_clean[pa][clean.scored[pa] & ~clean.prefall[pa]]
        v = v[torch.randperm(len(v), device=dev)[:16_000_000]]
        spike_thr[pname] = float(torch.quantile(v.float(), 1.0 - target_far / 2))
    print(f"[eval] spike thresholds (max |z| of base_lin_vel x, y): {spike_thr}", flush=True)
    rows = []
    for stat, wname, clip, k in itertools.product(grid["stat"], grid["weights"], grid["clip"], grid["k"]):
        w = torch.as_tensor(weights(wname), dtype=torch.float32, device=dev)
        S_clean, S_ev = clean.raw(w, clip, stat), events.raw(w, clip, stat)
        row = {"stat": stat, "weights": wname, "clip": clip, "k": k, "per_policy": {}}
        for pi, pname in enumerate(policies):
            thr = spike_thr[pname]
            pa, pb = half_a & (clean.policy_id == pi), ~half_a & (clean.policy_id == pi)
            vals = S_clean[pa][clean.scored[pa]]
            m0, s0 = float(vals.mean()), float(vals.std())
            xa = (S_clean[pa] - m0) / s0
            nf, _ = cusum_alarm_counts(xa, clean.scored[pa], clean.reset[pa], k, hs, clean.prefall[pa], sp_clean[pa] > thr)
            far = nf.float() / clean.scored[pa].sum()
            okh = (far <= target_far).nonzero()
            h = float(hs[okh[0, 0]]) if len(okh) else float(hs[-1])
            xb = (S_clean[pb] - m0) / s0
            ab, _ = cusum(xb, clean.scored[pb], clean.reset[pb], k, h, sp_clean[pb] > thr)
            far_b = float((ab & ~clean.prefall[pb]).sum() / clean.scored[pb].sum())
            pe = events.policy_id == pi
            xe = (S_ev[pe] - m0) / s0
            ae, _ = cusum(xe, events.scored[pe], events.reset[pe], k, h, sp_ev[pe] > thr)
            det = {}
            for eid, ename in enumerate(ev_names):
                sel = (ev_type[pe] == eid) & ~reset_before[pe]
                win = ae[sel][:, E + 1 : E + 151]
                first = torch.where(win.any(1), win.float().argmax(1) + 1, torch.full_like(win[:, 0], 10**6, dtype=torch.long))
                post = xe[sel][:, E + 51 : E + 151]
                post_scored = events.scored[pe][sel][:, E + 51 : E + 151]
                hit1 = first <= 50
                det[ename] = {"envs": int(sel.sum()),
                              "detect_1s": float(hit1.float().mean()),
                              "detect_2s": float((first <= 100).float().mean()),
                              "detect_3s": float((first <= 150).float().mean()),
                              "median_delay_steps": float(first[first <= 150].float().median()) if (first <= 150).any() else float("nan"),
                              "x_mean_1_3s_after": float(post[post_scored].mean()) if post_scored.any() else float("nan"),
                              "alarms_per_s_1_3s_after": float(ae[sel][:, E + 51 : E + 151].float().sum() / max(1, int(sel.sum())) / 2.0)}
            row["per_policy"][pname] = {"m0": m0, "s0": s0, "k": k, "h": h, "spike_z": thr,
                                        "far_calib": float(far[okh[0, 0]]) if len(okh) else float(far[-1]),
                                        "far_verify": far_b, "events": det}
            if pits is not None:
                pp = pits.policy_id == pi
                S_p = pits.raw(w, clip, stat)[pp]
                ap_, _ = cusum((S_p - m0) / s0, pits.scored[pp], pits.reset[pp], k, h, sp_pits[pp] > thr)
                dec = torch.from_numpy(pits_data.deceived).to(dev)[pp]  # [n, T]: scan fake at this frame
                after = torch.zeros_like(dec)
                after[:, E + 1 :] = True
                sc = pits.scored[pp] & after
                res = {}
                for tag, mm in (("deceived", sc & dec), ("truthful_scan", sc & ~dec.any(1, keepdim=True))):
                    res[tag] = {"alarms_per_s": float(ap_[mm].float().sum() / max(1, int(mm.sum())) * 50.0),
                                "scored_s": float(mm.sum() / 50.0)}
                row["per_policy"][pname]["pits"] = res
        key = [row["per_policy"][p]["events"][e]["detect_2s"] for p in policies for e in KEY_EVENTS if e in ev_names]
        row["key_detect_mean"] = float(np.mean(key))
        row["far_verify_max"] = max(row["per_policy"][p]["far_verify"] for p in policies)
        rows.append(row)
        print(f"[eval] {stat:4s} {wname:6s} clip {clip:4.0f} k {k:4.2f}: key detection (2 s) {row['key_detect_mean']:.3f}  "
              f"FAR verify {row['far_verify_max'] / target_far:.2f} x target", flush=True)
    feasible = [r for r in rows if r["far_verify_max"] <= 2 * target_far] or rows
    best = max(feasible, key=lambda r: (r["key_detect_mean"], -r["far_verify_max"]))
    return {"grid": rows, "best": best, "spike_thresholds": spike_thr}


# --- plots ------------------------------------------------------------------------------------------------
def plots(out: Path, results: dict, traces: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # event traces
    fig, axes = plt.subplots(1, len(traces), figsize=(5 * len(traces), 3.6), squeeze=False)
    for ax, (pname, tr) in zip(axes[0], traces.items()):
        for ename, (tt, mean) in tr.items():
            ax.plot(tt, mean, label=ename, lw=1.2)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"standardized surprisal around the event ({pname})")
        ax.set_xlabel("steps after the event (50 = 1 s)")
        ax.set_ylabel("x = (S - m0) / s0, mean over envs")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "surprisal_events.png", dpi=120)
    plt.close(fig)

    # multi-step errors
    ms = results["multi_step"]["horizons"]
    hs = [int(h) for h in ms]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, _ in state_term_slices():
        ax[0].plot(hs, [ms[str(h)][f"{name}_rmse"] / max(1e-9, ms[str(h)][f"{name}_rmse_hold"]) for h in hs], marker="o", label=name)
    ax[0].set_xlabel("steps ahead")
    ax[0].set_ylabel("RMSE / RMSE of 'state stays'")
    ax[0].set_title("open-loop state error (recorded actions)")
    ax[0].legend(fontsize=7)
    ax[1].plot(hs, [ms[str(h)]["pos_xy_err_m"] for h in hs], marker="o", label="mean xy error")
    ax[1].plot(hs, [ms[str(h)]["true_xy_dist_m"] for h in hs], marker="o", label="distance travelled")
    ax[1].set_xlabel("steps ahead")
    ax[1].set_ylabel("m")
    ax[1].set_title("integrated base displacement")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "multi_step.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="logs/wm/<run>")
    p.add_argument("--checkpoint", default="model.pt", help="model.pt, or model_fall.pt after refit_fall")
    p.add_argument("--dyn_split", default=None, help="split for the dynamics val metrics (default: the model's training split)")
    p.add_argument("--fall_split", default=None, help="split for the fall val metrics (default: <data>/<prefix>train)")
    p.add_argument("--data", default="logs/wm_data", help="dir with w4_train, w4_clean, w4_events, w4_events_pits")
    p.add_argument("--prefix", default="w4_")
    p.add_argument("--out", required=True)
    p.add_argument("--target_arl", type=float, default=30000.0, help="steps between false alarms (30000 = 10 min)")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--var_floor_frac", type=float, default=0.1, help="var_floor = (frac * delta_std)^2")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    model, ckpt = load_model(Path(args.run) / args.checkpoint, args.device)
    S = model.state_dim
    var_floor = (args.var_floor_frac * model.out_std[:S]) ** 2
    lay_pol = load_interface()["world_model"]["policies"]
    results = {"run": args.run, "checkpoint": args.checkpoint, "train_data": ckpt["data"], "fall_refit": ckpt.get("fall_refit"),
               "checkpoint_iter": ckpt["iter"], "val_at_checkpoint": ckpt["val"],
               "warmup": args.warmup, "var_floor_frac": args.var_floor_frac, "target_arl_steps": args.target_arl}

    # 1-3 on held-out envs of the dynamics training split and of the fall split, and on the clean split
    dyn_split = Path(args.dyn_split or ckpt["data"])
    fall_split = Path(args.fall_split or data / f"{args.prefix}train")
    results["dyn_split"], results["fall_split"] = str(dyn_split), str(fall_split)
    for split, tag in ((dyn_split, "dyn"), (fall_split, "fall")):
        if tag == "fall" and split.resolve() == dyn_split.resolve():
            results["fall_val"] = results.pop("_fall_dyn")
            continue
        n_envs = json.loads((split / "meta.json").read_text())["num_envs"]
        va = WMData(split, device=args.device, envs=val_envs(n_envs), extra=True)
        Pv = predict_split(model, va)
        if tag == "dyn":
            results["one_step_val"] = one_step_breakdown(va, Pv, va.train_ok, model)
            results["multi_step"] = multi_step(model, va)
            results["_fall_dyn"] = fall_eval(va, Pv)
            print("[eval] val done:", json.dumps({k: results["one_step_val"]["all"]["all"][k] for k in ("z_std", "cov_1sigma", "cov_2sigma")}), flush=True)
        else:
            results.pop("_fall_dyn", None)
            results["fall_val"] = fall_eval(va, Pv)
            results["one_step_fall_split"] = one_step_breakdown(va, Pv, va.train_ok, model)
        del va, Pv

    clean = WMData(data / f"{args.prefix}clean", device=args.device, extra=True)
    Pc = predict_split(model, clean)
    results["one_step_clean"] = one_step_breakdown(clean, Pc, clean.train_ok, model)
    results["fall_clean"] = fall_eval(clean, Pc)
    sc_clean = Scorer(clean, Pc, var_floor, args.warmup)
    del Pc
    clean.scan = clean.frames = None

    ev = WMData(data / f"{args.prefix}events", device=args.device, extra=True)
    Pe = predict_split(model, ev)
    results["fall_events"] = fall_eval(ev, Pe)
    sc_ev = Scorer(ev, Pe, var_floor, args.warmup)
    del Pe
    ev.scan = ev.frames = None
    pits_dir = data / f"{args.prefix}events_pits"
    pits, sc_pits = None, None
    if pits_dir.exists():
        pits = WMData(pits_dir, device=args.device, extra=True)
        Pp = predict_split(model, pits)
        results["fall_pits"] = fall_eval(pits, Pp)
        sc_pits = Scorer(pits, Pp, var_floor, args.warmup)
        del Pp
        pits.scan = pits.frames = None

    grid = {"stat": ["norm", "nll"], "weights": ["all", "group", "base", "joints"], "clip": [9.0, 25.0], "k": [0.1, 0.25, 0.5, 1.0]}
    cal = calibrate(sc_clean, sc_ev, ev, sc_pits, pits, 1.0 / args.target_arl, lay_pol, ev.meta["event_step"], grid)
    results["surprisal"] = cal
    best = cal["best"]
    print(f"[eval] best: {best['stat']} {best['weights']} clip {best['clip']} k {best['k']}  key detection {best['key_detect_mean']:.3f}")

    # traces of the standardized score around events for the chosen configuration
    w = torch.as_tensor(weights(best["weights"]), dtype=torch.float32, device=args.device)
    S_ev = sc_ev.raw(w, best["clip"], best["stat"])
    E = ev.meta["event_step"]
    ev_type = torch.from_numpy(ev.info["event_type"].astype(np.int64)).to(args.device)
    rb = torch.from_numpy(ev.info["reset_before_event"]).to(args.device)
    traces = {}
    for pi, pname in enumerate(lay_pol):
        cfgp = best["per_policy"][pname]
        x = (S_ev - cfgp["m0"]) / cfgp["s0"]
        tr = {}
        for eid, ename in enumerate(ev.meta["events"]):
            sel = (ev_type == eid) & ~rb & (ev.policy_id == pi)
            seg = x[sel][:, E - 50 : E + 150]
            m = sc_ev.scored[sel][:, E - 50 : E + 150]
            mean = (seg * m).sum(0) / m.sum(0).clamp(min=1)
            tr[ename] = (list(range(-50, -50 + seg.shape[1])), mean.cpu().numpy().tolist())
        traces[pname] = tr
    results["event_traces"] = traces
    plots(out, results, traces)

    # robot_interface.yaml snippet
    snippet = {"surprisal": {
        "statistic": best["stat"],
        "clip_z2": best["clip"],
        "weights": [round(float(v), 6) for v in weights(best["weights"])],
        "weights_scheme": best["weights"],
        "var_floor": [float(v) for v in var_floor.cpu().numpy()],
        "z_bias": [0.0] * S,
        "z_scale": [1.0] * S,
        "warmup_steps": args.warmup,
        "spike_dims": list(SPIKE_DIMS),
        "target_arl_steps": args.target_arl,
        "per_policy": {p: {k_: float(best["per_policy"][p][k_]) for k_ in ("m0", "s0", "k", "h", "spike_z")} for p in lay_pol},
    }}
    (out / "calibration.yaml").write_text(yaml.safe_dump(snippet, sort_keys=False, width=200, default_flow_style=None))
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"[eval] wrote {out}/results.json, calibration.yaml, figures", flush=True)


if __name__ == "__main__":
    main()
