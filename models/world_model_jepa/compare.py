"""Compare world models on one surprisal / fall / ghost-path protocol (W7: JEPA vs the W4-5 ensemble).

Models: --model NAME=KIND:CHECKPOINT (repeatable)
  ensemble:<ckpt>  W4-5 Gaussian ensemble (models/world_model). Two detectors: NAME (diagonal weightings, the W5
                   grid) and NAME_cov (maha_* weightings: residual correlation whitened, estimated on the
                   training split's val envs)
  latent:<ckpt>    latent world model (models/world_model_jepa); one detector NAME over LATENT_COMBOS
A seed suffix "@s<k>" (A@s1, B@s2) groups runs in the summary.

Stages (--stage):
  select   detector grid (models/world_model/evaluate.py calibrate) on the selection splits only: false alarms on
           w4_clean (h from half A, checked on half B), detection on w4_events / w4_events_pits. Writes
           <out>/models/<name>/selection.json; its key_detect_mean is also the sweep criterion (sweep_pick.py).
  report   the selected detectors, frozen, on report splits never used for any choice:
             w7_events_test       detection within 1 / 2 / 3 s per event and policy, delays, score traces;
                                  detection vs false-alarm curve (h scaled, false alarms on w4_clean half B)
             w7_events_pits_test  alarm rate with a deceived vs truthful scan
             w6_ice_test          alarms before falls on visible / hidden / no ice, ice contacts, false alarms off ice
             w6_geo_test          false alarms on unseen terrain (normal physics)
             fall heads           AUROC on clean / events_test / pits_test / ice_test
             ghost path           open-loop base position error on the training split's val envs
           -> <out>/models/<name>/report.json + arrays.npz (per-env / per-fall hits for paired comparisons)
  summary  all models in <out>/models: tables, paired bootstrap vs --baseline, figures -> <out>/results.json

  python -m models.world_model_jepa.compare --stage select report --model A=ensemble:logs/wm/<A run>/model_fall.pt \
      --model B=latent:logs/wm/<B run>/model_fall.pt --out experiments/w7_jepa_wm
  python -m models.world_model_jepa.compare --stage summary --baseline A --out experiments/w7_jepa_wm
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from isaac.interface import load_interface
from models.world_model import evaluate as wm_eval
from models.world_model.dataset import WMData, val_envs
from models.world_model.ensemble import load_model as load_ensemble
from models.world_model.surprisal import cusum, cusum_alarm_counts
from models.terrain_risk.evaluate_ice import CLASSES, ice_alarm_metrics

from .model import load_model as load_latent
from .predict import LATENT_COMBOS, LatentScorer, multi_step_latent, predict_latent

KEY_EVENTS = wm_eval.KEY_EVENTS
GRID_K = [0.1, 0.25, 0.5, 1.0]
ENSEMBLE_DIAG = list(itertools.product(["norm", "nll"], ["all", "group", "base", "joints"], [9.0, 25.0]))
ENSEMBLE_MAHA = [("norm", w, c) for w in ("maha_all", "maha_joints") for c in (9.0, 25.0)]
H_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 3.0]


# --- models -------------------------------------------------------------------------------------------------
class Model:
    """A world model behind one interface: scorer(d) and predictions for the fall / ghost-path metrics."""

    def __init__(self, name: str, kind: str, ckpt: str, device: str, warmup: int, var_floor_frac: float,
                 train_split: str | None = None):
        self.name, self.kind, self.ckpt_path, self.dev, self.warmup = name, kind, ckpt, device, warmup
        if kind == "ensemble":
            self.net, self.ckpt = load_ensemble(ckpt, device)
            S = self.net.state_dim
            self.var_floor = (var_floor_frac * self.net.out_std[:S]) ** 2
            self.whiten = None
        elif kind == "latent":
            self.net, self.ckpt = load_latent(ckpt, device)
        else:
            raise ValueError(kind)
        self.train_split = Path(train_split or self.ckpt["data"])  # dynamics training split (whitening, ghost path)

    def detectors(self) -> dict[str, list]:
        if self.kind == "ensemble":
            return {self.name: ENSEMBLE_DIAG, f"{self.name}_cov": ENSEMBLE_MAHA}
        return {self.name: LATENT_COMBOS}

    def combos(self) -> list:
        return [c for v in self.detectors().values() for c in v]

    def ensure_whitening(self):
        if self.kind != "ensemble" or self.whiten is not None:
            return
        n_envs = json.loads((self.train_split / "meta.json").read_text())["num_envs"]
        va = WMData(self.train_split, device=self.dev, envs=val_envs(n_envs))
        P = wm_eval.predict_split(self.net, va)
        sc = wm_eval.Scorer(va, P, self.var_floor, self.warmup)
        self.whiten = wm_eval.whitening(sc, sc.scored)
        del va, P, sc
        free()

    def run(self, d: WMData) -> tuple[object, dict]:
        """(scorer, P) for a split; P has "fall" [N, T, n_fall]."""
        if self.kind == "ensemble":
            self.ensure_whitening()
            P = wm_eval.predict_split(self.net, d)
            sc = wm_eval.Scorer(d, P, self.var_floor, self.warmup, whiten=self.whiten)
            return sc, {"fall": P["fall"]}
        P = predict_latent(self.net, d, combos=self.combos())
        return LatentScorer(d, P, self.warmup), {"fall": P["fall"]}

    def multi_step(self, d: WMData) -> dict:
        if self.kind == "ensemble":
            return wm_eval.multi_step(self.net, d)
        return multi_step_latent(self.net, d)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def load_split(path: Path, dev: str) -> WMData:
    return WMData(path, device=dev, extra=True)


def strip(d: WMData):
    d.scan = d.frames = None
    free()


# --- detectors ------------------------------------------------------------------------------------------------
def best_row(rows: list, combos: list, target_far: float) -> dict:
    """calibrate()'s selection rule restricted to `combos`."""
    keep = [r for r in rows if (r["stat"], r["weights"], float(r["clip"])) in {(s, w, float(c)) for s, w, c in combos}]
    feasible = [r for r in keep if r["far_verify_max"] <= 2 * target_far] or keep
    return max(feasible, key=lambda r: (r["key_detect_mean"], -r["far_verify_max"]))


def alarms(sc, det: dict, policies: list[str], h_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Alarms and standardized scores x [N, T] of a frozen detector (per-policy m0, s0, k, h, spike_z)."""
    S = sc.score(det["weights"], det["clip"], det["stat"])
    sp = sc.spike()
    al = torch.zeros_like(sc.scored)
    x = torch.zeros_like(S)
    for pi, pname in enumerate(policies):
        c = det["per_policy"][pname]
        sel = sc.policy_id == pi
        x[sel] = (S[sel] - c["m0"]) / c["s0"]
        al[sel], _ = cusum(x[sel], sc.scored[sel], sc.reset[sel], c["k"], c["h"] * h_scale, sp[sel] > c["spike_z"])
    return al, x


def false_alarm_rate(sc, al: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Alarms per 10 min (30000 scored steps), not counting the 50 steps before a fall."""
    m = sc.scored if mask is None else sc.scored & mask
    return float((al & m & ~sc.prefall).sum() / m.sum().clamp(min=1) * 30000)


def event_report(sc, al, x, d: WMData, policies: list[str]) -> tuple[dict, dict]:
    E = d.meta["event_step"]
    ev_type = torch.from_numpy(d.info["event_type"].astype(np.int64)).to(al.device)
    rb = torch.from_numpy(d.info["reset_before_event"]).to(al.device)
    first = wm_eval.first_alarm(al[:, E + 1 : E + 151])
    out = {}
    for pi, pname in enumerate(policies):
        for eid, ename in enumerate(d.meta["events"]):
            sel = (ev_type == eid) & ~rb & (d.policy_id == pi)
            f = first[sel]
            seg, m = x[sel][:, E - 50 : E + 150], sc.scored[sel][:, E - 50 : E + 150]
            out.setdefault(pname, {})[ename] = {
                "envs": int(sel.sum()), "detect_1s": float((f <= 50).float().mean()), "detect_2s": float((f <= 100).float().mean()),
                "detect_3s": float((f <= 150).float().mean()),
                "median_delay_steps": float(f[f <= 150].float().median()) if (f <= 150).any() else float("nan"),
                "trace_x_mean": ((seg * m).sum(0) / m.sum(0).clamp(min=1)).cpu().numpy().round(4).tolist()}
    arrays = {"ev_first": first.cpu().numpy(), "ev_type": ev_type.cpu().numpy(), "ev_reset_before": rb.cpu().numpy(),
              "ev_policy": d.policy_id.cpu().numpy()}
    return out, arrays


def pits_report(sc, al, d: WMData) -> dict:
    E = d.meta["event_step"]
    dec = torch.from_numpy(d.deceived).to(al.device)
    after = torch.zeros_like(dec)
    after[:, E + 1 :] = True
    res = {}
    for pi, pname in enumerate(load_interface()["world_model"]["policies"]):
        pp = (sc.policy_id == pi)[:, None]
        scd = sc.scored & after & pp
        r = {}
        for tag, mm in (("deceived", scd & dec), ("truthful_scan", scd & ~dec.any(1, keepdim=True))):
            r[tag] = {"alarms_per_s": float(al[mm].float().sum() / max(1, int(mm.sum())) * 50.0), "scored_s": float(mm.sum() / 50.0)}
        res[pname] = r
    return res


def fa_curve(clean_sc, ev_sc, ev_d: WMData, det: dict, policies: list[str]) -> dict:
    """Detection within 2 s on the report events vs false alarms per 10 min on clean half B, h scaled."""
    half_b = torch.arange(clean_sc.scored.shape[0], device=clean_sc.scored.device) % 2 == 1
    E = ev_d.meta["event_step"]
    ev_type = torch.from_numpy(ev_d.info["event_type"].astype(np.int64)).to(half_b.device)
    rb = torch.from_numpy(ev_d.info["reset_before_event"]).to(half_b.device)
    names = ev_d.meta["events"]
    out = {p: [] for p in policies}
    for s in H_SCALES:
        al_c, _ = alarms(clean_sc, det, policies, s)
        al_e, _ = alarms(ev_sc, det, policies, s)
        first = wm_eval.first_alarm(al_e[:, E + 1 : E + 151])
        for pi, pname in enumerate(policies):
            far = false_alarm_rate(clean_sc, al_c, (half_b & (clean_sc.policy_id == pi))[:, None])
            det2 = {e: float((first[(ev_type == i) & ~rb & (ev_d.policy_id == pi)] <= 100).float().mean()) for i, e in enumerate(names)}
            out[pname].append({"h_scale": s, "false_alarms_per_10min": far, "detect_2s": det2,
                               "key_detect_2s": float(np.mean([det2[e] for e in KEY_EVENTS]))})
    return out


# --- stages ---------------------------------------------------------------------------------------------------
def stage_select(m: Model, args, policies) -> dict:
    data = Path(args.data)
    out = {}
    cl = load_split(data / "w4_clean", m.dev)
    sc_c, _ = m.run(cl)
    strip(cl)
    ev = load_split(data / "w4_events", m.dev)
    sc_e, _ = m.run(ev)
    strip(ev)
    pits = load_split(data / "w4_events_pits", m.dev)
    sc_p, _ = m.run(pits)
    strip(pits)
    target = 1.0 / args.target_arl
    cal = wm_eval.calibrate(sc_c, sc_e, ev, sc_p, pits, target, policies, ev.meta["event_step"], {"k": GRID_K}, combos=m.combos())
    for det_name, combos in m.detectors().items():
        out[det_name] = best_row(cal["grid"], combos, target)
        b = out[det_name]
        print(f"[compare] {det_name}: {b['stat']} {b['weights']} clip {b['clip']} k {b['k']}  key detection (2 s, w4_events) "
              f"{b['key_detect_mean']:.3f}  FAR verify {b['far_verify_max'] / target:.2f} x target", flush=True)
    sel = {"model": m.name, "kind": m.kind, "checkpoint": m.ckpt_path, "train_data": str(m.train_split),
           "model_config": m.ckpt.get("model_config"), "args": m.ckpt.get("args"), "target_arl_steps": args.target_arl,
           "warmup": args.warmup, "detectors": out, "spike_thresholds": cal["spike_thresholds"],
           "grid": [{k: r[k] for k in ("stat", "weights", "clip", "k", "key_detect_mean", "far_verify_max")} for r in cal["grid"]]}
    return sel


def stage_report(m: Model, sel: dict, args, policies) -> tuple[dict, dict]:
    data = Path(args.data)
    rep, arr = {}, {}
    dets = sel["detectors"]
    # clean (false alarms on half B, fall AUROC, curve)
    cl = load_split(data / "w4_clean", m.dev)
    sc_c, Pc = m.run(cl)
    rep["fall_clean"] = wm_eval.fall_eval(cl, Pc)
    strip(cl)
    ev = load_split(data / "w7_events_test", m.dev)
    sc_e, Pe = m.run(ev)
    rep["fall_events_test"] = wm_eval.fall_eval(ev, Pe)
    strip(ev)
    half_b = (torch.arange(sc_c.scored.shape[0], device=sc_c.scored.device) % 2 == 1)[:, None]
    for dn, det in dets.items():
        r = rep.setdefault(dn, {"config": {k: det[k] for k in ("stat", "weights", "clip", "k")},
                                "per_policy": {p: {k: det["per_policy"][p][k] for k in ("m0", "s0", "k", "h", "spike_z")} for p in policies},
                                "key_detect_selection": det["key_detect_mean"]})
        al_c, _ = alarms(sc_c, det, policies)
        r["false_alarms_per_10min_clean_half_b"] = {p: false_alarm_rate(sc_c, al_c, half_b & (sc_c.policy_id == i)[:, None])
                                                    for i, p in enumerate(policies)}
        al, x = alarms(sc_e, det, policies)
        r["events_test"], a = event_report(sc_e, al, x, ev, policies)
        r["key_detect_2s_events_test"] = float(np.mean([r["events_test"][p][e]["detect_2s"] for p in policies for e in KEY_EVENTS]))
        arr.update({f"{dn}/{k}": v for k, v in a.items()})
        r["fa_curve"] = fa_curve(sc_c, sc_e, ev, det, policies)
        print(f"[compare] {dn}: events_test key detection (2 s) {r['key_detect_2s_events_test']:.3f}  "
              f"clean FA/10min {r['false_alarms_per_10min_clean_half_b']}", flush=True)
    del sc_c, Pc, sc_e, Pe
    free()
    # pits
    pits = load_split(data / "w7_events_pits_test", m.dev)
    sc_p, Pp = m.run(pits)
    rep["fall_pits_test"] = wm_eval.fall_eval(pits, Pp)
    strip(pits)
    for dn, det in dets.items():
        al, _ = alarms(sc_p, det, policies)
        rep[dn]["pits_test"] = pits_report(sc_p, al, pits)
    del sc_p, Pp
    free()
    # ice
    ice_dir = data / "w6_ice_test"
    ice = load_split(ice_dir, m.dev)
    sc_i, Pi = m.run(ice)
    rep["fall_ice_test"] = wm_eval.fall_eval(ice, Pi)
    strip(ice)
    T = ice.T
    ld = lambda n: np.asarray(np.load(ice_dir / f"{n}.npy", mmap_mode="r")[:T])  # noqa: E731
    term, trunc, es, foot_ice = ld("terminated"), ld("truncated"), ld("episode_step"), ld("foot_ice")
    for dn, det in dets.items():
        al, _ = alarms(sc_i, det, policies)
        res = ice_alarm_metrics(al.cpu().numpy(), term, trunc, es, foot_ice, details=True)
        f = res.pop("_falls")
        arr.update({f"{dn}/ice_{k}": v for k, v in f.items()})
        rep[dn]["ice_test"] = res
        print(f"[compare] {dn}: ice_test falls with alarm within 1 s {res['falls_with_alarm_within_1s']}", flush=True)
    del sc_i, Pi, ice
    free()
    # geo
    geo = load_split(data / "w6_geo_test", m.dev)
    sc_g, _ = m.run(geo)
    strip(geo)
    for dn, det in dets.items():
        al, _ = alarms(sc_g, det, policies)
        rep[dn]["false_alarms_per_10min_geo_test"] = {p: false_alarm_rate(sc_g, al, (sc_g.policy_id == i)[:, None])
                                                      for i, p in enumerate(policies)}
    del sc_g, geo
    free()
    # ghost path on the training split's val envs
    n_envs = json.loads((m.train_split / "meta.json").read_text())["num_envs"]
    va = WMData(m.train_split, device=m.dev, envs=val_envs(n_envs))
    rep["multi_step"] = m.multi_step(va)
    del va
    free()
    return rep, arr


# --- summary --------------------------------------------------------------------------------------------------
def group_of(name: str) -> str:
    base, _, rest = name.partition("@")
    det = rest.partition("_")[2] if "_" in rest else ""
    return base + ("_" + det if det else "")


def paired_boot(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    """Mean of b - a over units (rows) with a bootstrap CI over the rows; a, b [units] in [0, 1]."""
    rng = np.random.default_rng(seed)
    dlt = b.astype(float) - a.astype(float)
    if len(dlt) == 0:
        return {"diff": float("nan"), "ci95": [float("nan")] * 2, "units": 0}
    bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(n)]
    return {"diff": float(dlt.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "units": int(len(dlt))}


def stage_summary(args, policies):
    root = Path(args.out) / "models"
    reports, arrays = {}, {}
    for mdir in sorted(p for p in root.iterdir() if (p / "report.json").exists()):
        rep = json.loads((mdir / "report.json").read_text())
        arr = dict(np.load(mdir / "arrays.npz"))
        for dn, r in rep.items():
            if isinstance(r, dict) and "config" in r:
                reports[dn] = {**r, "_model": mdir.name, "_shared": {k: rep[k] for k in rep if k.startswith(("fall_", "multi_step"))}}
                arrays[dn] = {k.split("/", 1)[1]: v for k, v in arr.items() if k.startswith(dn + "/")}
    names = sorted(reports)
    groups: dict[str, list[str]] = {}
    for n_ in names:
        groups.setdefault(group_of(n_), []).append(n_)
    ev_names = ["none", "mu_0.3", "mu_0.1", "mu_0.05", "foot_0.05", "push_0.5", "push_1.0"]

    def ev_hits(dn: str, pi: int, eid: int, w: int) -> np.ndarray:
        a = arrays[dn]
        sel = (a["ev_type"] == eid) & ~a["ev_reset_before"] & (a["ev_policy"] == pi)
        return a["ev_first"][sel] <= w

    def metrics(dn: str) -> dict:
        r = reports[dn]
        out = {"config": r["config"], "key_detect_2s_events_test": r["key_detect_2s_events_test"],
               "key_detect_selection": r["key_detect_selection"]}
        for p in policies:
            for e in ev_names:
                for k in ("detect_1s", "detect_2s"):
                    out[f"{p}/{e}/{k}"] = r["events_test"][p][e][k]
            out[f"{p}/fa_clean"] = r["false_alarms_per_10min_clean_half_b"][p]
            out[f"{p}/fa_geo"] = r["false_alarms_per_10min_geo_test"][p]
            out[f"{p}/pits_ratio"] = r["pits_test"][p]["deceived"]["alarms_per_s"] / max(1e-9, r["pits_test"][p]["truthful_scan"]["alarms_per_s"])
            out[f"{p}/pits_deceived_per_s"] = r["pits_test"][p]["deceived"]["alarms_per_s"]
        for c in CLASSES:
            out[f"ice/{c}_falls_1s"] = r["ice_test"]["falls_with_alarm_within_1s"][c]
        for c in ("visible", "hidden"):
            out[f"ice/{c}_contacts_1s"] = r["ice_test"][f"{c}_ice_contacts"]["alarm_within_1s"]
        out["ice/fa_off_ice"] = r["ice_test"]["alarms_per_10min_off_ice_no_fall"]
        sh = r["_shared"]
        for split in ("clean", "events_test", "pits_test", "ice_test"):
            for h in ("h25", "h50"):
                out[f"fall_auroc/{split}/{h}"] = sh[f"fall_{split}"][f"{h}_all"]["auroc"]
        for hz in ("10", "25", "50"):
            out[f"ghost_xy_err_m/{hz}"] = sh["multi_step"]["horizons"][hz]["pos_xy_err_m"]
        return out

    per_det = {dn: metrics(dn) for dn in names}
    per_group = {}
    for g, mem in groups.items():
        keys = [k for k in per_det[mem[0]] if k != "config" and isinstance(per_det[mem[0]][k], (int, float)) and per_det[mem[0]][k] is not None]
        per_group[g] = {"members": mem, **{k: {"mean": float(np.nanmean([per_det[m_][k] if per_det[m_][k] is not None else np.nan for m_ in mem])),
                                                 "min": float(np.nanmin([per_det[m_][k] if per_det[m_][k] is not None else np.nan for m_ in mem])),
                                                 "max": float(np.nanmax([per_det[m_][k] if per_det[m_][k] is not None else np.nan for m_ in mem]))}
                                            for k in keys}}
    # paired comparisons against the baseline group: seed-averaged hit per env (events) / per fall (ice)
    paired = {}
    base = args.baseline
    if base in groups:
        for g in groups:
            if g == base:
                continue
            res = {}
            for pi, p in enumerate(policies):
                for eid, e in enumerate(ev_names):
                    for w, tag in ((50, "1s"), (100, "2s")):
                        A = np.mean([ev_hits(m_, pi, eid, w) for m_ in groups[base]], axis=0)
                        B = np.mean([ev_hits(m_, pi, eid, w) for m_ in groups[g]], axis=0)
                        res[f"{p}/{e}/detect_{tag}"] = paired_boot(A, B)
            for ci, c in enumerate(CLASSES):
                A = np.mean([arrays[m_]["ice_hit_1s"][arrays[m_]["ice_cls"] == ci] for m_ in groups[base]], axis=0)
                B = np.mean([arrays[m_]["ice_hit_1s"][arrays[m_]["ice_cls"] == ci] for m_ in groups[g]], axis=0)
                res[f"ice/{c}_falls_1s"] = paired_boot(A, B)
            paired[f"{g} - {base}"] = res
    results = {"detectors": per_det, "groups": per_group, "paired_vs_baseline": paired, "baseline": base}
    (Path(args.out) / "results.json").write_text(json.dumps(results, indent=1, default=float))
    figures(Path(args.out), reports, groups, policies)
    # console table
    print(f"{'group':14s} " + " ".join(f"{c:>9s}" for c in ("key2s", "foot2s v1", "foot2s v2", "mu.3 v1", "mu.3 v2", "iceHid1s", "iceVis1s", "faGeo v2")))
    for g, s in per_group.items():
        vals = [s["key_detect_2s_events_test"], s["v1/foot_0.05/detect_2s"], s["v2/foot_0.05/detect_2s"], s["v1/mu_0.3/detect_2s"],
                s["v2/mu_0.3/detect_2s"], s["ice/hidden_falls_1s"], s["ice/visible_falls_1s"], s["v2/fa_geo"]]
        print(f"{g:14s} " + " ".join(f"{v['mean']:9.3f}" for v in vals))
    print(f"[compare] wrote {Path(args.out) / 'results.json'}", flush=True)


def figures(out: Path, reports: dict, groups: dict, policies: list[str]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = {g: mem[0] for g, mem in groups.items()}
    fig, axes = plt.subplots(1, len(policies), figsize=(5.5 * len(policies), 4), squeeze=False)
    for ax, p in zip(axes[0], policies):
        for g, dn in first.items():
            c = reports[dn]["fa_curve"][p]
            ax.plot([r["false_alarms_per_10min"] for r in c], [r["key_detect_2s"] for r in c], marker="o", ms=3, label=g)
        ax.axvline(1.0, color="k", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("false alarms per 10 min (w4_clean half B)")
        ax.set_ylabel("key events detected within 2 s (events_test)")
        ax.set_title(f"detection vs false alarms ({p}, seed 0 of each group)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "detection_vs_false_alarms.png", dpi=120)
    plt.close(fig)

    evs = ["mu_0.3", "mu_0.1", "foot_0.05", "push_0.5"]
    fig, axes = plt.subplots(len(policies), len(evs), figsize=(4 * len(evs), 3.2 * len(policies)), squeeze=False)
    for i, p in enumerate(policies):
        for j, e in enumerate(evs):
            ax = axes[i, j]
            for g, dn in first.items():
                tr = reports[dn]["events_test"][p][e]["trace_x_mean"]
                ax.plot(range(-50, -50 + len(tr)), tr, lw=1.1, label=g)
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(f"{p}: {e}", fontsize=9)
            if j == 0:
                ax.set_ylabel("standardized score x")
            if i == len(policies) - 1:
                ax.set_xlabel("steps after the event (50 = 1 s)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "event_traces.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", nargs="+", default=["select", "report"], choices=["select", "report", "summary"])
    p.add_argument("--model", action="append", default=[], help="NAME=KIND:CHECKPOINT, KIND ensemble|latent")
    p.add_argument("--data", default="logs/wm_data")
    p.add_argument("--out", required=True)
    p.add_argument("--baseline", default="A")
    p.add_argument("--train_split", default=None, help="dynamics training split dir (default: the path in each checkpoint)")
    p.add_argument("--target_arl", type=float, default=30000.0)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--var_floor_frac", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    policies = load_interface()["world_model"]["policies"]
    for spec in args.model:
        name, _, rest = spec.partition("=")
        kind, _, ckpt = rest.partition(":")
        mdir = Path(args.out) / "models" / name
        mdir.mkdir(parents=True, exist_ok=True)
        m = Model(name, kind, ckpt, args.device, args.warmup, args.var_floor_frac, args.train_split)
        if "select" in args.stage:
            sel = stage_select(m, args, policies)
            (mdir / "selection.json").write_text(json.dumps(sel, indent=1, default=float))
        if "report" in args.stage:
            sel = json.loads((mdir / "selection.json").read_text())
            rep, arr = stage_report(m, sel, args, policies)
            (mdir / "report.json").write_text(json.dumps(rep, indent=1, default=float))
            np.savez_compressed(mdir / "arrays.npz", **arr)
        del m
        free()
    if "summary" in args.stage:
        stage_summary(args, policies)


if __name__ == "__main__":
    main()
