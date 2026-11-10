"""Run a latent world model over a whole split: surprisal scores, fall probabilities, ghost-path rollouts.

Streaming over time (one step for all envs at a time) with a rolling buffer of the last H context latents, so
memory does not grow with the latent dim. At step u the target latent z̄_u = Ē(window ending u) is scored
against the moment-matched prediction made at u - 1:
    r_d = (z̄_u,d - mu_d) / sqrt(var_d + var_floor_d)
    norm, all:   S = mean_d 0.5 * (min(r_d^2, clip) - 1)
    nll,  all:   S = mean_d 0.5 * (min(r_d^2, clip) + log(2 pi (var_d + var_floor_d)))
    norm, maha_all: as norm with u = W r (W = the model's whitening matrix) in place of r
    spike = max_d |r_d|
These are the same statistics as the ensemble's (models/world_model/surprisal.py) with uniform weights, so
models/world_model/evaluate.py calibrate() runs on them unchanged via LatentScorer.
"""

from __future__ import annotations

import math

import torch

from models.world_model.dataset import WMData
from models.world_model.evaluate import score_masks

from .model import LatentWM

LATENT_COMBOS = [(stat, "all", clip) for stat in ("norm", "nll") for clip in (9.0, 25.0)] + \
                [("norm", "maha_all", clip) for clip in (9.0, 25.0)]


def window_now(d: WMData, n: torch.Tensor, t: int, L: int) -> torch.Tensor:
    """Raw state window [N, L, S] ending at step t, filled with the episode's first frame before its start."""
    offs = torch.arange(-(L - 1), 1, device=n.device)
    tt = torch.maximum(t + offs[None], d.ep_start[n, t][:, None]).clamp(min=0)
    return d.frames[n[:, None], tt][..., torch.as_tensor(d.layout.state_in_frame, device=n.device)]


@torch.no_grad()
def predict_latent(model: LatentWM, d: WMData, combos=LATENT_COMBOS, keep_latents: bool = False, keep_pred: bool = False) -> dict:
    """scores {(stat, weights, clip): [N, T]}, spike [N, T], fall [N, T, n_fall], disagreement [N, T]
    (mean over latent dims of the member-mean variance / predictive variance); latents z̄ [N, T, d] if asked;
    keep_pred (tests): the prediction made at t for t + 1, mean / var [N, T, d] (float32)."""
    N, T = d.N, d.T
    dev = d.frames.device
    H, L, M = model.H, model.L, model.M
    lay = d.layout
    n = torch.arange(N, device=dev)
    act_idx = torch.arange(lay.state_dim, lay.frame_dim, device=dev)
    onehot = d.onehot(n).unsqueeze(0).expand(M, -1, -1)
    scores = {c: torch.zeros(N, T, device=dev) for c in combos}
    spike = torch.zeros(N, T, device=dev)
    disagree = torch.zeros(N, T, device=dev)
    fall = torch.zeros(N, T, model.config["n_fall"], device=dev)
    lat = torch.zeros(N, T, model.d, device=dev, dtype=torch.float32 if keep_pred else torch.float16) if keep_latents else None
    pred = (torch.zeros(N, T, model.d, device=dev), torch.zeros(N, T, model.d, device=dev)) if keep_pred else None
    floor, W = model.var_floor, model.whiten
    hist = mu_prev = var_prev = None
    for t in range(T):
        win = window_now(d, n, t, L)
        z_on, z_tg = model.encode(win), model.encode(win, target=True)
        if lat is not None:
            lat[:, t] = z_tg.to(lat.dtype)
        if t == 0:
            hist = z_on[:, None].repeat(1, H, 1)
        else:
            hist = torch.cat([hist[:, 1:], z_on[:, None]], dim=1)
            rs = d.episode_step[:, t] == 0
            hist[rs] = z_on[rs][:, None].expand(-1, H, -1)
            v = var_prev + floor
            r = (z_tg - mu_prev) / torch.sqrt(v)
            r2 = r * r
            u2 = (r @ W.T) ** 2 if any(w.startswith("maha") for _, w, _ in combos) else None
            for (stat, wname, clip) in combos:
                q = u2 if wname.startswith("maha") else r2
                q = torch.clamp(q, max=clip)
                per = 0.5 * (q - 1.0) if stat == "norm" else 0.5 * (q + torch.log(2 * math.pi * v))
                scores[(stat, wname, clip)][:, t] = per.mean(-1)
            spike[:, t] = r.abs().amax(-1)
        prev_act = d.history_frames(n, torch.full_like(n, t))[..., act_idx]
        x = model.predictor_input(hist, prev_act, d.action[:, t], d.scan[:, t].float())
        mean, lv, h = model.step(x.unsqueeze(0).expand(M, -1, -1), hist[:, -1].unsqueeze(0).expand(M, -1, -1))
        mu_prev, var_prev = LatentWM.moments(mean, lv)
        if pred is not None:
            pred[0][:, t], pred[1][:, t] = mu_prev, var_prev
        disagree[:, t] = (((mean - mu_prev) ** 2).mean(0) / (var_prev + floor)).mean(-1)
        fall[:, t] = torch.sigmoid(model.fall_logits(h, onehot)).mean(0)
    return {"scores": scores, "spike": spike, "fall": fall, "disagreement": disagree, "latents": lat, "pred": pred}


class LatentScorer:
    """calibrate()-compatible scorer (see models/world_model/evaluate.py Scorer) over predict_latent's output."""

    def __init__(self, d: WMData, P: dict, warmup: int):
        self.scores = P["scores"]
        self._spike = P["spike"]
        self.scored, self.reset, self.prefall = score_masks(d, warmup)
        self.policy_id = d.policy_id

    def score(self, wname: str, clip: float, stat: str) -> torch.Tensor:
        return self.scores[(stat, wname, float(clip))]

    def spike(self) -> torch.Tensor:
        return self._spike


@torch.no_grad()
def multi_step_latent(model: LatentWM, d: WMData, horizons=(1, 5, 10, 25, 50), windows: int = 4096, seed: int = 0) -> dict:
    """Open-loop ghost path in latent space with recorded actions and scans, integrating the displacement head.
    Same windows (selection and seed) and metrics as models/world_model/evaluate.py multi_step."""
    from .data import gather

    K = max(horizons)
    M, H = model.M, model.H
    ok = d.train_ok.int()
    run = torch.zeros_like(ok)
    for t in range(d.T - 1, -1, -1):
        run[:, t] = ok[:, t] * (1 + (run[:, t + 1] if t + 1 < d.T else 0))
    n, t = (run >= K).nonzero(as_tuple=True)
    g = torch.Generator(device=n.device).manual_seed(seed)
    sel = torch.randperm(len(n), generator=g, device=n.device)[:windows]
    n, t = n[sel], t[sel]
    gg = gather(d, n, t, H, model.L, K)
    hist = model.encode(gg["wins"][:, :H])
    with torch.no_grad():
        z_tg = model.encode(gg["wins"][:, H:], target=True)  # offsets 1..K
    pos, yaw = torch.zeros(len(n), 3, device=n.device), torch.zeros(len(n), device=n.device)
    tpos, tyaw = torch.zeros_like(pos), torch.zeros_like(yaw)
    res = {}
    for k in range(K):
        tk = t + k
        x = model.predictor_input(hist, gg["prev_act"][:, k : k + H], gg["act"][:, k], d.scan[n, tk].float())
        mean, lv, h = model.step(x.unsqueeze(0).expand(M, -1, -1), hist[:, -1].unsqueeze(0).expand(M, -1, -1))
        mu, _ = LatentWM.moments(mean, lv)
        dd = (model.disp(h) * model.disp_std + model.disp_mean).mean(0)
        td = d.disp[n, tk]
        for (P_, Y_, D_) in ((pos, yaw, dd), (tpos, tyaw, td)):
            c, si = torch.cos(Y_), torch.sin(Y_)
            P_[:, 0] += c * D_[:, 0] - si * D_[:, 1]
            P_[:, 1] += si * D_[:, 0] + c * D_[:, 1]
            P_[:, 2] += D_[:, 2]
            Y_ += D_[:, 3]
        hist = torch.cat([hist[:, 1:], mu[:, None]], dim=1)
        if k + 1 in horizons:
            err = ((mu - z_tg[:, k]) ** 2).mean(-1)
            res[str(k + 1)] = {"pos_xy_err_m": float((pos[:, :2] - tpos[:, :2]).norm(dim=1).mean()),
                               "pos_xy_err_m_p90": float((pos[:, :2] - tpos[:, :2]).norm(dim=1).quantile(0.9)),
                               "true_xy_dist_m": float(tpos[:, :2].norm(dim=1).mean()),
                               "z_err_m": float((pos[:, 2] - tpos[:, 2]).abs().mean()),
                               "yaw_err_rad": float((yaw - tyaw).abs().mean()),
                               "latent_mse": float(err.mean()),
                               "latent_mse_over_var": float(err.mean() / z_tg[:, k].var(0).mean())}
    return {"windows": len(n), "horizons": res}
