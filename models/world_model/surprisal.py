"""Surprisal and CUSUM change detection on world-model predictions (W5). Reference implementation of
robot_interface.yaml world_model.surprisal; Unity's C# port is checked against go2_wm.json.

At step u the observed state s_u is scored against the prediction made at step u-1:
    z_d = ((s_u,d - mu_d) / sqrt(var_d + var_floor_d) - z_bias_d) / z_scale_d
    norm: S = sum_d w_d * 0.5 * (min(z_d^2, clip) - 1)            (mean ~0 for a calibrated model)
    nll:  S = sum_d w_d * 0.5 * (min(z_d^2, clip) + log(2 pi (var_d + var_floor_d)))
    x = (S - m0) / s0                                               (per policy, from clean data)
    g = max(0, g + x - k);  alarm when g > h, then g = 0                 (persistent changes: ice)
    spike: alarm when max_{d in spike_dims} |z_d| > spike_z, then g = 0  (impulses: pushes)
g is reset to 0 at every episode start and nothing is scored while episode_step < warmup_steps.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from isaac.interface import load_interface

BASE_TERMS = ("base_lin_vel", "base_ang_vel", "projected_gravity")
JOINT_TERMS = ("joint_pos", "joint_vel")
SPIKE_DIMS = (0, 1)  # base_lin_vel x, y: a push changes them instantly, contacts barely do


def state_term_slices(itf: dict | None = None) -> list[tuple[str, slice]]:
    itf = itf or load_interface()
    dims = {t["name"]: t["dim"] for t in itf["observation"]["terms"]}
    out, i = [], 0
    for name in itf["world_model"]["state_terms"]:
        out.append((name, slice(i, i + dims[name])))
        i += dims[name]
    return out


def weights(scheme: str, itf: dict | None = None) -> np.ndarray:
    """Per-dim weights over the state, summing to 1.
    all: every dim equal; group: every state term equal (split over its dims); base / joints: only those terms,
    group-balanced."""
    terms = state_term_slices(itf)
    S = terms[-1][1].stop
    w = np.zeros(S)
    if scheme == "all":
        w[:] = 1.0
    elif scheme in ("group", "base", "joints"):
        for name, sl in terms:
            if scheme == "group" or (scheme == "base" and name in BASE_TERMS) or (scheme == "joints" and name in JOINT_TERMS):
                w[sl] = 1.0 / (sl.stop - sl.start)
    else:
        raise ValueError(scheme)
    return w / w.sum()


def zscores(x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, var_floor: torch.Tensor,
            z_bias: torch.Tensor | float = 0.0, z_scale: torch.Tensor | float = 1.0) -> torch.Tensor:
    return ((x - mu) / torch.sqrt(var + var_floor) - z_bias) / z_scale


def score(z: torch.Tensor, var: torch.Tensor, var_floor: torch.Tensor, w: torch.Tensor, clip: float, stat: str) -> torch.Tensor:
    z2 = torch.clamp(z * z, max=clip)
    if stat == "norm":
        per = 0.5 * (z2 - 1.0)
    elif stat == "nll":
        per = 0.5 * (z2 + torch.log(2 * math.pi * (var + var_floor)))
    else:
        raise ValueError(stat)
    return (per * w).sum(-1)


def spike_stat(z: torch.Tensor) -> torch.Tensor:
    return z[..., list(SPIKE_DIMS)].abs().amax(-1)


def cusum(x: torch.Tensor, scored: torch.Tensor, reset: torch.Tensor, k: float, h: float, spike: torch.Tensor | None = None):
    """CUSUM along time, vectorized over envs. x, scored, reset: [N, T] (x: standardized score; scored: count
    this step; reset: episode start); spike [N, T]: impulse alarms (also reset g). Returns alarm [N, T] bool
    and g [N, T] (after the alarm reset)."""
    N, T = x.shape
    g = torch.zeros(N, dtype=x.dtype, device=x.device)
    alarm = torch.zeros(N, T, dtype=torch.bool, device=x.device)
    gs = torch.zeros(N, T, dtype=x.dtype, device=x.device)
    xs = torch.where(scored, x, torch.zeros_like(x))
    for u in range(T):
        g = torch.where(reset[:, u], torch.zeros_like(g), g)
        g = torch.where(scored[:, u], torch.clamp(g + xs[:, u] - k, min=0.0), g)
        a = g > h
        if spike is not None:
            a = a | (spike[:, u] & scored[:, u])
        alarm[:, u] = a
        g = torch.where(a, torch.zeros_like(g), g)
        gs[:, u] = g
    return alarm, gs


def cusum_alarm_counts(x: torch.Tensor, scored: torch.Tensor, reset: torch.Tensor, k: float, hs: torch.Tensor,
                       excluded: torch.Tensor, spike: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Alarm counts for many thresholds hs [M] in one pass: (alarms outside `excluded`, alarms inside) [M] each.
    `excluded` [N, T] marks steps whose alarms are not false alarms (e.g. within h steps before a fall)."""
    N, T = x.shape
    h = hs.to(x.dtype).reshape(-1, 1)
    g = torch.zeros(h.shape[0], N, dtype=x.dtype, device=x.device)
    n_false = torch.zeros(h.shape[0], dtype=torch.long, device=x.device)
    n_excl = torch.zeros_like(n_false)
    xs = torch.where(scored, x, torch.zeros_like(x))
    for u in range(T):
        g = torch.where(reset[:, u][None], torch.zeros_like(g), g)
        g = torch.where(scored[:, u][None], torch.clamp(g + xs[:, u][None] - k, min=0.0), g)
        a = g > h
        if spike is not None:
            a = a | (spike[:, u] & scored[:, u])[None]
        n_false += (a & ~excluded[:, u][None]).sum(1)
        n_excl += (a & excluded[:, u][None]).sum(1)
        g = torch.where(a, torch.zeros_like(g), g)
    return n_false, n_excl


def cusum_seq(x: np.ndarray, scored: np.ndarray, reset: np.ndarray, k: float, h: float, spike: np.ndarray | None = None):
    """Plain single-sequence CUSUM (what Unity runs), for the reference file: x, scored, reset, spike [T]."""
    g, out_g, out_a = 0.0, [], []
    spike = np.zeros(len(x), bool) if spike is None else spike
    for xu, su, ru, sp in zip(x, scored, reset, spike):
        if ru:
            g = 0.0
        if su:
            g = max(0.0, g + float(xu) - k)
        a = bool(su and (g > h or sp))
        if a:
            g = 0.0
        out_g.append(g)
        out_a.append(a)
    return np.array(out_g), np.array(out_a)
