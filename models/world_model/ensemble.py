"""Ensemble of Gaussian MLP dynamics models (PETS-style) with a fall-probability head.

Each of the E members maps wm_in (minus the policy one-hot) through a 3 x 512 SiLU trunk to the mean and
log-variance of (Delta s, base displacement), normalized; a small head on the detached trunk features plus
the policy one-hot gives fall logits for the horizons in world_model.fall_horizons_steps.
Log-variances are soft-clamped between learned per-dim bounds (Chua et al. 2018).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class EnsembleLinear(nn.Module):
    """E independent linear layers applied to x [E, B, n_in] -> [E, B, n_out]."""

    def __init__(self, E: int, n_in: int, n_out: int):
        super().__init__()
        bound = 1.0 / math.sqrt(n_in)  # same as nn.Linear's default init
        self.weight = nn.Parameter(torch.empty(E, n_in, n_out).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(E, 1, n_out).uniform_(-bound, bound))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.baddbmm(self.bias, x, self.weight)


class WorldModel(nn.Module):
    def __init__(self, dyn_dim: int, n_policy: int, state_dim: int, disp_dim: int = 4, n_fall: int = 3,
                 E: int = 5, hidden: tuple[int, ...] = (512, 512, 512), fall_hidden: int = 128):
        super().__init__()
        self.config = dict(dyn_dim=dyn_dim, n_policy=n_policy, state_dim=state_dim, disp_dim=disp_dim, n_fall=n_fall,
                           E=E, hidden=list(hidden), fall_hidden=fall_hidden)
        self.E, self.dyn_dim, self.state_dim, self.out_dim = E, dyn_dim, state_dim, state_dim + disp_dim
        layers, d = [], dyn_dim
        for h in hidden:
            layers += [EnsembleLinear(E, d, h), nn.SiLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head = EnsembleLinear(E, d, 2 * self.out_dim)
        self.fall = nn.Sequential(EnsembleLinear(E, d + n_policy, fall_hidden), nn.SiLU(), EnsembleLinear(E, fall_hidden, n_fall))
        self.max_logvar = nn.Parameter(torch.full((E, 1, self.out_dim), 0.5))
        self.min_logvar = nn.Parameter(torch.full((E, 1, self.out_dim), -10.0))
        self.register_buffer("in_mean", torch.zeros(dyn_dim))
        self.register_buffer("in_std", torch.ones(dyn_dim))
        self.register_buffer("out_mean", torch.zeros(self.out_dim))  # normalization of (Delta s, disp)
        self.register_buffer("out_std", torch.ones(self.out_dim))

    def forward(self, wm_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """wm_in [E, B, input_dim] (raw) -> normalized mean, log-variance [E, B, out_dim]; fall logits [E, B, n_fall]."""
        x = (wm_in[..., : self.dyn_dim] - self.in_mean) / self.in_std
        h = self.trunk(x)
        mean, raw_lv = self.head(h).split(self.out_dim, dim=-1)
        lv = self.max_logvar - F.softplus(self.max_logvar - raw_lv)
        lv = self.min_logvar + F.softplus(lv - self.min_logvar)
        fall = self.fall(torch.cat([h.detach(), wm_in[..., self.dyn_dim :]], dim=-1))
        return mean, lv, fall

    def denormalize(self, mean: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return mean * self.out_std + self.out_mean, logvar.exp() * self.out_std**2

    def predictive(self, wm_in_1: torch.Tensor) -> dict[str, torch.Tensor]:
        """Moment-matched ensemble prediction for wm_in [B, input_dim] (all members see the same input).
        Returns raw-unit tensors: delta_mean, delta_var (aleatoric + epistemic), epistemic [B, out_dim];
        fall_prob [B, n_fall]."""
        x = wm_in_1.unsqueeze(0).expand(self.E, -1, -1)
        mean, lv, fall = self(x)
        mu, var = self.denormalize(mean, lv)
        mbar = mu.mean(0)
        epi = ((mu - mbar) ** 2).mean(0)
        return {"delta_mean": mbar, "delta_var": var.mean(0) + epi, "epistemic": epi,
                "fall_prob": torch.sigmoid(fall).mean(0), "member_mean": mu}


def load_model(path, device="cpu") -> tuple[WorldModel, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["model_config"])
    cfg["hidden"] = tuple(cfg["hidden"])
    model = WorldModel(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
