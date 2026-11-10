"""Latent world model (JEPA-style) for surprisal: the second world model, compared against the W4-5 ensemble.

    z_u   = E(s_{u-L+1..u})            frame encoder on the last L states (33-d each), u's window filled with the
                                       episode's first frame before its start (as the ensemble's history)
    z̄_u   = Ē(s_{u-L+1..u})            target encoder = EMA of E, no gradient
    ẑ_{t+1} = z_t + P_m(z_{t-H+1..t}, prev actions_{t-H+1..t}, a_t, height_scan_t)    member m of M heads

P sees the same information as the ensemble's wm_in (H = 6 frames, the action, the scan). Each member has a
3 x 512 trunk (as the ensemble); heads on it:
    mean    latent delta (trained by the JEPA prediction loss; gradients reach E through the context)
    logvar  aleatoric variance of ẑ, on the DETACHED trunk features (Gaussian NLL with the mean detached), so the
            uncertainty head cannot shape the representation
    disp    base displacement (dx, dy, dz, dyaw), detached features: ghost path
    fall    fall logits for world_model.fall_horizons_steps, detached features + policy one-hot (refit later)
Anti-collapse (`reg`):
    vicreg  VICReg variance hinge + covariance penalty on E's outputs; no output normalization
    ema     EMA target only (BYOL-style asymmetry); LayerNorm without affine on E's output fixes the scale
    recon   (objective C, not JEPA) a decoder reconstructs the window from z_t and the next state from ẑ_{t+1};
            output LayerNorm as `ema`
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from models.world_model.ensemble import EnsembleLinear

REGS = ("vicreg", "ema", "recon")


def mlp(n_in: int, hidden: tuple[int, ...], n_out: int) -> nn.Sequential:
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...], d: int, out_norm: bool):
        super().__init__()
        self.net = mlp(n_in, hidden, d)
        self.norm = nn.LayerNorm(d, elementwise_affine=False) if out_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class LatentWM(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, scan_dim: int, n_policy: int, H: int = 6, L: int = 3, d: int = 128,
                 M: int = 5, reg: str = "vicreg", enc_hidden: tuple[int, ...] = (512, 512, 512),
                 hidden: tuple[int, ...] = (512, 512, 512), n_fall: int = 3, fall_hidden: int = 128, disp_dim: int = 4):
        super().__init__()
        if reg not in REGS:
            raise ValueError(reg)
        self.config = dict(state_dim=state_dim, act_dim=act_dim, scan_dim=scan_dim, n_policy=n_policy, H=H, L=L, d=d, M=M,
                           reg=reg, enc_hidden=list(enc_hidden), hidden=list(hidden), n_fall=n_fall,
                           fall_hidden=fall_hidden, disp_dim=disp_dim)
        self.S, self.A, self.H, self.L, self.d, self.M, self.reg = state_dim, act_dim, H, L, d, M, reg
        self.encoder = Encoder(L * state_dim, tuple(enc_hidden), d, out_norm=reg != "vicreg")
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.in_dim = H * d + H * act_dim + act_dim + scan_dim
        layers, n = [], self.in_dim
        for h in hidden:
            layers += [EnsembleLinear(M, n, h), nn.SiLU()]
            n = h
        self.trunk = nn.Sequential(*layers)
        self.feat_dim = n
        self.head_mean = EnsembleLinear(M, n, d)
        self.head_var = EnsembleLinear(M, n, d)
        self.head_disp = EnsembleLinear(M, n, disp_dim)
        self.fall = nn.Sequential(EnsembleLinear(M, n + n_policy, fall_hidden), nn.SiLU(), EnsembleLinear(M, fall_hidden, n_fall))
        nn.init.zeros_(self.head_mean.weight)  # start as "the latent stays"
        nn.init.zeros_(self.head_mean.bias)
        self.max_logvar = nn.Parameter(torch.full((M, 1, d), 0.5))
        self.min_logvar = nn.Parameter(torch.full((M, 1, d), -10.0))
        if reg == "recon":
            self.decoder = mlp(d, (512, 512), L * state_dim)
            self.decoder_next = mlp(d, (512, 512), state_dim)
        for name, dim in (("s", state_dim), ("a", act_dim), ("scan", scan_dim), ("disp", disp_dim)):
            self.register_buffer(f"{name}_mean", torch.zeros(dim))
            self.register_buffer(f"{name}_std", torch.ones(dim))
        # set after training (models/world_model_jepa/train.py): per-dim floor of the predictive variance,
        # (0.1 * std of the 1-step target-latent delta)^2, as the ensemble's var_floor; and the whitening
        # matrix of the standardized 1-step residual (maha weights)
        self.register_buffer("var_floor", torch.zeros(d))
        self.register_buffer("whiten", torch.eye(d))

    # --- encoders ---------------------------------------------------------------------------------------
    def norm_states(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.s_mean) / self.s_std

    def encode(self, win: torch.Tensor, target: bool = False) -> torch.Tensor:
        """win [..., L, S] raw states, oldest first -> [..., d]."""
        x = self.norm_states(win).flatten(-2)
        return self.target(x) if target else self.encoder(x)

    @staticmethod
    def windows(seq: torch.Tensor, L: int) -> torch.Tensor:
        """seq [B, W, S] -> [B, W - L + 1, L, S]: window j covers seq[j .. j + L - 1]."""
        return seq.unfold(1, L, 1).transpose(-1, -2)

    @torch.no_grad()
    def ema_update(self, tau: float):
        for pt, po in zip(self.target.parameters(), self.encoder.parameters()):
            pt.mul_(tau).add_(po.detach(), alpha=1.0 - tau)
        for bt, bo in zip(self.target.buffers(), self.encoder.buffers()):
            bt.copy_(bo)

    # --- predictor ------------------------------------------------------------------------------------------
    def predictor_input(self, z_hist: torch.Tensor, prev_act: torch.Tensor, a: torch.Tensor, scan: torch.Tensor) -> torch.Tensor:
        """z_hist [..., H, d], prev_act [..., H, A] (raw), a [..., A] (raw), scan [..., scan_dim] (raw)."""
        return torch.cat([z_hist.flatten(-2), ((prev_act - self.a_mean) / self.a_std).flatten(-2),
                          (a - self.a_mean) / self.a_std, (scan - self.scan_mean) / self.scan_std], dim=-1)

    def step(self, x: torch.Tensor, z_last: torch.Tensor):
        """x [M, B, in_dim], z_last [M, B, d] -> mean [M, B, d], logvar [M, B, d], features [M, B, feat]."""
        h = self.trunk(x)
        mean = z_last + self.head_mean(h)
        lv = self.head_var(h.detach())
        lv = self.max_logvar - F.softplus(self.max_logvar - lv)
        lv = self.min_logvar + F.softplus(lv - self.min_logvar)
        return mean, lv, h

    def disp(self, h: torch.Tensor) -> torch.Tensor:
        """Normalized displacement prediction from (detached) trunk features."""
        return self.head_disp(h.detach())

    def fall_logits(self, h: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        return self.fall(torch.cat([h.detach(), onehot], dim=-1))

    @staticmethod
    def moments(mean: torch.Tensor, lv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Member means / log-variances [M, B, d] -> moment-matched mean, variance [B, d]."""
        mbar = mean.mean(0)
        return mbar, lv.exp().mean(0) + ((mean - mbar) ** 2).mean(0)


# --- anti-collapse terms and diagnostics -----------------------------------------------------------------
def vicreg_terms(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """z [B, d] -> (variance hinge: mean over dims of relu(gamma - std), covariance: sum of squared off-diagonal
    covariances / d) (Bardes et al. 2022)."""
    zc = z - z.mean(0)
    std = torch.sqrt(zc.var(0) + eps)
    var_loss = F.relu(gamma - std).mean()
    B, d = z.shape
    cov = (zc.T @ zc) / (B - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_loss, off.pow(2).sum() / d


@torch.no_grad()
def effective_rank(z: torch.Tensor) -> float:
    """exp(entropy of the normalized singular values of the centred batch) (Roy & Vetterli 2007)."""
    zc = (z - z.mean(0)).float()
    s = torch.linalg.svdvals(zc)
    p = s / s.sum().clamp(min=1e-12)
    return float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum()))


def load_model(path, device="cpu") -> tuple[LatentWM, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["model_config"])
    for k in ("enc_hidden", "hidden"):
        cfg[k] = tuple(cfg[k])
    model = LatentWM(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def cosine_tau(it: int, total: int, tau0: float) -> float:
    """EMA momentum, cosine from tau0 to 1 over training (BYOL)."""
    return 1.0 - (1.0 - tau0) * (math.cos(math.pi * min(1.0, it / max(1, total))) + 1.0) / 2.0
