"""Terrain-risk network: fully convolutional, valid (unpadded) dilated 3x3 convolutions.

Receptive field 17 x 17 (dilations 1, 1, 2, 4), so a 17 x 17 patch gives the risk of its centre cell and an
H x W window gives an (H - 16) x (W - 16) risk map, with identical per-cell results. The first layer's height
kernels are forced to sum to zero, which makes the output exactly invariant to a constant height offset (no
height reference needed). The condition (command + policy one-hot) enters as a bias of the first layer, which
under valid convolutions is the same as constant input channels. Platt parameters (scale, bias per horizon)
correct the probabilities for the positive oversampling used in training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

HEIGHT_SCALE = 10.0  # metres -> decimetres before the first layer (conditioning only)


class RiskNet(nn.Module):
    def __init__(self, width: int = 32, n_cond: int = 5, n_out: int = 2, dilations: tuple[int, ...] = (1, 1, 2, 4)):
        super().__init__()
        self.config = dict(width=width, n_cond=n_cond, n_out=n_out, dilations=list(dilations))
        self.conv1 = nn.Conv2d(1, width, 3, dilation=dilations[0])
        self.cond = nn.Linear(n_cond, width, bias=False)
        self.convs = nn.ModuleList(nn.Conv2d(width, width, 3, dilation=d) for d in dilations[1:])
        self.head = nn.Conv2d(width, n_out, 1)
        self.register_buffer("platt_scale", torch.ones(n_out))
        self.register_buffer("platt_bias", torch.zeros(n_out))
        self.rf = 1 + 2 * sum(dilations)

    def zero_sum_weight(self) -> torch.Tensor:
        w = self.conv1.weight
        return w - w.mean(dim=(2, 3), keepdim=True)

    def logits(self, height: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Uncalibrated logits [B, n_out, H - rf + 1, W - rf + 1] for height [B, 1, H, W], cond [B, n_cond]."""
        x = F.conv2d(height * HEIGHT_SCALE, self.zero_sum_weight(), self.conv1.bias, dilation=self.conv1.dilation)
        x = F.relu(x + self.cond(cond)[:, :, None, None])
        for conv in self.convs:
            x = F.relu(conv(x))
        return self.head(x)

    def forward(self, height: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Calibrated fall probabilities [B, n_out, H - rf + 1, W - rf + 1]."""
        z = self.logits(height, cond)
        return torch.sigmoid(z * self.platt_scale[None, :, None, None] + self.platt_bias[None, :, None, None])


def load_risk_model(path, device="cpu") -> tuple[RiskNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["model_config"])
    cfg["dilations"] = tuple(cfg["dilations"])
    model = RiskNet(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
