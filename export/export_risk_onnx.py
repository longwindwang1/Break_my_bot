"""Export the terrain-risk network to ONNX for Unity (W6).

Inputs "height" [1, 1, H, W] (terrain z at the grid points of configs/robot_interface.yaml terrain_risk, x fastest;
H and W dynamic), "ice" [1, 1, H, W] (ice models: 1 where the point is on visible ice, else 0; same points) and
"cond" [1, 5] (vx, vy, wz, policy one-hot); output "risk" [1, 2, H - 16, W - 16], calibrated probabilities of a fall
within 25 / 50 steps. The zero-sum first-layer kernel and the Platt scaling are baked in. With ice all zero the ice
model is a height-only model. Ops: Conv, Gemm, Unsqueeze, Add, Relu, Mul, Sigmoid.

  ~/miniforge3/envs/env_isaaclab/bin/python export/export_risk_onnx.py --run logs/terrain_risk/<run>
      -> export/models/go2_terrain_risk.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnx
import torch
from torch import nn

from isaac.interface import INTERFACE_PATH, REPO_ROOT
from models.terrain_risk.model import HEIGHT_SCALE, RiskNet, load_risk_model
from models.terrain_risk.patches import load_risk_layout

OUT = REPO_ROOT / "export" / "models" / "go2_terrain_risk.onnx"


class RiskOnnx(nn.Module):
    def __init__(self, model: RiskNet):
        super().__init__()
        c1 = model.conv1
        self.conv1 = nn.Conv2d(1, c1.out_channels, 3, dilation=c1.dilation)
        self.conv1.weight.data.copy_(model.zero_sum_weight().detach() * HEIGHT_SCALE)
        self.conv1.bias.data.copy_(c1.bias.detach())
        self.has_ice = model.has_ice
        if self.has_ice:
            self.conv_ice = nn.Conv2d(1, c1.out_channels, 3, dilation=c1.dilation, bias=False)
            self.conv_ice.weight.data.copy_(model.conv_ice.weight.detach())
        self.cond = nn.Linear(model.cond.in_features, model.cond.out_features, bias=False)
        self.cond.weight.data.copy_(model.cond.weight.detach())
        self.convs = nn.ModuleList()
        for c in model.convs:
            n = nn.Conv2d(c.in_channels, c.out_channels, 3, dilation=c.dilation)
            n.load_state_dict(c.state_dict())
            self.convs.append(n)
        self.head = nn.Conv2d(model.head.in_channels, model.head.out_channels, 1)
        self.head.load_state_dict(model.head.state_dict())
        self.register_buffer("scale", model.platt_scale.detach().clone().reshape(1, -1, 1, 1))
        self.register_buffer("bias", model.platt_bias.detach().clone().reshape(1, -1, 1, 1))

    def forward(self, height: torch.Tensor, ice: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.conv1(height)
        if self.has_ice:
            x = x + self.conv_ice(ice)
        x = torch.relu(x + self.cond(cond).unsqueeze(-1).unsqueeze(-1))
        for c in self.convs:
            x = torch.relu(c(x))
        return torch.sigmoid(self.head(x) * self.scale + self.bias)


def export(run: str | Path, out: str | Path = OUT, opset: int = 18) -> Path:
    run, out = Path(run), Path(out)
    model, ckpt = load_risk_model(run / "model.pt", "cpu")
    lay = load_risk_layout()
    module = RiskOnnx(model).eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    h = torch.zeros(1, 1, lay.ny, lay.nx)
    cond = torch.zeros(1, len(lay.cond))
    if model.has_ice:
        args, names = (h, torch.zeros_like(h), cond), ["height", "ice", "cond"]
        dyn = {"height": {2: "H", 3: "W"}, "ice": {2: "H", 3: "W"}, "risk": {2: "H_out", 3: "W_out"}}
    else:  # phase-1 height-only model
        module.forward = lambda height, cond: RiskOnnx.forward(module, height, None, cond)
        args, names = (h, cond), ["height", "cond"]
        dyn = {"height": {2: "H", 3: "W"}, "risk": {2: "H_out", 3: "W_out"}}
    torch.onnx.export(module, args, str(out), export_params=True, opset_version=opset, input_names=names,
                      output_names=["risk"], dynamic_axes=dyn, dynamo=False)
    m = onnx.load(out)
    meta = {"kind": "terrain_risk", "run": os.path.relpath(run, REPO_ROOT), "iter": str(ckpt["iter"]),
            "receptive_field": str(model.rf), "horizons_steps": ",".join(str(v) for v in lay.horizons),
            "ice_input": str(int(model.has_ice)),
            "robot_interface_sha256": hashlib.sha256(INTERFACE_PATH.read_bytes()).hexdigest()}
    del m.metadata_props[:]
    for k, v in meta.items():
        m.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(m)
    onnx.save(m, out)
    return out


def write_reference(onnx_path: Path, split_dir: Path, out: Path, n_windows: int = 8, seed: int = 0) -> dict:
    """A few egocentric windows from a recorded geo / ice split, with the ONNX risk maps and the slowdown decision
    (robot_interface.yaml terrain_risk.slowdown), for checking Unity's window builder / ONNX / trigger. From 600
    candidates (half 0.5 s before a fall, half random) the windows cover, in turn: a v1 ice-excess trigger, a v2
    total trigger, visible ice without a trigger, and no ice; then repeat."""
    import json

    import onnxruntime as ort

    from isaac.interface import load_interface
    from models.terrain_risk.dataset import RiskFrames
    from models.terrain_risk.evaluate import corridor_mask
    from models.terrain_risk.patches import cond_vector, ice_channel, window

    lay = load_risk_layout()
    slow = load_interface()["terrain_risk"]["slowdown"]
    fr = RiskFrames(split_dir, device="cpu", val_frac=0.0)
    hm, im = fr.maps[0], fr.ice_maps[0]
    rng = np.random.default_rng(seed)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    has_ice = "ice" in {i.name for i in sess.get_inputs()}
    pos = (fr.label[:, 0] & fr.known[:, 0]).nonzero().squeeze(1)  # frames 0.5 s before a fall
    cand = torch.cat([pos[torch.from_numpy(rng.choice(len(pos), min(300, len(pos)), replace=False))],
                      torch.from_numpy(rng.choice(fr.n_frames, 300, replace=False))])

    def evaluate(i: int) -> dict:
        i = torch.tensor([i])
        h = window(hm, lay, fr.x[i], fr.y[i], fr.yaw[i])
        ice = ice_channel(window(im, lay, fr.x[i], fr.y[i], fr.yaw[i]))
        cmd = fr.cmd[i]
        cond = cond_vector(cmd, fr.policy[i])
        feed = {"height": h.numpy().astype(np.float32), "cond": cond.numpy().astype(np.float32)}
        if has_ice:
            feed["ice"] = ice.numpy().astype(np.float32)
        risk = sess.run(["risk"], feed)[0]
        risk_no_ice = sess.run(["risk"], {**feed, "ice": np.zeros_like(feed["ice"])})[0] if has_ice else risk
        m = corridor_mask(cmd, lay)[0].numpy()
        ra = float(np.where(m, risk[0, 0], 0.0).max())
        ra_ex = float(np.where(m, risk[0, 0] - risk_no_ice[0, 0], 0.0).max())
        pol = ["v1", "v2"][int(fr.policy[i])]
        kind, thr = slow["trigger"][pol], slow["threshold"][pol]
        value = ra if kind == "total" else ra_ex
        return {"height": h[0, 0].numpy().astype(float).tolist(), "ice": ice[0, 0].numpy().astype(int).tolist(),
                "cond": cond[0].numpy().astype(float).tolist(),
                "policy": pol, "risk": risk[0].astype(float).tolist(), "risk_no_ice": risk_no_ice[0].astype(float).tolist(),
                "corridor_mask": m.astype(int).tolist(), "risk_ahead": ra, "risk_ahead_ice_excess": ra_ex,
                "trigger": kind, "threshold": thr, "slow_down": bool(thr is not None and value > thr),
                "fall_within_25_steps": bool(fr.label[i, 0])}

    cases = [evaluate(int(i)) for i in cand]
    kinds = [
        lambda w: w["slow_down"] and w["trigger"] == "ice_excess",
        lambda w: w["slow_down"] and w["trigger"] == "total",
        lambda w: not w["slow_down"] and any(map(any, w["ice"])),
        lambda w: not w["slow_down"] and not any(map(any, w["ice"])),
    ]
    windows, used = [], set()
    while len(windows) < n_windows:
        before = len(windows)
        for want in kinds:
            k = next((j for j, w in enumerate(cases) if j not in used and want(w)), None)
            if k is not None and len(windows) < n_windows:
                used.add(k)
                windows.append(cases[k])
        if len(windows) == before:  # nothing left of any kind
            windows += [w for j, w in enumerate(cases) if j not in used][: n_windows - len(windows)]
            break
    doc = {"model": os.path.relpath(onnx_path, REPO_ROOT), "source": os.path.relpath(split_dir, REPO_ROOT),
           "layout": {"nx": lay.nx, "ny": lay.ny, "x0": lay.x0, "y0": lay.y0, "res": lay.res},
           "note": "height[j][i] and ice[j][i] (1 = visible ice) at (x0 + res*i, y0 + res*j) in the base yaw frame; risk[c][j][i] "
                   "for output cell centred at (x0 + res*(i+8), y0 + res*(j+8)); risk_no_ice = same window with ice all 0; "
                   "corridor_mask uses cond[0:2]; risk_ahead(_ice_excess) = max over the corridor of risk[0] "
                   "(- risk_no_ice[0]), floored at 0; slow_down = the policy's trigger value > threshold",
           "windows": windows}
    out.write_text(json.dumps(doc))
    return {"windows": len(windows), "slow_down": sum(w["slow_down"] for w in windows),
            "with_ice": sum(any(map(any, w["ice"])) for w in windows)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--reference_split", default=None, help="geo split dir to cut reference windows from")
    a = p.parse_args()
    path = export(a.run, a.out, a.opset)
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"[INFO] exported {path} ({path.stat().st_size / 1024:.0f} KB); ops:",
          sorted({n.op_type for n in onnx.load(path).graph.node}), "inputs", [(i.name, i.shape) for i in s.get_inputs()])
    if a.reference_split:
        ref = REPO_ROOT / "experiments" / "w3_reference" / "go2_terrain_risk.json"
        print(f"[INFO] {ref}: {write_reference(path, Path(a.reference_split), ref)}")
