"""Export the world-model ensemble to ONNX for Unity, and write the Unity reference traces (W5).

The E members are unrolled into E independent Gemm chains (batch 1, static shapes), so the graph only uses
Gemm, Sigmoid, Mul, Add, Sub, Div, Exp, Softplus, ReduceMean, Gather, Concat, Slice and Unsqueeze.
Input "wm_in" [1, input_dim] in raw units (robot_interface.yaml world_model.input); outputs as in
world_model.outputs. Moment matching happens in Delta space (two-pass variance) before s_t is added.

  ~/miniforge3/envs/env_isaaclab/bin/python export/export_wm_onnx.py --run logs/wm/<run>
      -> export/models/go2_wm.onnx
  ... --reference experiments/w3_reference/go2_v2.npz [...]
      -> experiments/w3_reference/go2_wm_<name>.json: wm_in, all outputs, z, S, x, spike, g, alarm per step
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnx
import torch
import yaml
from torch import nn

from isaac.interface import INTERFACE_PATH, REPO_ROOT, load_interface
from models.world_model.ensemble import WorldModel, load_model
from models.world_model.layout import WMLayout, load_layout

MODELS_DIR = REPO_ROOT / "export" / "models"
OUTPUTS = ["state_mean", "state_var", "disp_mean", "disp_var", "fall_prob", "disagreement"]


def _linear(ens_layer, e: int) -> nn.Linear:
    w, b = ens_layer.weight[e], ens_layer.bias[e, 0]
    lin = nn.Linear(w.shape[0], w.shape[1])
    lin.weight.data.copy_(w.T)
    lin.bias.data.copy_(b)
    return lin


class WMOnnx(nn.Module):
    """WorldModel with the ensemble unrolled; forward(wm_in [1, input_dim]) -> OUTPUTS."""

    def __init__(self, model: WorldModel, layout: WMLayout):
        super().__init__()
        self.E, self.S, self.D, self.dyn = model.E, model.state_dim, model.out_dim, model.dyn_dim
        trunk_lin = [m for m in model.trunk if not isinstance(m, nn.SiLU)]
        fall_lin = [m for m in model.fall if not isinstance(m, nn.SiLU)]
        self.trunks = nn.ModuleList(
            nn.Sequential(*[mod for lin in trunk_lin for mod in (_linear(lin, e), nn.SiLU())]) for e in range(self.E))
        self.heads = nn.ModuleList(_linear(model.head, e) for e in range(self.E))
        self.falls = nn.ModuleList(nn.Sequential(_linear(fall_lin[0], e), nn.SiLU(), _linear(fall_lin[1], e)) for e in range(self.E))
        for name in ("in_mean", "in_std", "out_mean", "out_std"):
            self.register_buffer(name, getattr(model, name).detach().clone())
        self.register_buffer("max_lv", model.max_logvar.detach()[:, 0].clone())  # [E, D]
        self.register_buffer("min_lv", model.min_logvar.detach()[:, 0].clone())
        self.register_buffer("cur", torch.as_tensor(layout.current_state_in_input, dtype=torch.long))

    def forward(self, wm_in: torch.Tensor):
        x = (wm_in[:, : self.dyn] - self.in_mean) / self.in_std
        onehot = wm_in[:, self.dyn :]
        mus, vars_, falls = [], [], []
        for e in range(self.E):
            h = self.trunks[e](x)
            o = self.heads[e](h)
            mean, raw = o[:, : self.D], o[:, self.D :]
            lv = self.max_lv[e] - nn.functional.softplus(self.max_lv[e] - raw)
            lv = self.min_lv[e] + nn.functional.softplus(lv - self.min_lv[e])
            mus.append(mean * self.out_std + self.out_mean)
            vars_.append(torch.exp(lv) * self.out_std * self.out_std)
            falls.append(torch.sigmoid(self.falls[e](torch.cat([h, onehot], dim=1))))
        mu = torch.stack(mus)  # [E, 1, D]
        mbar = mu.mean(0)
        dev = mu - mbar
        epi = (dev * dev).mean(0)
        var = torch.stack(vars_).mean(0) + epi
        s_t = torch.index_select(wm_in, 1, self.cur)
        state_mean = s_t + mbar[:, : self.S]
        disagreement = (epi[:, : self.S] / (self.out_std[: self.S] * self.out_std[: self.S])).mean(1, keepdim=True)
        return state_mean, var[:, : self.S], mbar[:, self.S :], var[:, self.S :], torch.stack(falls).mean(0), disagreement


def export(run: str | Path, out: str | Path | None = None, opset: int = 18, checkpoint: str = "model.pt") -> Path:
    run = Path(run)
    model, ckpt = load_model(run / checkpoint, "cpu")
    layout = load_layout()
    if model.dyn_dim != layout.dyn_dim:
        raise ValueError(f"model input {model.dyn_dim} != yaml {layout.dyn_dim}")
    module = WMOnnx(model, layout).eval()
    out = Path(out) if out else MODELS_DIR / "go2_wm.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, layout.input_dim)
    torch.onnx.export(module, dummy, str(out), export_params=True, opset_version=opset, input_names=["wm_in"],
                      output_names=OUTPUTS, dynamic_axes={}, dynamo=False)
    m = onnx.load(out)
    itf = load_interface()
    meta = {
        "kind": "world_model",
        "run": os.path.relpath(run, REPO_ROOT),
        "checkpoint": checkpoint,
        "iter": str(ckpt["iter"]),
        "members": str(model.E),
        "input_dim": str(layout.input_dim),
        "state_dim": str(layout.state_dim),
        "robot_interface_sha256": hashlib.sha256(INTERFACE_PATH.read_bytes()).hexdigest(),
        "world_model_spec_sha256": hashlib.sha256(yaml.safe_dump(
            {k: v for k, v in itf["world_model"].items() if k != "surprisal"}, sort_keys=True).encode()).hexdigest(),
    }
    del m.metadata_props[:]
    for k, v in meta.items():
        m.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(m)
    onnx.save(m, out)
    return out


# --- Unity reference traces ----------------------------------------------------------------------------
def build_wm_input(ref: dict, itf: dict, policy: str) -> np.ndarray:
    """wm_in for every step of a reference rollout, from its raw physics state (yaml only), as Unity builds it."""
    from export.test_parity import history_block, obs_parts

    parts = obs_parts(ref, itf)
    n = len(ref["root_link_quat_w"])
    blocks = []
    for b in itf["world_model"]["input"]:
        if b["name"] == "history":
            blocks += [history_block(parts[t], b["history_length"]) for t in b["terms"]]
        elif b["name"] == "action":
            blocks.append(ref["actions"])
        elif b["name"] == "height_scan":
            blocks.append(parts["height_scan"])
        elif b["name"] == "policy_onehot":
            oh = np.zeros((n, len(itf["world_model"]["policies"])), np.float32)
            oh[:, itf["world_model"]["policies"].index(policy)] = 1.0
            blocks.append(oh)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def surprisal_trace(state: np.ndarray, outputs: dict, itf: dict, policy: str, episode_step: np.ndarray) -> dict:
    """Per-step z, S, x, spike, g, alarm following world_model.surprisal (frame u scored against prediction u-1)."""
    from models.world_model.surprisal import cusum_seq

    sp = itf["world_model"]["surprisal"]
    w = np.asarray(sp["weights"])
    floor = np.asarray(sp["var_floor"])
    n = len(state)
    z = np.zeros_like(state)
    S = np.zeros(n)
    z[1:] = ((state[1:] - outputs["state_mean"][:-1]) / np.sqrt(outputs["state_var"][:-1] + floor) - np.asarray(sp["z_bias"])) / np.asarray(sp["z_scale"])
    z2 = np.minimum(z * z, sp["clip_z2"])
    if sp["statistic"] == "norm":
        per = 0.5 * (z2 - 1.0)
    else:
        var = np.ones_like(state)
        var[1:] = outputs["state_var"][:-1]
        per = 0.5 * (z2 + np.log(2 * np.pi * (var + floor)))
    S[1:] = (per[1:] * w).sum(1)
    pp = sp["per_policy"][policy]
    x = (S - pp["m0"]) / pp["s0"]
    scored = (np.arange(n) >= 1) & (episode_step >= sp["warmup_steps"])
    reset = episode_step == 0
    spike = np.zeros(n, bool)
    spike[1:] = np.abs(z[1:, sp["spike_dims"]]).max(1) > pp["spike_z"]
    g, alarm = cusum_seq(x, scored, reset, pp["k"], pp["h"], spike)
    return {"z": z, "S": S, "x": x, "scored": scored, "spike": spike & scored, "g": g, "alarm": alarm}


def write_reference(onnx_path: Path, npz: Path, out: Path, policy: str = "v2") -> dict:
    import onnxruntime as ort

    itf = load_interface()
    ref = dict(np.load(npz))
    if ref["dones"].any():
        raise ValueError(f"{npz} resets mid-rollout")
    wm_in = build_wm_input(ref, itf, policy)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rows = [sess.run(OUTPUTS, {"wm_in": x[None]}) for x in wm_in]
    outputs = {k: np.concatenate([r[i] for r in rows]) for i, k in enumerate(OUTPUTS)}
    lay = load_layout(itf)
    state = wm_in[:, lay.current_state_in_input]
    episode_step = np.arange(len(wm_in))  # one episode from step 0
    tr = surprisal_trace(state, outputs, itf, policy, episode_step)

    def arr(a):
        a = np.asarray(a)
        return {"shape": list(a.shape), "data": a.astype(np.float64 if a.dtype != bool else np.int32).ravel().tolist()}

    doc = {
        "source": os.path.relpath(npz, REPO_ROOT),
        "model": os.path.relpath(onnx_path, REPO_ROOT),
        "policy": policy,
        "steps": len(wm_in),
        "friction_drop": {"at_step": int(ref["friction_drop_step"]), "to": float(ref["friction_drop_to"])} if "friction_drop_step" in ref else None,
        "wm_in": arr(wm_in),
        **{k: arr(v) for k, v in outputs.items()},
        "episode_step": arr(episode_step),
        **{k: arr(v) for k, v in tr.items()},
    }
    out.write_text(json.dumps(doc, separators=(",", ":")))
    return {"alarms": [int(i) for i in np.nonzero(tr["alarm"])[0]], "max_g": float(tr["g"].max()), "max_x": float(tr["x"][tr["scored"]].max())}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="logs/wm/<run>; omit to only write references for the existing model")
    p.add_argument("--checkpoint", default="model.pt", help="model.pt, or model_fall.pt after refit_fall")
    p.add_argument("--out", default=None, help="default export/models/go2_wm.onnx")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--reference", nargs="*", default=[], help="reference .npz rollouts to trace (policy v2)")
    args = p.parse_args()
    path = Path(args.out) if args.out else MODELS_DIR / "go2_wm.onnx"
    if args.run:
        path = export(args.run, path, args.opset, args.checkpoint)
        print(f"[INFO] exported {path} ({path.stat().st_size / 2**20:.1f} MB)")
    for npz in args.reference:
        npz = Path(npz)
        name = npz.stem.replace("go2_v2", "go2_wm")
        res = write_reference(path, npz, npz.with_name(f"{name}.json"))
        print(f"[INFO] {npz.with_name(f'{name}.json')}: {res}")
