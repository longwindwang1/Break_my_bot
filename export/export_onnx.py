"""Export a trained RSL-RL Go2 actor to ONNX without launching Isaac Sim.

The checkpoint is loaded into rsl-rl's own ActorCritic (strict state_dict load), and the ONNX file is
written by Isaac Lab's exporter (isaaclab_rl/rsl_rl/exporter.py, imported by file path so that the
isaaclab_rl package -- which pulls in Isaac Sim -- is never imported). Input "obs" [1, obs_dim],
output "actions" [1, 12]; metadata records the task, checkpoint and robot_interface.yaml hash.

For v2 (deploy: student) the distilled student is exported; its input is the concatenation of the
deployed observation groups (robot_interface.yaml policies.v2.groups).

Usage (repo root):
    ~/miniforge3/envs/env_isaaclab/bin/python export/export_onnx.py \
        --run logs/rsl_rl/unitree_go2_flat/2026-10-12_21-19-01_w1_go2_flat_baseline --variant flat
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `isaac.interface` when run as a script

# rsl_rl imports GitPython at import time; we never use git here, so don't fail if it's not on PATH.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import onnx
import torch
from rsl_rl.modules import ActorCritic, StudentTeacher
from tensordict import TensorDict

from isaac.interface import INTERFACE_PATH, REPO_ROOT, detect_variant, load_env_yaml, load_interface, policy_obs_dim

ISAACLAB_EXPORTER = Path.home() / "IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py"
MODELS_DIR = REPO_ROOT / "export" / "models"


def latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(re.search(r"model_(\d+)", p.name).group(1)))
    if not ckpts:
        raise FileNotFoundError(f"no model_*.pt in {run_dir}")
    return ckpts[-1]


def load_actor_critic(run_dir: str | Path, variant: str, checkpoint: str | None = None) -> tuple[ActorCritic, Path]:
    """Rebuild the rsl-rl ActorCritic described by robot_interface.yaml and load the checkpoint into it."""
    run_dir = Path(run_dir)
    ckpt_path = run_dir / checkpoint if checkpoint else latest_checkpoint(run_dir)
    spec = load_interface()["policies"][variant]
    obs_dim = spec["obs_dim"]
    # Same (actor == critic) obs layout as the stock Go2 tasks; only the shapes matter here.
    obs = TensorDict({"policy": torch.zeros(1, obs_dim)}, batch_size=[1])
    policy = ActorCritic(
        obs,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=len(load_interface()["joint_names"]),
        actor_obs_normalization=spec["obs_normalization"],
        critic_obs_normalization=spec["obs_normalization"],
        actor_hidden_dims=spec["actor_hidden_dims"],
        critic_hidden_dims=spec["actor_hidden_dims"],
        activation=spec["activation"],
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy, ckpt_path


def load_student_teacher(run_dir: str | Path, variant: str, checkpoint: str | None = None) -> tuple[StudentTeacher, Path]:
    """Rebuild the rsl-rl StudentTeacher of a distillation run (deployed part: .student).

    The student's input is fed as a single "policy" tensor holding the concatenated deployed groups
    (policies.<variant>.groups in robot_interface.yaml)."""
    run_dir = Path(run_dir)
    ckpt_path = run_dir / checkpoint if checkpoint else latest_checkpoint(run_dir)
    itf = load_interface()
    spec = itf["policies"][variant]
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    teacher_dim = state["teacher.0.weight"].shape[1]
    obs = TensorDict(
        {"policy": torch.zeros(1, policy_obs_dim(itf, variant)), "teacher": torch.zeros(1, teacher_dim)}, batch_size=[1]
    )
    policy = StudentTeacher(
        obs,
        obs_groups={"policy": ["policy"], "teacher": ["teacher"]},
        num_actions=len(itf["joint_names"]),
        student_obs_normalization=spec["obs_normalization"],
        teacher_obs_normalization=spec["obs_normalization"],
        student_hidden_dims=spec["actor_hidden_dims"],
        teacher_hidden_dims=spec["actor_hidden_dims"],
        activation=spec["activation"],
    )
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy, ckpt_path


def load_policy(run_dir: str | Path, variant: str, checkpoint: str | None = None):
    """(policy, checkpoint path); call policy.act_inference(TensorDict({"policy": obs})) for actions."""
    if load_interface()["policies"][variant].get("deploy") == "student":
        return load_student_teacher(run_dir, variant, checkpoint)
    return load_actor_critic(run_dir, variant, checkpoint)


def _isaaclab_exporter_module():
    spec = importlib.util.spec_from_file_location("isaaclab_rsl_rl_exporter", ISAACLAB_EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _export_with_opset(policy, out: Path, opset: int) -> None:
    """Same module as Isaac Lab's exporter (opset 18 hard-coded there), exported at another opset."""
    module = _isaaclab_exporter_module()._OnnxPolicyExporter(policy, normalizer=None)
    module.to("cpu").eval()
    obs = torch.zeros(1, module.actor[0].in_features)
    torch.onnx.export(
        module, obs, str(out), export_params=True, opset_version=opset,
        input_names=["obs"], output_names=["actions"], dynamic_axes={},
    )


def export(
    run_dir: str | Path,
    variant: str,
    checkpoint: str | None = None,
    out: str | Path | None = None,
    opset: int | None = None,
) -> Path:
    run_dir = Path(run_dir)
    itf = load_interface()
    env = load_env_yaml(run_dir / "params")
    task_cfg = itf["policies"][variant]
    if detect_variant(env["observations"], itf) != variant:
        raise ValueError(f"run {run_dir} does not look like a {variant!r} run")

    policy, ckpt_path = load_policy(run_dir, variant, checkpoint)
    out = Path(out) if out else MODELS_DIR / f"go2_{variant}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    if opset is None:
        _isaaclab_exporter_module().export_policy_as_onnx(policy, path=str(out.parent), normalizer=None, filename=out.name)
    else:
        _export_with_opset(policy, out, opset)

    model = onnx.load(out)
    meta = {
        "task": task_cfg["task"],
        "variant": variant,
        "checkpoint": os.path.relpath(ckpt_path, REPO_ROOT),
        "obs_dim": str(policy_obs_dim(itf, variant)),
        "action_dim": str(len(itf["joint_names"])),
        "robot_interface_sha256": hashlib.sha256(INTERFACE_PATH.read_bytes()).hexdigest(),
    }
    del model.metadata_props[:]
    for k, v in meta.items():
        model.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(model)
    onnx.save(model, out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Isaac Lab run dir (logs/rsl_rl/<exp>/<time>_<name>).")
    parser.add_argument("--variant", required=True, choices=list(load_interface()["policies"]))
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file name; default: latest model_*.pt.")
    parser.add_argument("--out", default=None, help="Output path; default: export/models/go2_<variant>.onnx.")
    parser.add_argument(
        "--opset", type=int, default=None, help="ONNX opset (default 18, Isaac Lab's). Older Unity Inference Engine: 15."
    )
    args = parser.parse_args()
    path = export(args.run, args.variant, args.checkpoint, args.out, args.opset)
    print(f"[INFO] Exported {path}")
