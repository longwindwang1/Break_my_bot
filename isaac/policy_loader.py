"""Load trained locomotion policies as plain torch MLPs (no rsl-rl or Isaac Sim imports).

Policy kinds:
  v1       PPO actor on obs["policy"]                          (W1 baseline, 235-d)
  teacher  PPO actor on obs["teacher"]                         (v2 stage A; privileged, reference only)
  student  StudentTeacher student on policy + proprio_history  (v2 stage B; deployable, 1360-d)
"""

from __future__ import annotations

import torch

PREFIX = {"v1": "actor", "teacher": "actor", "student": "student"}


def load_mlp(checkpoint: str, prefix: str) -> torch.nn.Sequential:
    """Rebuild an rsl-rl MLP (Linear/ELU alternating) from the `<prefix>.<i>.weight` entries of a checkpoint."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    idx = sorted({int(k.split(".")[1]) for k in state if k.startswith(prefix + ".") and k.endswith(".weight")})
    layers: list[torch.nn.Module] = []
    for n, i in enumerate(idx):
        w, b = state[f"{prefix}.{i}.weight"], state[f"{prefix}.{i}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if n < len(idx) - 1:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).eval()


def load_policy(kind: str, checkpoint: str) -> torch.nn.Sequential:
    return load_mlp(checkpoint, PREFIX[kind])


def policy_input(kind: str, obs) -> torch.Tensor:
    """The tensor a policy of this kind reads from the env's observation dict."""
    if kind == "v1":
        return obs["policy"]
    if kind == "teacher":
        return obs["teacher"]
    if kind == "student":
        return torch.cat([obs["policy"], obs["proprio_history"]], dim=1)
    raise ValueError(f"unknown policy kind {kind!r}")
