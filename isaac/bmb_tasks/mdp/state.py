"""Per-env randomization state shared between event, observation and curriculum terms.

Events write the values they sampled here; privileged observations read them, so nothing has to be
read back from the (CPU-side) PhysX views every step. Stored as attributes on the env. Created lazily
because the observation manager evaluates every term once, before any event has run, to infer shapes.
"""

from __future__ import annotations

import torch

NUM_FEET = 4
NOMINAL_FOOT_FRICTION = 0.8  # stock Go2 startup material


def foot_friction(env) -> torch.Tensor:
    """[num_envs, 4] static friction currently set on each foot (policy foot order)."""
    if not hasattr(env, "bmb_foot_friction"):
        env.bmb_foot_friction = torch.full((env.num_envs, NUM_FEET), NOMINAL_FOOT_FRICTION, device=env.device)
    return env.bmb_foot_friction


def added_base_mass(env) -> torch.Tensor:
    """[num_envs, 1] mass added to the base at startup (kg)."""
    if not hasattr(env, "bmb_added_mass"):
        env.bmb_added_mass = torch.zeros(env.num_envs, 1, device=env.device)
    return env.bmb_added_mass


def base_com_offset(env) -> torch.Tensor:
    """[num_envs, 3] offset added to the base centre of mass at startup (m, base frame)."""
    if not hasattr(env, "bmb_com_offset"):
        env.bmb_com_offset = torch.zeros(env.num_envs, 3, device=env.device)
    return env.bmb_com_offset


def deception_window(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env [start, end) episode step window during which the height scan is deceptive."""
    if not hasattr(env, "bmb_deception_start"):
        env.bmb_deception_start = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.bmb_deception_end = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env.bmb_deception_start, env.bmb_deception_end


def deception_active(env) -> torch.Tensor:
    """[num_envs] bool: is the height scan currently replaced by a fake flat reading."""
    start, end = deception_window(env)
    step = env.episode_length_buf if hasattr(env, "episode_length_buf") else torch.zeros_like(start)
    return (step >= start) & (step < end)
