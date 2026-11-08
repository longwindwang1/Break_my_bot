"""Evaluate locomotion policies side by side under controlled disturbance scenarios.

All policies run in the same env instance: env i is driven by policy i % P, so every policy sees the
same terrain / command distribution. One scenario per process (Isaac Sim does not reliably recreate a
scene in-process); with several --scenarios the script re-invokes itself per scenario and then prints
a summary table.

Policies (name=kind:checkpoint):
  v1       PPO actor on obs["policy"]                          (W1 baseline, 235-d)
  teacher  PPO actor on obs["teacher"]                         (v2 stage A; privileged, reference only)
  student  StudentTeacher student on policy + proprio_history  (v2 stage B; deployable)

Example (repo root):
  PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python isaac/eval_policy.py --scenarios all \
    --policy v1=v1:logs/rsl_rl/unitree_go2_rough/<run>/model_1499.pt \
    --policy v2_student=student:logs/rsl_rl/bmb_go2_v2/<run>/model_999.pt \
    --out experiments/v2_eval/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = {
    # name: description (applied in configure_scenario)
    "mu_1.0": "constant foot friction 1.0",
    "mu_0.5": "constant foot friction 0.5 (wet)",
    "mu_0.2": "constant foot friction 0.2",
    "mu_0.1": "constant foot friction 0.1 (ice)",
    "mu_0.05": "constant foot friction 0.05 (polished ice)",
    "mu_drop": "friction 1.0 -> 0.1 on all feet at t = 5 s",
    "push": "friction 1.0, velocity pushes of up to 1.0 m/s every 3-5 s",
    "pits_clean": "pit terrain only, height scan truthful",
    "pits_deceived": "pit terrain only, height scan reports flat ground for the whole episode",
    "mu_0.1_cautious": "friction 0.1, command capped to the cautious gait (0.3 m/s, 0.5 rad/s)",
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--scenarios", default="all", help=f"comma-separated subset of {list(SCENARIOS)} or 'all'")
parser.add_argument("--policy", action="append", required=True, help="name=kind:checkpoint (kind: v1|teacher|student)")
parser.add_argument("--envs_per_policy", type=int, default=512)
parser.add_argument("--steps", type=int, default=1000, help="policy steps (1000 = one 20 s episode)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", required=True, help="results .jsonl (appended)")
args_cli, extra = parser.parse_known_args()

scenarios = list(SCENARIOS) if args_cli.scenarios == "all" else args_cli.scenarios.split(",")
for name in scenarios:
    if name not in SCENARIOS:
        sys.exit(f"unknown scenario {name!r}")

if len(scenarios) > 1:
    # Driver mode: one subprocess per scenario, then summarize.
    for name in scenarios:
        cmd = [sys.executable, __file__, "--scenarios", name, "--out", args_cli.out, "--seed", str(args_cli.seed),
               "--envs_per_policy", str(args_cli.envs_per_policy), "--steps", str(args_cli.steps)]
        for p in args_cli.policy:
            cmd += ["--policy", p]
        print(f"[eval] scenario {name}", flush=True)
        if subprocess.run(cmd + extra).returncode != 0:
            sys.exit(f"scenario {name} failed")
    os.execv(sys.executable, [sys.executable, str(REPO_ROOT / "isaac" / "eval_summary.py"), args_cli.out])

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from isaaclab.managers import EventTermCfg  # noqa: E402

import isaac.bmb_tasks  # noqa: E402, F401
from isaac.bmb_tasks import mdp  # noqa: E402
from isaac.bmb_tasks.go2_v2_env_cfg import FEET, Go2V2EnvCfg_PLAY  # noqa: E402
from isaac.interface import load_interface  # noqa: E402
from isaac.policy_loader import load_policy, policy_input  # noqa: E402

SCENARIO = scenarios[0]


def configure_scenario(cfg: Go2V2EnvCfg_PLAY, name: str, num_envs: int) -> None:
    cfg.scene.num_envs = num_envs
    cfg.seed = args_cli.seed
    cfg.commands.base_velocity.debug_vis = False
    gen = cfg.scene.terrain.terrain_generator
    gen.num_rows, gen.num_cols = 10, 10
    gen.difficulty_range = (0.0, 0.5)  # isolate the disturbance from terrain difficulty
    ev = cfg.events

    def set_friction(mu: float):
        for k in ("lower_start", "lower_end", "upper"):
            ev.foot_friction_reset.params[k] = mu
        ev.foot_friction_change = None

    if name.startswith("mu_") and name != "mu_drop":
        set_friction(float(name.split("_")[1]))
    elif name in ("mu_drop", "push"):
        set_friction(1.0)
    if name == "push":  # (the Play config disables pushes)
        ev.push = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 5.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        )
    if name.startswith("pits"):
        set_friction(1.0)
        pits = gen.sub_terrains["pits"]
        pits.proportion = 1.0
        gen.sub_terrains = {"pits": pits}
        gen.difficulty_range = (0.5, 1.0)
    if name == "pits_deceived":
        ev.scan_deception = EventTermCfg(
            func=mdp.schedule_scan_deception,
            mode="reset",
            params={"prob": 1.0, "duration_s": (1.0e3, 1.0e3), "start_s": (0.0, 0.0), "pit_sub_terrain": "pits"},
        )
    if name.endswith("_cautious"):
        cg = load_interface()["policies"]["v2"]["cautious_gait"]
        r = cfg.commands.base_velocity.ranges
        r.lin_vel_x = r.lin_vel_y = (-cg["max_lin_vel"], cg["max_lin_vel"])
        r.ang_vel_z = (-cg["max_ang_vel"], cg["max_ang_vel"])


def main():
    policies = []
    for spec in args_cli.policy:
        name, rest = spec.split("=", 1)
        kind, ckpt = rest.split(":", 1)
        policies.append((name, kind, load_policy(kind, ckpt).to(args_cli.device or "cuda:0"), ckpt))
    P = len(policies)
    num_envs = P * args_cli.envs_per_policy

    cfg = Go2V2EnvCfg_PLAY()
    configure_scenario(cfg, SCENARIO, num_envs)
    env = gym.make("BMB-Go2-Rough-v2-Play", cfg=cfg).unwrapped
    device = env.device
    for i, (n, k, net, c) in enumerate(policies):
        policies[i] = (n, k, net.to(device), c)
    owner = torch.arange(num_envs, device=device) % P

    robot = env.scene["robot"]
    contact = env.scene.sensors["contact_forces"]
    foot_body_ids, _ = robot.find_bodies(FEET)
    foot_sensor_ids, _ = contact.find_bodies(FEET)
    friction_term = env.event_manager.get_term_cfg("foot_friction_reset")

    obs, _ = env.reset(seed=args_cli.seed)
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    track_err = torch.zeros(num_envs, device=device)
    slip = torch.zeros(num_envs, device=device)
    alive_steps = torch.zeros(num_envs, device=device)

    with torch.inference_mode():
        for step in range(args_cli.steps):
            if SCENARIO == "mu_drop" and step == args_cli.steps // 4:
                params = dict(friction_term.params, lower_start=0.1, lower_end=0.1, upper=0.1)
                friction_term.func(env, None, **params)
            actions = torch.zeros(num_envs, 12, device=device)
            for i, (_, kind, net, _) in enumerate(policies):
                mask = owner == i
                actions[mask] = net(policy_input(kind, obs)[mask])
            obs, _, terminated, _, _ = env.step(actions)

            cmd = env.command_manager.get_command("base_velocity")[:, :2]
            err = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2] - cmd, dim=1)
            in_contact = contact.data.net_forces_w[:, foot_sensor_ids].norm(dim=-1) > 1.0
            foot_speed = robot.data.body_lin_vel_w[:, foot_body_ids, :2].norm(dim=-1)
            track_err += err * alive
            slip += (foot_speed * in_contact).sum(dim=1) / in_contact.sum(dim=1).clamp(min=1) * alive
            alive_steps += alive
            alive &= ~terminated

    with open(args_cli.out, "a") as f:
        for i, (name, kind, _, ckpt) in enumerate(policies):
            m = owner == i
            row = {
                "scenario": SCENARIO,
                "policy": name,
                "kind": kind,
                "checkpoint": os.path.relpath(ckpt, REPO_ROOT),
                "envs": int(m.sum()),
                "success_rate": alive[m].float().mean().item(),
                "tracking_error": (track_err[m].sum() / alive_steps[m].sum()).item(),
                "slip_speed": (slip[m].sum() / alive_steps[m].sum()).item(),
                "seed": args_cli.seed,
                "steps": args_cli.steps,
            }
            f.write(json.dumps(row) + "\n")
            print(f"[eval] {SCENARIO:16s} {name:12s} success {row['success_rate']:.3f}  "
                  f"track_err {row['tracking_error']:.3f}  slip {row['slip_speed']:.3f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
