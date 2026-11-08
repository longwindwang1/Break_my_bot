"""Closed-loop test of the terrain-risk slowdown in Isaac Lab (W6): does slowing down where the model sees risk
ahead reduce falls on unseen terrain, and what does it cost in speed?

Terrain: the geo_test terrain (seed 700, never trained on); rollouts are varied with --seed. Env i runs policy
(v2, v2, v1)[i % 3]. Commands are restricted to mostly-forward ones (vx 0.3-1.0, |vy| <= 0.2, |wz| <= 0.5),
the range the egocentric window covers. Every 10 steps (5 Hz... 0.2 s) the risk ahead is computed from the
egocentric window with the original command; a trigger holds the cautious gait for 50 steps (1 s).

Modes:
  none    never slow down
  risk    slow down when risk ahead (h25) > the policy's threshold (--trigger 10pct|20pct from the eval results)
  random  slow down at random with the same fraction of cautious time as given by --cautious_frac
  always  always cautious
The cautious gait (robot_interface.yaml policies.v2.cautious_gait) is applied by writing the clamped command
into obs["policy"][:, 9:12], which both policies read; the command term itself is left alone.

  PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python isaac/eval_risk_slowdown.py --mode risk --trigger 10pct \
    --risk_run logs/terrain_risk/<run> --risk_eval experiments/w6_terrain_risk/results.json \
    --v2 <student ckpt> --v1 <v1 ckpt> --out experiments/w6_terrain_risk/slowdown.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mode", required=True, choices=["none", "risk", "random", "always"])
parser.add_argument("--trigger", default="10pct", choices=["10pct", "20pct"])
parser.add_argument("--cautious_frac", type=float, default=None, help="random mode: target fraction of cautious time")
parser.add_argument("--risk_run", required=True)
parser.add_argument("--risk_eval", required=True, help="results.json of models.terrain_risk.evaluate (trigger thresholds)")
parser.add_argument("--v2", required=True)
parser.add_argument("--v1", required=True)
parser.add_argument("--num_envs", type=int, default=1023)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0, help="rollout seed (the terrain is always geo_test's)")
parser.add_argument("--out", required=True)
args_cli, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(namespace=args_cli)
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaac.bmb_tasks  # noqa: E402, F401
from isaac.bmb_tasks.go2_v2_env_cfg import Go2V2EnvCfg_PLAY  # noqa: E402
from isaac.interface import load_interface  # noqa: E402
from isaac.policy_loader import load_policy, policy_input  # noqa: E402
from isaac.terrain_map import raycast_height_map  # noqa: E402
from isaac.wm_splits import POLICY_IDS, POLICY_PATTERN, SPLITS, configure  # noqa: E402
from models.terrain_risk.evaluate import risk_ahead  # noqa: E402
from models.terrain_risk.model import load_risk_model  # noqa: E402
from models.terrain_risk.patches import HeightMap, load_risk_layout  # noqa: E402

_SLOW = load_interface()["terrain_risk"]["slowdown"]
_DT = load_interface()["timing"]["policy_dt"]
UPDATE_EVERY, HOLD = round(_SLOW["update_period_s"] / _DT), round(_SLOW["hold_s"] / _DT)


def main():
    N, T = args_cli.num_envs, args_cli.steps
    cfg = Go2V2EnvCfg_PLAY()
    configure(cfg, "geo_test", N, SPLITS["geo_test"]["seed"])  # terrain of geo_test
    r = cfg.commands.base_velocity.ranges
    r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (0.3, 1.0), (-0.2, 0.2), (-0.5, 0.5)
    env = gym.make("BMB-Go2-Rough-v2-Play", cfg=cfg).unwrapped
    dev = env.device
    robot = env.scene["robot"]
    hmap, hmeta = raycast_height_map(cfg.scene.terrain.terrain_generator, device=dev)
    hm = HeightMap(hmap, hmeta, device=dev)
    lay = load_risk_layout()
    model, _ = load_risk_model(Path(args_cli.risk_run) / "model.pt", dev)
    thr_tab = json.loads(Path(args_cli.risk_eval).read_text())["trigger_thresholds"]
    cg = load_interface()["policies"]["v2"]["cautious_gait"]

    nets = {"v2": load_policy("student", args_cli.v2).to(dev), "v1": load_policy("v1", args_cli.v1).to(dev)}
    kinds = {"v2": "student", "v1": "v1"}
    owner = [POLICY_PATTERN[i % len(POLICY_PATTERN)] for i in range(N)]
    masks = {p: torch.tensor([o == p for o in owner], device=dev) for p in nets}
    pid = torch.tensor([POLICY_IDS[o] for o in owner], device=dev)
    thr = torch.where(pid == POLICY_IDS["v1"], torch.tensor(thr_tab["v1"][f"trigger_{args_cli.trigger}"], device=dev),
                      torch.tensor(thr_tab["v2"][f"trigger_{args_cli.trigger}"], device=dev))
    p_rand = 0.0
    if args_cli.mode == "random":
        f = args_cli.cautious_frac
        p_rand = 1.0 - (1.0 - f) ** (UPDATE_EVERY / HOLD)

    obs, _ = env.reset(seed=args_cli.seed)
    hold = torch.zeros(N, dtype=torch.long, device=dev)
    alive_first = torch.ones(N, dtype=torch.bool, device=dev)  # first episode not ended by a fall yet
    first_done = torch.zeros(N, dtype=torch.bool, device=dev)
    falls = torch.zeros(N, device=dev)
    dist = torch.zeros(N, device=dev)
    cautious_steps = torch.zeros(N, device=dev)
    last_xy = robot.data.root_pos_w[:, :2].clone()
    g = torch.Generator(device=dev).manual_seed(args_cli.seed)
    t0 = time.time()
    with torch.inference_mode():
        for t in range(T):
            cmd = env.command_manager.get_command("base_velocity").clone()
            if t % UPDATE_EVERY == 0:
                if args_cli.mode == "risk":
                    x, y = robot.data.root_pos_w[:, 0], robot.data.root_pos_w[:, 1]
                    q = robot.data.root_quat_w
                    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
                    ra = risk_ahead(model, hm, lay, x, y, yaw, cmd, pid)
                    hold = torch.where(ra > thr, torch.full_like(hold, HOLD), hold)
                elif args_cli.mode == "random":
                    trig = torch.rand(N, device=dev, generator=g) < p_rand
                    hold = torch.where(trig, torch.full_like(hold, HOLD), hold)
            cautious = (hold > 0) | (args_cli.mode == "always")
            if cautious.any():
                v = cmd[:, :2]
                scale = torch.clamp(cg["max_lin_vel"] / v.norm(dim=1).clamp(min=1e-6), max=1.0)
                c = torch.cat([v * scale[:, None], cmd[:, 2:3].clamp(-cg["max_ang_vel"], cg["max_ang_vel"])], dim=1)
                obs["policy"][:, 9:12] = torch.where(cautious[:, None], c, cmd)
            actions = torch.zeros(N, 12, device=dev)
            for p, net in nets.items():
                actions[masks[p]] = net(policy_input(kinds[p], obs)[masks[p]])
            obs, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            xy = robot.data.root_pos_w[:, :2]
            step_d = (xy - last_xy).norm(dim=1)
            dist += torch.where(done | (step_d > 1.0), torch.zeros_like(step_d), step_d)  # skip resets
            last_xy = xy.clone()
            falls += terminated.float()
            cautious_steps += cautious.float()
            alive_first &= ~(terminated & ~first_done)
            first_done |= done
            hold = torch.clamp(hold - 1, min=0)
            hold[done] = 0
    rows = []
    for p in ("v1", "v2"):
        m = masks[p]
        rows.append({
            "mode": args_cli.mode, "trigger": args_cli.trigger if args_cli.mode == "risk" else None,
            "cautious_frac_target": args_cli.cautious_frac, "policy": p, "envs": int(m.sum()), "seed": args_cli.seed,
            "steps": T, "success_first_episode": float(alive_first[m].float().mean()),
            "falls_per_100m": float(falls[m].sum() / dist[m].sum().clamp(min=1e-6) * 100),
            "mean_speed": float(dist[m].sum() / (int(m.sum()) * T * env.step_dt)),
            "cautious_frac": float(cautious_steps[m].sum() / (int(m.sum()) * T)),
            "falls": int(falls[m].sum()), "distance_m": float(dist[m].sum()),
        })
        print(f"[slowdown] {args_cli.mode:6s} {p}: success {rows[-1]['success_first_episode']:.3f} falls/100m "
              f"{rows[-1]['falls_per_100m']:.3f} speed {rows[-1]['mean_speed']:.3f} cautious {rows[-1]['cautious_frac']:.3f}", flush=True)
    with open(args_cli.out, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[slowdown] {time.time() - t0:.0f} s", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
