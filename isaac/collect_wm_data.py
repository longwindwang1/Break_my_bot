"""Collect world-model data in Isaac Lab (W4): v2-student and v1 rollouts with logged disturbances.

One split per process. Env i is driven by policy (v2, v2, v1)[i % 3]; in the train split every 4th env
(i % 4 == 0) also gets Gaussian action noise. All disturbances (friction changes, pushes, events, scan
deception) are scheduled by this script and flagged per step, so the dataset can mask them exactly.

Splits:
  train        full v2 randomization: foot mu 0.05-1.25 changing every 4-8 s (half single-foot), pushes of
               0-1.0 m/s in a random direction every 5-10 s, full terrain mix, difficulty 0-1
  clean        mu fixed per episode in [0.6, 1.25], no pushes, no noise; same terrain   (threshold calibration)
  events       clean conditions, difficulty 0-0.5; at --event_step every env gets one event (EVENTS[i % 7])
  events_pits  pit terrain only, pits 15-30 cm; from --event_step the height scan reports flat ground in
               about half of the envs (deceived flag)

Per policy step t the files hold o_t (and the pose / friction / tile it was computed from), the action a_t,
and the done flags of the step a_t was applied in. Layout [T, N, ...] .npy memmaps + meta.json.

Example (repo root):
  PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python isaac/collect_wm_data.py --split train \
    --v2 logs/rsl_rl/bmb_go2_v2/2026-11-06_16-34-51_v2_student/model_999.pt \
    --v1 logs/rsl_rl/unitree_go2_rough/2026-10-12_21-27-17_w1_go2_rough_baseline/model_1499.pt
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SPLITS = {
    "train": dict(num_envs=4096, steps=1500, seed=100),
    "clean": dict(num_envs=2048, steps=1500, seed=200),
    "events": dict(num_envs=4096, steps=600, seed=300),
    "events_pits": dict(num_envs=2048, steps=600, seed=400),
}
EVENTS = ["none", "mu_0.3", "mu_0.1", "mu_0.05", "foot_0.05", "push_0.5", "push_1.0"]  # id = index
POLICY_PATTERN = ["v2", "v2", "v1"]  # env i -> POLICY_PATTERN[i % 3]
POLICY_IDS = {"v1": 0, "v2": 1}  # world-model policy one-hot order (robot_interface.yaml world_model.policies)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--split", required=True, choices=list(SPLITS))
parser.add_argument("--v2", required=True, help="v2 distillation checkpoint (student)")
parser.add_argument("--v1", required=True, help="v1 rough PPO checkpoint")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--steps", type=int, default=None, help="policy steps to record (50 Hz)")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--event_step", type=int, default=200, help="events / events_pits: step of the event")
parser.add_argument("--action_noise", type=float, default=0.1, help="train split: noise std on every 4th env")
parser.add_argument("--check_envs", type=int, default=8, help="also record obs['proprio_history'] for the first N envs")
parser.add_argument("--out_dir", default=None, help="default: logs/wm_data/<split>_<time>")
args_cli, _ = parser.parse_known_args()
for k, v in SPLITS[args_cli.split].items():
    if getattr(args_cli, k) is None:
        setattr(args_cli, k, v)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(namespace=args_cli)
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from isaaclab.managers import EventTermCfg  # noqa: E402
from isaaclab.terrains import TerrainGenerator  # noqa: E402

import isaac.bmb_tasks  # noqa: E402, F401
from isaac.bmb_tasks import mdp  # noqa: E402
from isaac.bmb_tasks.go2_v2_env_cfg import Go2V2EnvCfg_PLAY  # noqa: E402
from isaac.bmb_tasks.mdp import state as bmb_state  # noqa: E402
from isaac.policy_loader import load_policy, policy_input  # noqa: E402

# --- terrain bookkeeping: (row, col) -> (sub-terrain name, difficulty) ------------------------------
TILES: dict[tuple[int, int], tuple[str, float]] = {}
_get_mesh, _add_sub = TerrainGenerator._get_terrain_mesh, TerrainGenerator._add_sub_terrain


def _get_mesh_logged(self, difficulty, cfg):
    self._bmb_last = (next(k for k, v in self.cfg.sub_terrains.items() if v is cfg), float(difficulty))
    return _get_mesh(self, difficulty, cfg)


def _add_sub_logged(self, mesh, origin, row, col, sub_terrain_cfg):
    TILES[(int(row), int(col))] = self._bmb_last
    return _add_sub(self, mesh, origin, row, col, sub_terrain_cfg)


TerrainGenerator._get_terrain_mesh = _get_mesh_logged
TerrainGenerator._add_sub_terrain = _add_sub_logged


def configure(cfg: Go2V2EnvCfg_PLAY) -> None:
    split = args_cli.split
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    cfg.commands.base_velocity.debug_vis = False
    cfg.curriculum.terrain_levels = None  # envs keep their tile; the generator has no curriculum anyway
    gen = cfg.scene.terrain.terrain_generator
    gen.num_rows, gen.num_cols = 10, 20
    gen.curriculum = False
    gen.difficulty_range = (0.0, 1.0)
    ev = cfg.events
    ev.foot_friction_change = None  # mid-episode changes are scheduled by this script
    ev.push = None
    ev.scan_deception = None
    fr = ev.foot_friction_reset.params
    if split == "train":
        fr["lower_start"], fr["lower_end"], fr["upper"] = 0.05, 0.05, 1.25
    else:
        fr["lower_start"], fr["lower_end"], fr["upper"] = 0.6, 0.6, 1.25
    if split == "events":
        gen.difficulty_range = (0.0, 0.5)
    if split == "events_pits":
        pits = gen.sub_terrains["pits"]
        pits.proportion = 1.0
        pits.obstacle_height_range = (0.15, 0.30)  # eval only: deeper than training (4-16 cm)
        gen.sub_terrains = {"pits": pits}
        gen.difficulty_range = (0.5, 1.0)
        t0 = (args_cli.event_step + 0.5) * cfg.sim.dt * cfg.decimation  # floor(t0 / step_dt) == event_step
        ev.scan_deception = EventTermCfg(
            func=mdp.schedule_scan_deception,
            mode="reset",
            params={"prob": 0.5, "duration_s": (1.0e3, 1.0e3), "start_s": (t0, t0), "pit_sub_terrain": "pits"},
        )


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
        return out.stdout.strip() + ("-dirty" if dirty.strip() else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    T, N = args_cli.steps, args_cli.num_envs
    out_dir = Path(args_cli.out_dir or REPO_ROOT / "logs" / "wm_data" / f"{args_cli.split}_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Go2V2EnvCfg_PLAY()
    configure(cfg)
    env = gym.make("BMB-Go2-Rough-v2-Play", cfg=cfg).unwrapped
    device = env.device
    robot = env.scene["robot"]
    terrain = env.scene.terrain
    gen = cfg.scene.terrain.terrain_generator
    tile_size = float(gen.size[0])
    origins_xy = terrain.terrain_origins.reshape(-1, 3)[:, :2]

    nets = {"v2": load_policy("student", args_cli.v2).to(device), "v1": load_policy("v1", args_cli.v1).to(device)}
    kinds = {"v2": "student", "v1": "v1"}
    env_idx = torch.arange(N, device=device)
    owner = [POLICY_PATTERN[i % len(POLICY_PATTERN)] for i in range(N)]
    masks = {p: torch.tensor([o == p for o in owner], device=device) for p in nets}
    policy_id = np.array([POLICY_IDS[o] for o in owner], dtype=np.int8)
    noise_std = torch.zeros(N, device=device)
    if args_cli.split == "train":
        noise_std[env_idx % 4 == 0] = args_cli.action_noise
    event_type = (env_idx % len(EVENTS)) if args_cli.split == "events" else torch.zeros(N, dtype=torch.long, device=device)

    fric = env.event_manager.get_term_cfg("foot_friction_reset")
    step_dt = env.step_dt

    def set_friction(ids: torch.Tensor, lo: float, hi: float, single_foot_prob: float = 0.0):
        if len(ids):
            fric.func(env, ids, **dict(fric.params, lower_start=lo, lower_end=lo, upper=hi, single_foot_prob=single_foot_prob))

    def push(ids: torch.Tensor, speed: torch.Tensor):
        """Add a horizontal velocity of the given magnitude in a uniformly random direction."""
        if len(ids):
            vel = robot.data.root_vel_w[ids].clone()
            ang = torch.rand(len(ids), device=device) * 2 * math.pi
            vel[:, 0] += speed * torch.cos(ang)
            vel[:, 1] += speed * torch.sin(ang)
            robot.write_root_velocity_to_sim(vel, env_ids=ids)

    # --- output files -------------------------------------------------------------------------------
    fields = {
        "proprio": ((48,), np.float32),  # obs["policy"][:, :48] (incl. command and last action)
        "scan": ((187,), np.float16),  # obs["policy"][:, 48:]  (what the robot sees; may be deceived)
        "action": ((12,), np.float32),  # a_t actually applied (incl. noise)
        "root_pos": ((3,), np.float32),
        "root_quat": ((4,), np.float32),  # (w, x, y, z)
        "foot_mu": ((4,), np.float16),
        "episode_step": ((), np.int16),  # env.episode_length_buf when o_t was computed (0 = first frame)
        "tile": ((), np.int16),  # row * num_cols + col of the tile under the base, -1 on the border
        "deceived": ((), np.bool_),  # height scan in o_t is fake
        "push": ((), np.bool_),  # a push was applied right before step t (o_t -> o_t+1 contains it)
        "mu_change": ((), np.bool_),  # foot friction changed right before step t
        "event": ((), np.int8),  # events split: EVENTS id applied right before step t (0 = none)
        "terminated": ((), np.bool_),  # step t ended the episode with a fall (base contact)
        "truncated": ((), np.bool_),  # step t ended the episode by time-out
    }
    mm = {
        k: np.lib.format.open_memmap(out_dir / f"{k}.npy", mode="w+", dtype=dt, shape=(T, N, *shape))
        for k, (shape, dt) in fields.items()
    }
    C = min(args_cli.check_envs, N)
    check = np.lib.format.open_memmap(out_dir / "check_proprio_history.npy", mode="w+", dtype=np.float32, shape=(T, C, 1125)) if C else None

    meta = {
        "split": args_cli.split,
        "num_envs": N,
        "steps": T,
        "steps_written": 0,
        "seed": args_cli.seed,
        "step_dt": step_dt,
        "event_step": args_cli.event_step,
        "events": EVENTS if args_cli.split == "events" else (["none", "deceived"] if args_cli.split == "events_pits" else ["none"]),
        "policy_pattern": POLICY_PATTERN,
        "policy_ids": POLICY_IDS,
        "checkpoints": {"v2": args_cli.v2, "v1": args_cli.v1},
        "action_noise": args_cli.action_noise if args_cli.split == "train" else 0.0,
        "git": git_hash(),
        "terrain": {
            "num_rows": gen.num_rows,
            "num_cols": gen.num_cols,
            "tile_size": tile_size,
            "difficulty_range": list(gen.difficulty_range),
            "tiles": [[r, c, name, d] for (r, c), (name, d) in sorted(TILES.items())],
        },
        "fields": {k: {"shape": [T, N, *s], "dtype": np.dtype(d).name} for k, (s, d) in fields.items()},
        "check_envs": C,
    }
    np.savez(
        out_dir / "env_info.npz",
        policy_id=policy_id,
        noise_std=noise_std.cpu().numpy(),
        event_type=event_type.cpu().numpy().astype(np.int8),
    )

    def write_meta():
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    write_meta()

    # --- schedules (train split) -------------------------------------------------------------------
    def countdown(lo_s: float, hi_s: float, n: int) -> torch.Tensor:
        return torch.randint(int(lo_s / step_dt), int(hi_s / step_dt) + 1, (n,), device=device)

    next_mu = countdown(4.0, 8.0, N)
    next_push = countdown(5.0, 10.0, N)

    obs, _ = env.reset(seed=args_cli.seed)
    ever_done = torch.zeros(N, dtype=torch.bool, device=device)
    last_action_mismatch = 0
    t_start = time.time()
    with torch.inference_mode():
        for t in range(T):
            push_flag = torch.zeros(N, dtype=torch.bool, device=device)
            mu_flag = torch.zeros(N, dtype=torch.bool, device=device)
            event = torch.zeros(N, dtype=torch.int8, device=device)

            # disturbances, applied before o_t is used (they act during step t)
            if args_cli.split == "train":
                next_mu -= 1
                next_push -= 1
                ids = (next_mu <= 0).nonzero().squeeze(1)
                set_friction(ids, 0.05, 1.25, single_foot_prob=0.5)
                mu_flag[ids] = True
                next_mu[ids] = countdown(4.0, 8.0, len(ids))
                ids = (next_push <= 0).nonzero().squeeze(1)
                push(ids, torch.rand(len(ids), device=device) * 1.0)
                push_flag[ids] = True
                next_push[ids] = countdown(5.0, 10.0, len(ids))
            elif args_cli.split == "events" and t == args_cli.event_step:
                for eid, name in enumerate(EVENTS):
                    ids = (event_type == eid).nonzero().squeeze(1)
                    if name.startswith("mu_"):
                        mu = float(name.split("_")[1])
                        set_friction(ids, mu, mu)
                        mu_flag[ids] = True
                    elif name.startswith("foot_"):
                        mu = float(name.split("_")[1])
                        set_friction(ids, mu, mu, single_foot_prob=1.0)
                        mu_flag[ids] = True
                    elif name.startswith("push_"):
                        push(ids, torch.full((len(ids),), float(name.split("_")[1]), device=device))
                        push_flag[ids] = True
                    event[ids] = eid

            # record o_t and the state it was computed from
            pol = obs["policy"]
            root_xy = robot.data.root_pos_w[:, :2]
            dist, idx = (root_xy[:, None, :] - origins_xy[None]).abs().amax(-1).min(1)
            tile = torch.where(dist <= tile_size / 2, idx, torch.full_like(idx, -1))
            deceived = bmb_state.deception_active(env)
            if args_cli.split == "events_pits":
                was = torch.from_numpy(mm["deceived"][t - 1]).to(device) if t > 0 else torch.zeros_like(deceived)
                event[deceived & ~was] = 1

            actions = torch.zeros(N, 12, device=device)
            for p, net in nets.items():
                actions[masks[p]] = net(policy_input(kinds[p], obs)[masks[p]])
            actions += noise_std[:, None] * torch.randn_like(actions)

            mm["proprio"][t] = pol[:, :48].cpu().numpy()
            mm["scan"][t] = pol[:, 48:].cpu().numpy().astype(np.float16)
            mm["action"][t] = actions.cpu().numpy()
            mm["root_pos"][t] = robot.data.root_pos_w.cpu().numpy()
            mm["root_quat"][t] = robot.data.root_quat_w.cpu().numpy()
            mm["foot_mu"][t] = bmb_state.foot_friction(env).cpu().numpy().astype(np.float16)
            mm["episode_step"][t] = env.episode_length_buf.cpu().numpy().astype(np.int16)
            mm["tile"][t] = tile.cpu().numpy().astype(np.int16)
            mm["deceived"][t] = deceived.cpu().numpy()
            mm["push"][t] = push_flag.cpu().numpy()
            mm["mu_change"][t] = mu_flag.cpu().numpy()
            mm["event"][t] = event.cpu().numpy()
            if check is not None:
                check[t] = obs["proprio_history"][:C].cpu().numpy()

            obs, _, terminated, truncated, _ = env.step(actions)
            mm["terminated"][t] = terminated.cpu().numpy()
            mm["truncated"][t] = truncated.cpu().numpy()

            # o_{t+1}'s last-action term must be a_t wherever the episode continues
            alive = ~(terminated | truncated)
            last_action_mismatch += int((obs["policy"][alive, 36:48] != actions[alive]).any(dim=1).sum())
            if t < args_cli.event_step:
                ever_done |= ~alive

            if (t + 1) % 100 == 0 or t == T - 1:
                meta["steps_written"] = t + 1
                meta["wall_s"] = round(time.time() - t_start, 1)
                write_meta()
                print(f"[collect] {args_cli.split} step {t + 1}/{T}  falls so far "
                      f"{int(np.asarray(mm['terminated'][: t + 1]).sum())}  {meta['wall_s']} s", flush=True)

    for m in list(mm.values()) + ([check] if check is not None else []):
        m.flush()
    info = dict(np.load(out_dir / "env_info.npz"))
    info["reset_before_event"] = ever_done.cpu().numpy()
    np.savez(out_dir / "env_info.npz", **info)
    meta["last_action_mismatch"] = last_action_mismatch
    meta["falls"] = int(np.asarray(mm["terminated"]).sum())
    meta["timeouts"] = int(np.asarray(mm["truncated"]).sum())
    write_meta()
    print(f"[collect] wrote {out_dir}  falls {meta['falls']}  last_action_mismatch {last_action_mismatch}", flush=True)
    env.close()
    if last_action_mismatch:
        sys.exit(f"last-action term != applied action in {last_action_mismatch} env-steps")


if __name__ == "__main__":
    main()
    simulation_app.close()
