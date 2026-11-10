"""Collect world-model data in Isaac Lab (W4): v2-student and v1 rollouts with logged disturbances.

One split per process. Env i is driven by policy (v2, v2, v1)[i % 3]; in the train split every 4th env
(i % 4 == 0) also gets Gaussian action noise. All disturbances (friction changes, pushes, events, scan
deception) are scheduled by this script and flagged per step, so the dataset can mask them exactly.

Splits (sizes, seeds, conditions): isaac/wm_splits.py.

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
sys.path.insert(0, str(REPO_ROOT))
from isaac.wm_splits import (  # noqa: E402  (no isaaclab imports at module level)
    EVENTS, GEO_SPLITS, ICE_SPLITS, NOISY_SPLITS, POLICY_IDS, POLICY_PATTERN, SPLITS, TILES, configure, install_tile_logger, split_kind,
)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--split", required=True, choices=list(SPLITS))
parser.add_argument("--v2", required=True, help="v2 distillation checkpoint (student)")
parser.add_argument("--v1", required=True, help="v1 rough PPO checkpoint")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--steps", type=int, default=None, help="policy steps to record (50 Hz)")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--event_step", type=int, default=200, help="events / events_pits: step of the event")
parser.add_argument("--action_noise", type=float, default=0.1, help="train splits: noise std on every 4th env")
parser.add_argument("--check_envs", type=int, default=8, help="also record obs['proprio_history'] for the first N envs")
parser.add_argument("--out_dir", default=None, help="default: logs/wm_data/<split>_<time>")
parser.add_argument("--height_map", action="store_true", help="also save the terrain height map (always for geo splits)")
args_cli, _ = parser.parse_known_args()
for k, v in SPLITS[args_cli.split].items():
    if getattr(args_cli, k) is None:
        setattr(args_cli, k, v)
KIND = split_kind(args_cli.split)  # events_test behaves like events, etc.

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(namespace=args_cli)
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaac.bmb_tasks  # noqa: E402, F401
from isaac.bmb_tasks.go2_v2_env_cfg import FEET, Go2V2EnvCfg_PLAY  # noqa: E402
from isaac.bmb_tasks.mdp import state as bmb_state  # noqa: E402
from isaac.ice import IcePhysics, IceSpec, grid_meta, ice_meta, make_ice_map, tile_centres  # noqa: E402
from isaac.interface import load_interface  # noqa: E402
from isaac.policy_loader import load_policy, policy_input  # noqa: E402
from isaac.terrain_map import raycast_height_map  # noqa: E402

install_tile_logger()

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
    configure(cfg, args_cli.split, N, args_cli.seed, args_cli.event_step)
    env = gym.make("BMB-Go2-Rough-v2-Play", cfg=cfg).unwrapped
    device = env.device
    robot = env.scene["robot"]
    foot_ids, _ = robot.find_bodies(FEET, preserve_order=True)
    terrain = env.scene.terrain
    gen = cfg.scene.terrain.terrain_generator
    tile_size = float(gen.size[0])
    origins_xy = terrain.terrain_origins.reshape(-1, 3)[:, :2]

    ice = None
    if args_cli.split in ICE_SPLITS:
        centres = torch.as_tensor(tile_centres(gen), dtype=torch.float32, device=device).reshape(-1, 2)
        if not torch.allclose(centres, origins_xy.float(), atol=1e-3):
            raise RuntimeError("terrain origins are not the tile centres; isaac/ice.py's patch placement assumes they are")
        ice_grid = grid_meta(gen)
        ice_spec = IceSpec(hidden_frac=ICE_SPLITS[args_cli.split])
        ice_map, ice_patches = make_ice_map(gen, ice_grid, args_cli.seed, ice_spec)
        ice_friction = float(load_interface()["terrain_risk"]["ice"]["friction"])
        ice = IcePhysics(env, ice_map, ice_grid, ice_friction, foot_ids)
        print(f"[collect] ice: {len(ice_patches)} patches, cover visible "
              f"{(ice_map == 1).mean():.3f} hidden {(ice_map == 2).mean():.3f}", flush=True)

    nets = {"v2": load_policy("student", args_cli.v2).to(device), "v1": load_policy("v1", args_cli.v1).to(device)}
    kinds = {"v2": "student", "v1": "v1"}
    env_idx = torch.arange(N, device=device)
    owner = [POLICY_PATTERN[i % len(POLICY_PATTERN)] for i in range(N)]
    masks = {p: torch.tensor([o == p for o in owner], device=device) for p in nets}
    policy_id = np.array([POLICY_IDS[o] for o in owner], dtype=np.int8)
    noise_std = torch.zeros(N, device=device)
    if args_cli.split in NOISY_SPLITS:
        noise_std[env_idx % 4 == 0] = args_cli.action_noise
    event_type = (env_idx % len(EVENTS)) if KIND == "events" else torch.zeros(N, dtype=torch.long, device=device)

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
        "feet_pos": ((4, 3), np.float32),  # world positions of the 4 feet (policy foot order FL, FR, RL, RR)
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
    if ice is not None:
        fields["foot_ice"] = ((4,), np.uint8)  # ice under each foot during step t (0 none, 1 visible, 2 hidden)
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
        "events": EVENTS if KIND == "events" else (["none", "deceived"] if KIND == "events_pits" else ["none"]),
        "policy_pattern": POLICY_PATTERN,
        "policy_ids": POLICY_IDS,
        "checkpoints": {"v2": args_cli.v2, "v1": args_cli.v1},
        "action_noise": args_cli.action_noise if args_cli.split in NOISY_SPLITS else 0.0,
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
    ice_s = 0.0
    ever_done = torch.zeros(N, dtype=torch.bool, device=device)
    last_action_mismatch = 0
    t_start = time.time()
    with torch.inference_mode():
        for t in range(T):
            push_flag = torch.zeros(N, dtype=torch.bool, device=device)
            mu_flag = torch.zeros(N, dtype=torch.bool, device=device)
            event = torch.zeros(N, dtype=torch.int8, device=device)

            # disturbances, applied before o_t is used (they act during step t)
            if KIND == "train":
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
            elif KIND == "events" and t == args_cli.event_step:
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

            if ice is not None:  # friction under each foot for step t
                t_ice = time.perf_counter()
                ice.update()
                ice_s += time.perf_counter() - t_ice
                mm["foot_ice"][t] = ice.code.cpu().numpy()

            # record o_t and the state it was computed from
            pol = obs["policy"]
            root_xy = robot.data.root_pos_w[:, :2]
            dist, idx = (root_xy[:, None, :] - origins_xy[None]).abs().amax(-1).min(1)
            tile = torch.where(dist <= tile_size / 2, idx, torch.full_like(idx, -1))
            deceived = bmb_state.deception_active(env)
            if KIND == "events_pits":
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
            mm["feet_pos"][t] = robot.data.body_pos_w[:, foot_ids].cpu().numpy()
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
                if ice is not None:
                    meta["ice_update_s"] = round(ice_s, 1)
                    meta["ice_writes"] = ice.writes
                write_meta()
                print(f"[collect] {args_cli.split} step {t + 1}/{T}  falls so far "
                      f"{int(np.asarray(mm['terminated'][: t + 1]).sum())}  {meta['wall_s']} s"
                      + (f"  (ice update {ice_s:.1f} s, {ice.writes} env writes)" if ice is not None else ""), flush=True)

    for m in list(mm.values()) + ([check] if check is not None else []):
        m.flush()
    info = dict(np.load(out_dir / "env_info.npz"))
    info["reset_before_event"] = ever_done.cpu().numpy()
    np.savez(out_dir / "env_info.npz", **info)
    if args_cli.split in GEO_SPLITS or args_cli.height_map:
        hmap, hmeta = raycast_height_map(gen, device=device)
        np.save(out_dir / "terrain_height.npy", hmap)
        meta["height_map"] = hmeta
        if ice is not None:
            if any(hmeta[k] != ice_grid[k] for k in ("res", "x_min", "y_min", "nx", "ny")):
                raise RuntimeError(f"ice grid {ice_grid} != height map grid {hmeta}")
            np.save(out_dir / "ice_mask.npy", ice_map)
            (out_dir / "ice_patches.json").write_text(json.dumps(ice_patches))
            meta["ice_map"] = ice_meta(ice_grid, args_cli.seed, ice_spec, ice_friction, ice_patches, ice_map)
        print(f"[collect] height map {hmap.shape} res {hmeta['res']} m, misses {hmeta['misses']}", flush=True)
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
