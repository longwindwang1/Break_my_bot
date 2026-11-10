"""Data splits for the world model and the terrain-risk model: sizes, seeds and env configuration.

Importable before Isaac Sim starts (argparse needs SPLITS); everything that imports isaaclab lives inside
functions. Used by isaac/collect_wm_data.py, isaac/dump_terrain.py and isaac/eval_risk_slowdown.py.

  train          full v2 randomization: foot mu 0.05-1.25 changing every 4-8 s (half single-foot), pushes of
                 0-1.0 m/s in a random direction every 5-10 s, full terrain mix, difficulty 0-1
  train_nominal  "normal physics" only: as clean, plus action noise on every 4th env (surprisal model)
  clean          mu fixed per episode in [0.6, 1.25], no pushes, no noise; same terrain   (threshold calibration)
  events         clean conditions, difficulty 0-0.5; at --event_step every env gets one event (EVENTS[i % 7])
  events_pits    pit terrain only, pits 15-30 cm; from --event_step the height scan reports flat ground in
                 about half of the envs (deceived flag)
  geo_train      terrain-risk data: clean conditions, terrain mix + pits_deep (15-30 cm, truthful scan)
  geo_test       same, another terrain seed (held-out terrain)
  geo_train2/3   more terrain-risk training data (other terrain seeds)
  ice_train(2/3) as geo_train plus ice patches (isaac/ice.py), all visible; other terrain seeds
  ice_test       as ice_train, held-out terrain seed, half of the ice patches hidden
  events_test, events_pits_test
                 as events / events_pits with other seeds: report-only copies (W7 JEPA comparison), so
                 detector settings chosen on events / events_pits are scored on unseen rollouts
"""

from __future__ import annotations

import copy

SPLITS = {
    "train": dict(num_envs=4096, steps=1500, seed=100),
    "train_nominal": dict(num_envs=4096, steps=1500, seed=500),
    "clean": dict(num_envs=2048, steps=1500, seed=200),
    "events": dict(num_envs=4096, steps=600, seed=300),
    "events_pits": dict(num_envs=2048, steps=600, seed=400),
    "geo_train": dict(num_envs=4096, steps=1500, seed=600),
    "geo_test": dict(num_envs=2048, steps=1500, seed=700),
    "geo_train2": dict(num_envs=4096, steps=1500, seed=610),
    "geo_train3": dict(num_envs=4096, steps=1500, seed=620),
    "ice_train": dict(num_envs=4096, steps=1500, seed=800),
    "ice_train2": dict(num_envs=4096, steps=1500, seed=810),
    "ice_train3": dict(num_envs=4096, steps=1500, seed=820),
    "ice_test": dict(num_envs=2048, steps=1500, seed=900),
    "events_test": dict(num_envs=4096, steps=600, seed=310),
    "events_pits_test": dict(num_envs=2048, steps=600, seed=410),
}
SPLIT_KIND = {"events_test": "events", "events_pits_test": "events_pits"}  # copies with another seed
NOISY_SPLITS = ("train", "train_nominal")  # every 4th env gets action noise
ICE_SPLITS = {"ice_train": 0.0, "ice_train2": 0.0, "ice_train3": 0.0, "ice_test": 0.5}  # -> hidden patch fraction
GEO_SPLITS = ("geo_train", "geo_test", "geo_train2", "geo_train3", *ICE_SPLITS)  # pits_deep + height map
EVENTS = ["none", "mu_0.3", "mu_0.1", "mu_0.05", "foot_0.05", "push_0.5", "push_1.0"]  # id = index
POLICY_PATTERN = ["v2", "v2", "v1"]  # env i -> POLICY_PATTERN[i % 3]
POLICY_IDS = {"v1": 0, "v2": 1}  # world-model policy one-hot order (robot_interface.yaml world_model.policies)



def split_kind(split: str) -> str:
    """The split whose conditions `split` uses (itself, except for the seed-only copies in SPLIT_KIND)."""
    return SPLIT_KIND.get(split, split)


# (row, col) -> (sub-terrain name, difficulty), filled while the terrain generator runs
TILES: dict[tuple[int, int], tuple[str, float]] = {}


def install_tile_logger() -> None:
    """Record every generated tile's sub-terrain type and difficulty in TILES (call before gym.make)."""
    from isaaclab.terrains import TerrainGenerator

    if getattr(TerrainGenerator, "_bmb_logged", False):
        return
    get_mesh, add_sub = TerrainGenerator._get_terrain_mesh, TerrainGenerator._add_sub_terrain

    def _get_mesh_logged(self, difficulty, cfg):
        self._bmb_last = (next(k for k, v in self.cfg.sub_terrains.items() if v is cfg), float(difficulty))
        return get_mesh(self, difficulty, cfg)

    def _add_sub_logged(self, mesh, origin, row, col, sub_terrain_cfg):
        TILES[(int(row), int(col))] = self._bmb_last
        return add_sub(self, mesh, origin, row, col, sub_terrain_cfg)

    TerrainGenerator._get_terrain_mesh = _get_mesh_logged
    TerrainGenerator._add_sub_terrain = _add_sub_logged
    TerrainGenerator._bmb_logged = True


def configure(cfg, split: str, num_envs: int, seed: int, event_step: int = 200) -> None:
    """Edit a Go2V2EnvCfg_PLAY for the given split."""
    from isaaclab.managers import EventTermCfg

    split = split_kind(split)
    from isaac.bmb_tasks import mdp

    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.commands.base_velocity.debug_vis = False
    cfg.curriculum.terrain_levels = None  # envs keep their tile; the generator has no curriculum anyway
    gen = cfg.scene.terrain.terrain_generator
    gen.num_rows, gen.num_cols = 10, 20
    gen.curriculum = False
    gen.difficulty_range = (0.0, 1.0)
    ev = cfg.events
    ev.foot_friction_change = None  # mid-episode changes are scheduled by the collector
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
        t0 = (event_step + 0.5) * cfg.sim.dt * cfg.decimation  # floor(t0 / step_dt) == event_step
        ev.scan_deception = EventTermCfg(
            func=mdp.schedule_scan_deception,
            mode="reset",
            params={"prob": 0.5, "duration_s": (1.0e3, 1.0e3), "start_s": (t0, t0), "pit_sub_terrain": "pits"},
        )
    if split in GEO_SPLITS:
        deep = copy.deepcopy(gen.sub_terrains["pits"])
        deep.proportion = 0.15
        deep.obstacle_height_range = (0.15, 0.30)  # deeper than the policy's training pits; the scan stays truthful
        gen.sub_terrains = {**gen.sub_terrains, "pits_deep": deep}
