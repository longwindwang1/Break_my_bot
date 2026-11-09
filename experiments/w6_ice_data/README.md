# W6 phase 2 data: terrain with visible and hidden ice

Rollouts for the ice-aware terrain-risk model, recorded by `isaac/collect_wm_data.py`. Splits are in `isaac/wm_splits.py`; ice is in `isaac/ice.py`.

```
V2=logs/rsl_rl/bmb_go2_v2/2026-11-06_16-34-51_v2_student/model_999.pt
V1=logs/rsl_rl/unitree_go2_rough/2026-10-12_21-27-17_w1_go2_rough_baseline/model_1499.pt
for s in ice_test ice_train ice_train2 ice_train3; do
  PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python isaac/collect_wm_data.py --split $s --v2 $V2 --v1 $V1 --out_dir logs/wm_data/w6_$s
done
```

## Setup
Same as the geo splits (`experiments/w6_geo_data/`), plus ice patches:
- normal physics: foot μ fixed per episode in [0.6, 1.25], no pushes, no action noise;
- both policies, `(v2, v2, v1)[i % 3]`;
- the v2 terrain mix plus `pits_deep`, on 10 × 20 tiles.

**Ice patches** (`isaac/ice.py`, deterministic from the split seed):
- 80 % of the tiles get Poisson(6) rotated rectangles. Sides are 0.4–2.5 m, the centres are uniform in the tile, and each patch is clipped to its tile.
- Nothing lies within 1.2 m of a tile origin, where envs spawn (± 0.5 m).
- Ice sits on every sub-terrain type, stairs and slopes included. It covers about 12 % of the tile area.
- In `ice_test`, each patch is hidden with probability 0.5. Hidden ice behaves the same physically, but it is drawn as ground, so the model's ice channel is 0 there.

**Ice physics.** Ice is flush with the ground and changes only friction (yaml `terrain_risk.ice.friction` = 0.1, combine multiply).
- Before every policy step, a foot counts as on ice if its position, either now or one step ahead at its current velocity, lies over an ice cell.
- A foot on ice gets friction (episode μ) × 0.1. A foot that leaves the ice goes back to the episode μ.
- This is the same per-foot material write that v2's single-foot friction changes use. Only envs whose feet changed state are written.
- It adds about 1 % to the wall time: 2–3 s per split.
- **Limitation.** The friction is updated at 50 Hz, so a touchdown on ice can see up to 20 ms of normal friction. The lookup one step ahead shortens this.

**Extra files** (besides the geo-split files, including `terrain_height.npy`):
- `ice_mask.npy`: uint8 [4800, 8000] on the height map's grid. 0 = none, 1 = visible, 2 = hidden.
- `ice_patches.json`: the patch list.
- `meta.json` → `ice_map`: grid, spec, cover and seed.
- `foot_ice.npy` [T, N, 4]: the ice code under each foot during step t.

**Checks** (`models/terrain_risk/test_terrain_risk.py`):
- `test_ice_map_layout`: no ice on the border or at spawn points; hiding patches does not move the ice.
- `test_ice_friction_follows_the_map`, on a 4096-env × 150-step smoke run:
  - recorded foot μ / episode μ = 0.100 exactly where a foot is on ice;
  - every foot standing over an ice cell was put on ice.
- **Ice is slippery.** In the same smoke run, the median foot speed while the foot is low and stays over the same surface for two steps is:
  - over ice: 0.51 m/s;
  - off ice: 0.05 m/s;
  - over the same cells in a run without ice physics: 0.06 m/s.

## Splits
Falls are classed by the ice under the feet in the second before the fall. When both kinds occur, the more recent one counts.

| split | seed | envs × steps | wall | falls | v1: none / visible / hidden | v2: none / visible / hidden | frames with a foot on ice |
|---|---:|---|---:|---:|---|---|---:|
| ice_train | 800 | 4096 × 1500 | 386 s | 1 058 | 196 / 240 / 0 | 374 / 248 / 0 | 16.6 % |
| ice_train2 | 810 | 4096 × 1500 | 379 s | 963 | 178 / 263 / 0 | 293 / 229 / 0 | 17.4 % |
| ice_train3 | 820 | 4096 × 1500 | 373 s | 883 | 164 / 233 / 0 | 270 / 216 / 0 | 15.8 % |
| ice_test | 900 | 2048 × 1500 | 211 s | 528 | 101 / 76 / 62 | 167 / 69 / 53 | 16.5 % |

- For comparison, the geo splits without ice have about 640 falls per 4096-env split. Ice adds about 250–420 falls per split.
- Each split takes 2.2–4.3 GB in `logs/wm_data/w6_ice_*`, which is gitignored.
