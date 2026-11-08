# W6 terrain-risk data (geo splits)

Rollouts for the terrain-risk ("vision") model, recorded by `isaac/collect_wm_data.py`. Split definitions are in `isaac/wm_splits.py`.

```
V2=logs/rsl_rl/bmb_go2_v2/2026-11-06_16-34-51_v2_student/model_999.pt
V1=logs/rsl_rl/unitree_go2_rough/2026-10-12_21-27-17_w1_go2_rough_baseline/model_1499.pt
for s in geo_train geo_train2 geo_train3 geo_test; do
  PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python isaac/collect_wm_data.py --split $s --v2 $V2 --v1 $V1 --out_dir logs/wm_data/w6_$s
done
```

## Setup
Same recording format as `experiments/w4_wm_data/`, with two extra files per split.
- **Conditions:** normal physics (foot μ fixed per episode in [0.6, 1.25], no pushes, no action noise) and both policies (`(v2, v2, v1)[i % 3]`).
- **Commands:** the training distribution.
- **Terrain:** the v2 training mix plus `pits_deep`, with blocks of 15–30 cm, 0.15 of the mix, and a truthful scan. The other sub-terrains are stairs (5–23 cm steps), inverted stairs, boxes, random rough, slopes, and pits of 4–16 cm.
  - 10 × 20 tiles of 8 m with random type and difficulty 0–1.
  - Each split has its own terrain seed, so geo_test's terrain was never trained on.
- **Extra files:**
  - `terrain_height.npy` is the global height map (`isaac/terrain_map.py`). It covers 120 × 200 m including the border at 2.5 cm, as float16, 4800 × 8000. Cell (i, j) is centred at (x_min + (i + .5)·res, y_min + (j + .5)·res); axis 0 is x.
  - It is made by ray-casting the same warp mesh as the height scanner. 0 rays missed.
  - `feet_pos.npy` holds the feet world positions per step, for the visible-ice phase.
- **Check:** on the smoke split, the map sampled at the recorded height-scan rays matches Isaac's own scan values. Over 7.6M rays the median error is 0.05 mm, p95 2.3 mm and p99 4.8 mm. The worst case is 0.27 m at step edges, from nearest-cell lookup. This is `models/terrain_risk/test_terrain_risk.py::test_height_map_matches_isaac_scans`.

| split | seed | envs × steps | wall | falls (v1 / v2) | where the falls are |
|---|---:|---|---:|---|---|
| geo_train | 600 | 4096 × 1500 | 375 s | 639 (246 / 393) | inverted stairs 256, deep pits 191, stairs 100, boxes 44 |
| geo_train2 | 610 | 4096 × 1500 | 376 s | 674 (272 / 402) | deep pits 243, inverted stairs 238, stairs 125, boxes 35 |
| geo_train3 | 620 | 4096 × 1500 | 367 s | 627 (246 / 381) | deep pits 258, inverted stairs 149, stairs 137, boxes 51 |
| geo_test | 700 | 2048 × 1500 | 233 s | 323 (135 / 188) | inverted stairs 119, deep pits 90, stairs 60, boxes 31 |

- The data was collected at commit cd439d5. The collector changes (wm_splits, height map, feet) were committed with W6.
- Each split takes about 3.5 GB plus 77 MB of map, in `logs/wm_data/w6_*` (gitignored).
