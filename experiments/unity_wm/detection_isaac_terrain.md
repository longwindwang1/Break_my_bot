# World-model surprisal in Unity physics: event detection (Unity vs Isaac)

Events split (2026-11-08 20:59): 196 envs per policy, one 12 s episode each, event i % 7 at step 200; Isaac tiles at difficulty 0–0.5, foot μ 0.6–1.25 before the event, seed 300, PGS. Detection = first alarm within 1 / 2 s after the event (%); delay in policy steps (50 = 1 s). Isaac: experiments/w5_surprisal/results.json. Unity is scored from the same rollouts with the yaml's Isaac thresholds (I-thr) and with Unity's calibration (Isaac z std).

**v1**

| event | envs Isaac / Unity | ≤ 1 s Isaac | Unity I-thr | Unity cal | ≤ 2 s Isaac | Unity I-thr | Unity cal | delay Isaac | Unity I-thr | Unity cal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 194 / 26 | 0 | 0 | 0 | 0 | 0 | 0 | – | – | – |
| mu_0.3 | 194 / 27 | 13 | 22 | 0 | 23 | 41 | 0 | 54 | 27 | – |
| mu_0.1 | 195 / 28 | 88 | 93 | 0 | 90 | 96 | 0 | 14 | 11 | 143 |
| mu_0.05 | 194 / 26 | 96 | 96 | 4 | 96 | 96 | 15 | 9 | 8 | 115 |
| foot_0.05 | 195 / 27 | 21 | 33 | 0 | 48 | 52 | 0 | 62 | 47 | – |
| push_0.5 | 194 / 26 | 97 | 96 | 92 | 97 | 96 | 92 | 1 | 1 | 1 |
| push_1.0 | 195 / 28 | 100 | 100 | 100 | 100 | 100 | 100 | 1 | 1 | 1 |

v1: 8 robots fell before their event (left out), 47 after it.

**v2**

| event | envs Isaac / Unity | ≤ 1 s Isaac | Unity I-thr | Unity cal | ≤ 2 s Isaac | Unity I-thr | Unity cal | delay Isaac | Unity I-thr | Unity cal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 391 / 27 | 0 | 7 | 0 | 0 | 7 | 0 | – | 30 | – |
| mu_0.3 | 390 / 28 | 2 | 14 | 0 | 4 | 21 | 0 | 86 | 41 | – |
| mu_0.1 | 389 / 28 | 61 | 82 | 0 | 81 | 86 | 32 | 34 | 22 | 82 |
| mu_0.05 | 389 / 28 | 94 | 89 | 11 | 96 | 96 | 68 | 20 | 17 | 73 |
| foot_0.05 | 388 / 28 | 5 | 21 | 0 | 21 | 43 | 0 | 85 | 60 | – |
| push_0.5 | 389 / 28 | 99 | 96 | 96 | 99 | 96 | 96 | 1 | 1 | 1 |
| push_1.0 | 390 / 28 | 100 | 100 | 100 | 100 | 100 | 100 | 1 | 1 | 1 |

v2: 1 robots fell before their event (left out), 12 after it.

**All calibration variants**, detection within 2 s (%), per policy:

| z correction | v1 none | v1 mu_0.3 | v1 mu_0.1 | v1 mu_0.05 | v1 foot_0.05 | v1 push_0.5 | v1 push_1.0 | v2 none | v2 mu_0.3 | v2 mu_0.1 | v2 mu_0.05 | v2 foot_0.05 | v2 push_0.5 | v2 push_1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Isaac thresholds (yaml, no Unity correction) | 0 | 41 | 96 | 96 | 52 | 96 | 100 | 7 | 21 | 86 | 96 | 43 | 96 | 100 |
| MAD (handoff §11.4) | 0 | 0 | 0 | 8 | 0 | 92 | 100 | 0 | 0 | 25 | 71 | 0 | 96 | 100 |
| none | 0 | 0 | 0 | 12 | 0 | 96 | 100 | 0 | 0 | 29 | 64 | 0 | 96 | 100 |
| Isaac z std | 0 | 0 | 0 | 15 | 0 | 92 | 100 | 0 | 0 | 32 | 68 | 0 | 96 | 100 |
