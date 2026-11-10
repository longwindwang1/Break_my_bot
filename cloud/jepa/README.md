# Cloud run: W7 world-model comparison (JEPA vs the W4–5 ensemble)

Everything here is plain PyTorch: no Isaac Sim, no `env_isaaclab`. The data is the recorded world-model splits.
The protocol is in `experiments/w7_jepa_wm/README.md`, the code in `models/world_model_jepa/`.

## 1. Machine
- **GPU.** One CUDA GPU; PyTorch 2.7 with CUDA 12.8 (needed for RTX 50-series cards).
- **GPU memory per job.** A training job keeps its ~4 GB split on the GPU, plus about 3 GB for a latent dim of 256.
  - With 24–32 GB, set `PARALLEL=3`.
  - With 8 GB, use `PARALLEL=1`. This is the setting on the Linux laptop: peak 6.7 GB at d = 256.
- **Disk.** About 25 GB: 19 GB of data plus runs.
- **Image.** `cloud/jepa/Dockerfile` (`pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` + `requirements.txt`). On a box that already has
  torch 2.7, `pip install -r cloud/jepa/requirements.txt` is enough.

## 2. Code and data
Get the code with `git clone https://github.com/longwindwang1/Break_my_bot.git`, then copy the data into it:

```
# from the Linux training machine, repo root
HOST=user@cloud:Break_my_bot
rsync -avP logs/wm_data/{w4_train_nominal,w4_train,w4_clean,w4_events,w4_events_pits} $HOST/logs/wm_data/    # training + selection (12 GB)
rsync -avP logs/wm_data/{w7_events_test,w7_events_pits_test,w6_ice_test,w6_geo_test} $HOST/logs/wm_data/       # report splits (7 GB)
rsync -avP --relative logs/wm/./2026-11-07_22-45-27_w4_wm_nominal $HOST/logs/wm/                                # deployed ensemble (A, seed 0)
```

| split | size | used for |
|---|---:|---|
| `w4_train_nominal` | 3.9 GB | dynamics training of every model (normal physics only) |
| `w4_train` | 3.9 GB | fall-head refit (full randomization), probes |
| `w4_clean`, `w4_events`, `w4_events_pits` | 4.4 GB | detector selection: false alarms, event detection, hidden pits |
| `w7_events_test`, `w7_events_pits_test` | 2.6 GB | report only: unseen copies of the event splits (seeds 310 / 410) |
| `w6_ice_test`, `w6_geo_test` | 4.4 GB | report only: falls on visible / hidden ice; false alarms on unseen terrain |

## 3. Run
Run everything from the repo root.

Check the setup first (about 1 min):

```
python -m pytest models/world_model_jepa -q
```

Then run the training and the sweep selection, about 7–10 h on one RTX 5090 with `PARALLEL=3` (an estimate from the laptop's speed: 14–21 it/s there):

```
PARALLEL=3 cloud/jepa/run_all.sh
```

Finally the evaluation, about 2–3 h. It can also run on the Linux machine after the runs are copied back:

```
cloud/jepa/run_all.sh report summary probes
```

With Docker:

```
docker build -t bmb-wm -f cloud/jepa/Dockerfile cloud/jepa
docker run --gpus all --shm-size 8g -e PARALLEL=3 -v "$PWD":/work -w /work bmb-wm cloud/jepa/run_all.sh
```

**What the stages run** (see the header of `run_all.sh`):

| stage | jobs | notes |
|---|---|---|
| `sweep` | 12 latent models, seed 0, then a fall-head refit each: vicreg × d {64, 256} × L {1, 3} × prediction weight {25, 100}, and ema × d × L | 20 epochs each, as the ensemble |
| `select` | detector grid per sweep model on the selection splits | |
| `pick` | best configuration by the W5 criterion | key events within 2 s, false alarms ≤ 2 × target → `experiments/w7_jepa_wm/sweep_pick.json` |
| `finals` | B seeds 1–2 (winner), C seeds 0–2 (reconstruction objective, winner's d and L), A seeds 1–2 (ensemble retrained) | |
| `report` | selection + report splits for A, A_cov, B, C (all seeds) | |
| `summary` | tables, paired bootstrap against A, figures | |
| `probes` | friction / ice probes on frozen representations | |

- **Resuming.** A stage skips work whose output exists, so after an interruption just run the same command again.
- **Logs.** Per-job logs are in `experiments/w7_jepa_wm/cloud_logs/`.
- **Smoke run.** For example `SWEEP_CONFIGS="vicreg 64 1 25;ema 64 1 25" EPOCHS=0.005` runs every stage on a tiny sweep, in about 30 min with `PARALLEL=1`; it was checked this way on the laptop. Set `OUT=/tmp/... RUNS=/tmp/...` to keep its runs apart.

## 4. Bring back
```
rsync -avP $HOST/logs/wm/ logs/wm/ --include '*_w7_*/***' --exclude '*'   # runs: model.pt, model_fall.pt, last.pt, history, tensorboard
rsync -avP $HOST/experiments/w7_jepa_wm/ experiments/w7_jepa_wm/           # selections, reports, results.json, figures, logs
```
The run directories are gitignored (`logs/`). `experiments/w7_jepa_wm/` is tracked.
