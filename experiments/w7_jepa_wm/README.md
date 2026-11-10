# W7: a second world model (JEPA-style latent dynamics) against the W4–5 ensemble

**Status (2026-11-09).**
- **Done:** code, unit tests, the local smoke and pipeline checks below, and the two report-only data splits.
- **Pending:** the training runs, on a rented GPU (`cloud/jepa/README.md`, `cloud/jepa/run_all.sh`).
- **Results:** none yet.

## Question
The deployed world model (A, `models/world_model/`, `experiments/w5_surprisal/`) is an ensemble of Gaussian MLPs.
- **How it scores.** It predicts the next 33-d proprioceptive state, and surprisal is a weighted sum of per-dim z².
- **Weak cases** (detection within 2 s at 1 false alarm per 10 min):

  | case | v1 | v2 |
  |---|---:|---:|
  | a single icy foot | 48 % | 21 % |
  | all feet μ → 0.3 | 23 % | 4 % |

  It also alarms within 1 s before only about 45 % of falls on ice.
- **Hypothesis.** A JEPA-style model predicts in a learned latent space, so it can drop what is unpredictable (joint-velocity noise) and keep contact and gait structure. That would make it more sensitive to slipping.
- **Counter-risk.** Slipping is rare in the normal-physics training data, so the encoder may drop it too.

**What we want to know:** at the same data, inputs and false-alarm rate, does a latent model detect better? If it does, is the gain due to the JEPA objective, or only to predicting in a latent space, or to using residual correlations?

## Models
All four use the same dynamics data (`w4_train_nominal`, normal physics only), the same train/val envs, the same inputs (6-frame history, action, height scan) and the same fall-head refit on `w4_train`.

| name | what | rules out |
|---|---|---|
| **A** | W4–5 ensemble: deployed seed 0, plus seeds 1–2 retrained; W5 diagonal weightings | baseline |
| **A_cov** | A's z-scores whitened with their residual covariance (Mahalanobis), estimated on the training split's val envs; no retraining | a gain that only comes from residual correlations |
| **B** | latent model, JEPA objective (`models/world_model_jepa/model.py`) | – |
| **C** | B's architecture with a reconstruction objective instead of VICReg | a gain that only comes from "predict in a latent space" |

**B in detail:**
- **Encoder E.** Maps the last L states (L ∈ {1, 3}) to z (d ∈ {64, 256}). MLP 3 × 512 with LayerNorm.
- **Target encoder Ē.** EMA of E; τ goes from 0.996 to 1 on a cosine schedule.
- **Predictor.** 5 members, each a 3 × 512 trunk. Inputs: the latents of the last 6 frames, their actions, a_t and the scan. Output: ẑ_{t+1} = z_t + Δ.
- **Loss.**
  - The JEPA prediction MSE against sg(Ē(window ending t + k)), rolled out K = 5 steps in latent space.
  - VICReg variance and covariance terms on E's output (`reg vicreg`); `reg ema` keeps only the EMA asymmetry plus an output LayerNorm.
  - A log-variance head and a displacement head on detached features; they cannot shape the representation.
- **Surprisal.** r = (Ē(window ending u) − μ) / √(var + var_floor), with the same statistics as the ensemble (norm or nll, clip, CUSUM, spike on max |r|). An `maha_all` variant whitens r.

## Protocol
- **Selection.** Each model's detector grid (statistic, weighting, clip, CUSUM k) and the B sweep (12 configs: vicreg × d {64, 256} × L {1, 3} × prediction weight {25, 100}, and ema × d × L) use only `w4_clean` (false alarms: h from half A, checked on half B), `w4_events` and `w4_events_pits`. These are the splits the deployed A was calibrated on.
- **Criterion.** The W5 one: mean detection within 2 s over μ → 0.1, μ → 0.05, a single foot μ → 0.05, and pushes of 0.5 and 1.0 m/s, with false alarms on half B at most 2 × the target.
- **Report.** All numbers come from splits never used for a choice:

  | split | collected | content | measured |
  |---|---|---|---|
  | `w7_events_test` | new | 4096 × 600, seed 310 | detection per event and policy, delays, detection vs false-alarm curves |
  | `w7_events_pits_test` | new | 2048 × 600, seed 410 | alarm rate with a deceived vs a truthful scan |
  | `w6_ice_test` | existing | – | alarms before falls on visible, hidden and no ice; ice contacts; false alarms off ice |
  | `w6_geo_test` | existing | unseen terrain, normal physics | false alarms (a player builds new levels) |

  The new splits copy the conditions of `events` and `events_pits` (`isaac/wm_splits.py` `SPLIT_KIND`). Their collection logs are `logs/wm_data/w7_*.log`:

  | split | falls | other |
  |---|---:|---|
  | `w7_events_test` | 248 | 585–586 envs per event type; 6 envs reset before the event |
  | `w7_events_pits_test` | 569 | 1 061 envs deceived |

- **Also reported:** fall-head AUROC, the ghost-path error (open-loop base position at 1–50 steps), and probes (`probes.py`): min foot μ, any foot icy, which foot is icy, and base velocity, from frozen representations.
- **Comparison.** A against B and C on the same envs and the same falls: a paired bootstrap of the seed-averaged hit rate per env (events) or per fall (ice); 3 seeds each.

## Local checks (Linux laptop, RTX 4070 8 GB)
- **Tests.** `python -m pytest models/world_model_jepa` passes 7 tests:
  - window gathering, including the episode-start fill (the same as the ensemble's history) and the validity chain;
  - streaming prediction equals batched prediction;
  - causality: changing future frames leaves past latents and predictions unchanged;
  - hand-computed scores;
  - the EMA update;
  - VICReg and effective rank;
  - the Mahalanobis whitening.
- **Regression.** The generalized `models/world_model/evaluate.py` (scorer interface, combos, maha weightings, `first_alarm`) re-calibrates the deployed A to an identical `calibration.yaml`.
- **Speed and memory** at batch 1024 × 5 members on `w4_train_nominal`: 21 it/s at d = 128 and 14 it/s at d = 256, peak 6.7 GB. A full 20-epoch run (114k iterations) therefore takes 1.5–2.3 h here, hence the cloud.
- **Pipeline check.** `compare.py` (select → report → summary) ran end to end on A and on a 3 000-iteration B.
  - A reproduces its W5 / W6 numbers:
    - key detection within 2 s: 0.829 on `w4_events`, 0.833 on the unseen `w7_events_test`;
    - alarm within 1 s before falls on visible / hidden ice in `w6_ice_test`: 46 % / 45 %.
  - A_cov, on its first run, got 0.845 on `w7_events_test`; ice falls 56 % / 49 %.
- **Cloud script.** `cloud/jepa/run_all.sh sweep select pick finals probes` ran end to end locally, with `SWEEP_CONFIGS="vicreg 64 1 25;ema 64 1 25" EPOCHS=0.005` (9 runs, all with `model_fall.pt`). Running it again skips the finished work in 0.1 s. The report stage is the `compare.py` path from the pipeline check.
- **Pilots.** Short local runs (vicreg, d = 128, L = 3), scored on the selection splits only. They are a sanity check, not results: a full run has 114k iterations.

  | run | iterations | 1-step error / "latent stays" | effective rank | key detection 2 s (`w4_events`) |
  |---|---:|---:|---:|---:|
  | w_pred 25 | 3 000 | 0.12 | 94 / 128 | 0.056 |
  | w_pred 25 | 20 000 | 0.09 | 121 / 128 | 0.285 |
  | w_pred 5 | 20 000 | 0.12 | 127 / 128 | 0.207 |
  | A (deployed, full training) | 102 000 | – | – | 0.829 |

  - **Training length matters most.** Detection rises about 5× from 3k to 20k iterations.
  - **Ice raises every latent dim.** At 20k iterations (w_pred 25), μ → 0.05 raises mean r² 3.6× (0.56 → 2.03) and a single icy foot 2.2×, with more than 2× in most dims.
  - **Per-step signal is weaker than A's.**

    | standardized score 1–3 s after the event | B, 20k iterations | A |
    |---|---:|---:|
    | μ → 0.05 | 1.2 | 2.5–3.7 |
    | single foot μ → 0.05 | 0.44–0.49 | 1.1 |

    Pushes are caught only 33–70 % of the time, against about 100 % for A, which has a spike detector on base velocity.
  - **What the encoder looks at.** Its gradient, in normalized state units, comes mostly from base_lin_vel (71 %) and joint_pos (19 %); joint_vel gets only 3 %. Slipping shows mainly in joint velocities, which A's best weighting uses. So the JEPA objective dropped the hard-to-predict joint velocities, as the counter-risk predicted.
  - **Lower prediction weight did not help.** w_pred 5 roughly doubles the joint share (joint_vel 6 %, joint_pos 29 %), but detection fell to 0.21. The sweep therefore tries w_pred 100 instead of 5.
- **Probes** (`probes.py`, all 4096 envs of `w4_train`, split by env; 20k pilot vs A).

  | target | A, trunk | B, z | B, trunk |
  |---|---:|---:|---:|
  | base velocity (R²) | 0.99 | 0.95–0.98 | 0.93–0.97 |
  | any foot icy (AUROC) | 0.69–0.73 | 0.61–0.68 | 0.64–0.68 |
  | min foot μ (R²) | 0.09–0.10 | 0.03–0.06 | 0.03–0.06 |
  | which foot is icy (4 classes) | 0.34–0.35 | 0.28–0.31 | 0.28–0.30 |

  Neither representation encodes friction well, which is expected: both were trained on normal physics only.

## Files
| file | what |
|---|---|
| `models/world_model_jepa/model.py` | the latent model (E, Ē, predictor members, heads), VICReg, effective rank |
| `models/world_model_jepa/data.py` | window gathering from WMData |
| `models/world_model_jepa/train.py` | training; writes `logs/wm/<time>_<name>/model.pt` with var_floor and the whitening matrix |
| `models/world_model_jepa/refit_heads.py` | fall-head refit on `w4_train` → `model_fall.pt` |
| `models/world_model_jepa/predict.py` | streaming scores, LatentScorer (calibrate-compatible), ghost path |
| `models/world_model_jepa/compare.py` | select / report / summary for any mix of ensemble and latent models |
| `models/world_model_jepa/probes.py` | probes on frozen representations |
| `models/world_model_jepa/sweep_pick.py` | the sweep winner |
| `models/world_model_jepa/diagnose.py` | encoder sensitivity per window frame / state term; latent residual per event type |
| `cloud/jepa/` | Dockerfile, requirements, `run_all.sh` (all stages, resumable), README |

After the cloud run, this directory holds:
- `models/<name>/{selection,report}.json` and `arrays.npz`;
- `sweep_pick.{txt,json}`;
- `results.json`, `probes.json`;
- `detection_vs_false_alarms.png`, `event_traces.png`;
- `cloud_logs/`.
