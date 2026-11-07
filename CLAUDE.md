# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Break My Bot

Course project (Physical AI / world models / games), 10 weeks.
A human player builds obstacle levels and disrupts a Unitree Go2 robot dog in real time;
the robot uses a learned world model to predict outcomes, detect physics changes (surprisal),
and adapt online.

## Current status
- Phase: W4–5 (world model)       <!-- update at the start of every phase -->
- Last milestone done: W3 smoke test, both sides. Python: `configs/robot_interface.yaml`, ONNX export, parity tests.
  Unity (`experiments/w3_unity/`): flat, rough and v2 walk in Unity physics; verification levels 1–4 of
  `docs/unity_handoff.md` §8 pass (ONNX ≤ 2.2e-6, obs ≤ 3.6e-7, kinematics ≤ 0.006 mm); 44 EditMode tests.
  Unity vs Isaac gait statistics in `experiments/w3_unity/trials.md` (Unity's gait period 15–18 % longer for flat/v2).
  Robust policy v2 (`BMB-Go2-Rough-v2`, teacher-student) done: `export/models/go2_v2.onnx` (student, 1360-d input),
  parity tests pass; at foot μ 0.1 it survives 90 % vs 52 % for v1 (`experiments/v2_eval/`).
  Open: pit / deceptive-scan test does not discriminate (pits too shallow); pushes not better than v1.

## Two machines (this file is shared by both)
- **Linux** (training): Isaac Lab, PyTorch, ONNX export, parity tests. Every command under "Commands" assumes it.
- **Windows** (Unity): Unity 6000.3.x editors. No Isaac Sim or `env_isaaclab`, and the system Python has no numpy,
  so the Isaac/export commands and pytest don't run there (the old `~/IsaacLab` clone on it is not a working install).
  Work there is Unity only: building terrain, importing the Go2 and other models, scene setup, C# scripts. Claude
  drives the editor through a Unity MCP server. Read `docs/unity_handoff.md` first (the Unity-side spec); `git pull`
  to get new models and references. Physics, friction and robot settings applied through the MCP come from
  `configs/robot_interface.yaml` / handoff §5–7, never typed in from memory.
  `*.onnx` is gitignored except under `export/models/`; Unity's copies (`unity/Assets/Resources/BMB/Models/`,
  made by `BMB/Sync models`) are gitignored on purpose.
  Unity work happens on the `unity-w3` branch until merged; it merges `origin/main` to pick up new models.

## Architecture
Isaac Lab + PyTorch (train) → ONNX (export) → Unity + Inference Engine (play)

- **Isaac Lab**: locomotion policy (PPO), adaptation module, world model data collection
- **PyTorch**: world model + adaptation module training
- **Unity**: level editor, disruption controls, surprisal HUD, ghost-path preview;
  runs all three ONNX models in Unity physics (this is also our sim-to-sim transfer test)

## Repo layout
```
isaac/        launchers (train.py/play.py), our tasks (bmb_tasks/), training wrapper, reference recorder,
              eval_policy.py, interface.py (yaml loader, env.yaml checker, apply_to_env_cfg)
models/       world model, adaptation module, training code (PyTorch)  [not created yet]
export/       ONNX export + parity tests; deployable models in export/models/ (tracked for the Unity machine)
unity/        Unity 6000.3.8f1 project: Assets/Scripts/{Runtime,Editor,Tests/EditMode}, Assets/Robots (URDF, Go2.prefab),
              Assets/Scenes (W3_Flat, W3_Rough)
configs/      shared configs, incl. robot_interface.yaml
experiments/  results, plots, notes per experiment
docs/         design notes
```

## How the Python side fits together
- `isaac/train.py` / `isaac/play.py` run Isaac Lab's stock rsl_rl scripts through `isaac/_launch.py`, which imports
  `isaac.bmb_tasks` first so the `BMB-*` gym ids exist before hydra looks them up. Use them, not
  `~/IsaacLab/scripts/...`, for any BMB task. `ISAACLAB_DIR` / `CONDA_ENV` override the default paths.
- `isaac/interface.py` is pure Python + PyYAML (importable without Isaac Sim) and is the only Python reader of the yaml.
- v2 (`isaac/bmb_tasks/go2_v2_env_cfg.py`, `mdp/`) has three obs groups: `policy` (deployed terms; the height scan
  may lie during training), `proprio_history` (25 steps), and `teacher` (privileged, sim-only). The teacher PPO
  trains on `teacher`; the student distills onto `policy` + `proprio_history` (1360-d), which is what ships.
  `-Distill` = same env with the friction curriculum already finished; `-Play` = small scene, clean obs.
- `export/export_onnx.py` loads checkpoints into rsl-rl modules and imports Isaac Lab's exporter by file path, so it
  runs without Isaac Sim. Output has a static batch of 1 and metadata (variant, obs_dim, yaml hash).
- `export/test_parity.py::build_obs` is the reference observation builder that the Unity C# must reproduce.

## How the Unity side fits together
- `BMB.Runtime`: `RobotInterface` reads the yaml straight from the repo (YamlDotNet); `IsaacFrame` converts
  Unity ↔ Isaac axes (polar vs axial vectors); `ObservationBuilder` ports `build_obs` (incl. v2 history);
  `HeightScanner`, `DCMotor`, `PolicyRunner` (Inference Engine, CPU); `Go2Body` (articulation, policy joint order);
  `Go2Controller` runs the §5.1 loop in FixedUpdate, or headless via `ExternallyStepped` + `ControlStep()`.
- Anything that must match the yaml or the Isaac model is an idempotent editor menu (`BMB/...` in
  `Assets/Scripts/Editor`), which the MCP calls; scenes are rebuilt by `W3Scenes`, the robot by `Go2Setup`.
- `Go2.prefab` reproduces Isaac's 19-body model, not the raw URDF (which adds rotor links and 1 kg defaults).
  ArticulationBody velocity caps / solver iterations are not serialized: `Go2Body.ApplyRuntimeSettings`.
- Reference data (`experiments/w3_reference/*.json`) is read from the repo by the tests; nothing is duplicated into Assets
  except the ONNX copies Unity must import.

## Core specs
- Robot: Unitree Go2, same official URDF in Isaac Lab and Unity
- Base task: `Isaac-Velocity-Rough-Unitree-Go2-v0` (config: `~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/`)
- Control: physics 200 Hz, policy every 4 physics steps (50 Hz)
- Observation / action / actuator: exact definitions in `configs/robot_interface.yaml`. Gotchas:
  - joint order is grouped by type (4 hips, 4 thighs, 4 calves), not by leg
  - base_lin_vel is the COM velocity in the base frame; height scan is clipped to [-1, 1]
  - actuator is an explicit DC motor (Kp 25, Kd 0.5, torque-speed saturation) computed every physics step
- Note: stock Go2 rough config sets `push_robot = None` and `base_com = None`; pushes must be re-enabled for our DR
- Policy: PPO + terrain curriculum + domain randomization (friction, mass, pushes, payload)
- Adaptation: RMA-style module, estimates environment latent from recent history
- World model: ensemble of 5 Gaussian MLPs (3 x 512 each, ~4M params total)
  - input: state + last 5 steps of history + action
  - output: mean & log-variance of next-state delta, base displacement, fall probability
  - multi-step rollout loss (5–10 steps); check variance calibration on clean data
- Surprisal: S_t = −log p(s_{t+1} | s_t, a_t); CUSUM accumulation;
  threshold set from a target false-alarm rate on clean data
- On alarm: clear adaptation history, re-estimate quickly, switch to a cautious gait

## Hard rules
- **Single source of truth**: obs ordering, action scaling, PD gains, dt and decimation live in
  `configs/robot_interface.yaml`. Isaac Lab code and Unity C# must both read from it. Never hardcode these.
  Current state: the v1 baselines use the stock Isaac Lab tasks, so the yaml is *checked* against each run's
  `params/env.yaml` (`check_against_env_yaml`). Our `BMB-*` tasks *inject* it (`apply_to_env_cfg(self, variant)` at
  the end of the env cfg's `__post_init__`, which also raises if the obs groups disagree); new tasks must do the same.
  The Unity side only reads the yaml and never edits it.
- Any change to the interface must update the parity test in `export/` (same input → same output
  in PyTorch and ONNX; same obs vector in Isaac Lab and Unity).
- World model runs as a side monitor at 50 Hz; it never produces actions.
- Training scripts must support headless mode and a configurable `num_envs`.
- Fix random seeds and log every run (config + seed + git hash).

## Hardware
- Linux machine: RTX 4070 Laptop (8 GB), development and single runs. Go2 rough at 4096 envs peaks at ~5.9 GB VRAM
  and ~13.7k steps/s (needs `nvidia-powerd` running, else GPU caps at 55 W); see `experiments/w1_vram_bench/`.
- Python env: conda `env_isaaclab` (Isaac Sim 5.1, Isaac Lab 0.54.2 at `~/IsaacLab`, torch 2.7 cu128, rsl-rl 3.1.2).
  `git` is only installed inside this env; rsl-rl needs it on PATH (the train wrapper handles this).
- Cloud: rented RTX 5090, only for multi-seed batch experiments. Keep scripts runnable via Docker.
- RTX 50 series needs a recent Isaac Sim / Isaac Lab and a CUDA 12.8+ PyTorch build.

## Commands
Run from the repo root. Isaac Lab writes checkpoints and `params/{env,agent}.yaml` to `logs/rsl_rl/<exp>/<time>_<run>/`
(cwd-relative); the wrapper writes `experiments/<exp_name>/run_info.txt` (seed, git hash, peak VRAM, log dir).
- Train policy (headless): `isaac/train_policy.sh <task> <num_envs> <seed> <max_iter> <exp_name> [extra train.py args]`
  (runs `isaac/train.py` = Isaac Lab's train.py with our `BMB-*` tasks registered; `isaac/play.py` likewise)
  - tasks: `Isaac-Velocity-Rough-Unitree-Go2-v0`, `Isaac-Velocity-Flat-Unitree-Go2-v0` (Play variants: `-Play-v0`),
    `BMB-Go2-Rough-v2` (teacher PPO), `BMB-Go2-Rough-v2-Distill` (student: add
    `--agent rsl_rl_distillation_cfg_entry_point --load_run <teacher run> --checkpoint model_N.pt`), `BMB-Go2-Rough-v2-Play`
  - always pass `--load_run`/`--checkpoint` for v2: teacher and student share `logs/rsl_rl/bmb_go2_v2/`
  - quick smoke test: `isaac/train_policy.sh Isaac-Velocity-Flat-Unitree-Go2-v0 512 0 5 scratch_smoke`
- Play: `PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python isaac/play.py --task <...-Play[-v0]> --load_run <time>_<run> --checkpoint model_N.pt`
  (also writes Isaac Lab's own `<log_dir>/exported/policy.onnx`; the models Unity uses come from `export/export_onnx.py`)
- TensorBoard: `~/miniforge3/envs/env_isaaclab/bin/tensorboard --logdir logs/rsl_rl`
- Collect world model data: `TODO`
- Train world model: `TODO`
- Export ONNX: `~/miniforge3/envs/env_isaaclab/bin/python export/export_onnx.py --run <log_dir> --variant flat|rough|v2 [--checkpoint model_N.pt] [--opset 15]`
  → `export/models/go2_<variant>.onnx` (input `obs` [1,48|235|1360], output `actions` [1,12]; v2 exports the student;
  opset 18 default, 15 for older Unity Inference Engine versions)
- Parity tests: `~/miniforge3/envs/env_isaaclab/bin/python -m pytest -v`; one test: `... -m pytest -k "obs_layout and rough"`.
  Driven by `experiments/w3_reference/go2_<variant>.npz`; tests that need the training run (`logs/`, gitignored)
  skip when it's missing. The module imports torch/rsl_rl/onnxruntime, so it only runs in `env_isaaclab`.
- Check yaml vs a run: `~/miniforge3/envs/env_isaaclab/bin/python isaac/interface.py <log_dir>/params`
- Record reference rollout (needs Isaac Sim, ~1 min): see `experiments/w3_reference/README.md`;
  then `python export/npz_to_json.py <npz>` for the Unity-readable JSON
- Evaluate policies side by side (scenarios: friction levels, friction drop, pushes, pits ± deceptive scan):
  `python isaac/eval_policy.py --scenarios all --policy name=v1|teacher|student:<ckpt> ... --out experiments/v2_eval/results.jsonl`
  (one subprocess per scenario; `python isaac/eval_summary.py <results.jsonl>` reprints the table)

Unity (Windows; editor menu or MCP `execute_menu_item`):
- After pulling new models: `BMB/Sync models`; after yaml / reference changes: `BMB/Apply interface settings`, `BMB/Setup Go2`
- Scenes: `BMB/Build W3_Flat scene`, `BMB/Build W3_Rough scene` (Play needs the editor in the foreground)
- Unity vs Isaac report: `BMB/Headless trials/Run all and write report` → `experiments/w3_unity/trials.md`
- Tests: MCP `run_tests` (EditMode, assembly `BMB.Tests.EditMode`), or with the project closed:
  `Unity.exe -batchmode -projectPath unity -runTests -testPlatform EditMode -testResults results.xml`.
  Keep a saved scene open: a modified untitled scene makes the test runner block on a save dialog.

## Working rules for Claude
- Work on one milestone at a time; use plan mode before multi-file changes.
- Ask before installing packages or modifying Isaac Lab source.
- Unity editor/scene setup (terrain, model import, components, physics settings) is done by Claude through the
  Unity MCP on the Windows machine. Anything the MCP can't do: tell us exactly what to set up by hand.
- On the Linux machine an `isaac-sim` MCP server is connected: use it to inspect and debug scenes
  (prim sizes, joint config, stepping). Training itself runs headless from scripts, not through MCP.
- Put experiment outputs in `experiments/<name>/` with a short README of what was run.
- Update "Current status" above when a milestone is finished.

## Milestones
- [x] W1–2 Baseline: Isaac Lab setup, Go2 PPO policy, measure VRAM/speed locally
- [x] W3 Smoke test: flat-ground policy running in Unity, interface parity verified (also rough and v2)
- [ ] W4–5 World model: data collection, ensemble training, surprisal + CUSUM
- [ ] W6–8 Game: adaptation trigger, level editor, disruptions, HUD, ghost path
- [ ] W9–10 Results: 3 experiments, demo video, report

## Experiments
1. Robustness: success rate under disruption, with vs. without surprisal trigger
2. Foresight: fall prediction lead time and accuracy
3. Transfer: success rate retained from Isaac Lab to Unity physics

## Stretch (only after core loop works)
Level Scout AI (world model screens candidate levels) · MuJoCo sim-to-sim · C++ ROS 2 inference node
