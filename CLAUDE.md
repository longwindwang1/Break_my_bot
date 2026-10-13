# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Break My Bot

Course project (Physical AI / world models / games), 10 weeks.
A human player builds obstacle levels and disrupts a Unitree Go2 robot dog in real time;
the robot uses a learned world model to predict outcomes, detect physics changes (surprisal),
and adapt online.

## Current status
- Phase: W3 (smoke test)          <!-- update at the start of every phase -->
- Last milestone done: W3 Python side — `configs/robot_interface.yaml`, ONNX export, parity tests passing.
  Next: Unity side (Unity not installed on this machine yet). Visual check of both policies in play.py still pending.

## Architecture
Isaac Lab + PyTorch (train) → ONNX (export) → Unity + Inference Engine (play)

- **Isaac Lab**: locomotion policy (PPO), adaptation module, world model data collection
- **PyTorch**: world model + adaptation module training
- **Unity**: level editor, disruption controls, surprisal HUD, ghost-path preview;
  runs all three ONNX models in Unity physics (this is also our sim-to-sim transfer test)

## Repo layout
```
isaac/        training wrapper, reference recorder, interface.py (yaml loader + env.yaml checker)
models/       world model, adaptation module, training code (PyTorch)
export/       ONNX export + parity tests (exported models go to export/models/, gitignored)
unity/        Unity project (C# scripts under unity/Assets/Scripts)
configs/      shared configs, incl. robot_interface.yaml
experiments/  results, plots, notes per experiment
docs/         design notes
```

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
  Current state: the baselines use the stock Isaac Lab tasks, so the yaml is *checked* against each run's
  `params/env.yaml` (`isaac/interface.py`) rather than injected. When we first deviate from the stock config
  (e.g. re-enabling pushes), register our own task whose env cfg reads the yaml.
- Any change to the interface must update the parity test in `export/` (same input → same output
  in PyTorch and ONNX; same obs vector in Isaac Lab and Unity).
- World model runs as a side monitor at 50 Hz; it never produces actions.
- Training scripts must support headless mode and a configurable `num_envs`.
- Fix random seeds and log every run (config + seed + git hash).

## Hardware
- Local: RTX 4070 Laptop (8 GB), development and single runs. Go2 rough at 4096 envs peaks at ~5.9 GB VRAM
  and ~13.7k steps/s (needs `nvidia-powerd` running, else GPU caps at 55 W); see `experiments/w1_vram_bench/`.
- Python env: conda `env_isaaclab` (Isaac Sim 5.1, Isaac Lab 0.54.2 at `~/IsaacLab`, torch 2.7 cu128, rsl-rl 3.1.2).
  `git` is only installed inside this env; rsl-rl needs it on PATH (the train wrapper handles this).
- Cloud: rented RTX 5090, only for multi-seed batch experiments. Keep scripts runnable via Docker.
- RTX 50 series needs a recent Isaac Sim / Isaac Lab and a CUDA 12.8+ PyTorch build.

## Commands
Run from the repo root. Isaac Lab writes checkpoints and `params/{env,agent}.yaml` to `logs/rsl_rl/<exp>/<time>_<run>/`
(cwd-relative); the wrapper writes `experiments/<exp_name>/run_info.txt` (seed, git hash, peak VRAM, log dir).
- Train policy (headless): `isaac/train_policy.sh <task> <num_envs> <seed> <max_iter> <exp_name> [extra train.py args]`
  - tasks: `Isaac-Velocity-Rough-Unitree-Go2-v0`, `Isaac-Velocity-Flat-Unitree-Go2-v0` (Play variants: `-Play-v0`)
  - quick smoke test: `isaac/train_policy.sh Isaac-Velocity-Flat-Unitree-Go2-v0 512 0 5 scratch_smoke`
- Play + export ONNX (export happens in play.py, not train.py → `<log_dir>/exported/policy.onnx`):
  `PATH=~/miniforge3/envs/env_isaaclab/bin:$PATH python ~/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-Play-v0 --load_run <time>_<run>`
- TensorBoard: `~/miniforge3/envs/env_isaaclab/bin/tensorboard --logdir logs/rsl_rl`
- Collect world model data: `TODO`
- Train world model: `TODO`
- Export ONNX: `~/miniforge3/envs/env_isaaclab/bin/python export/export_onnx.py --run <log_dir> --variant flat|rough`
  → `export/models/go2_<variant>.onnx` (input `obs` [1,48|235], output `actions` [1,12])
- Parity tests: `~/miniforge3/envs/env_isaaclab/bin/python -m pytest -v`; one test: `... -m pytest -k "obs_layout and rough"`
- Check yaml vs a run: `~/miniforge3/envs/env_isaaclab/bin/python isaac/interface.py <log_dir>/params`
- Record reference rollout (needs Isaac Sim, ~1 min): see `experiments/w3_reference/README.md`

## Working rules for Claude
- Work on one milestone at a time; use plan mode before multi-file changes.
- Ask before installing packages or modifying Isaac Lab source.
- Unity editor/scene setup is done manually by us; you write C# scripts and tell us what to wire up.
- An `isaac-sim` MCP server is connected: use it to inspect and debug scenes
  (prim sizes, joint config, stepping). Training itself runs headless from scripts, not through MCP.
- Put experiment outputs in `experiments/<name>/` with a short README of what was run.
- Update "Current status" above when a milestone is finished.

## Milestones
- [x] W1–2 Baseline: Isaac Lab setup, Go2 PPO policy, measure VRAM/speed locally
- [ ] W3 Smoke test: flat-ground policy running in Unity, interface parity verified
- [ ] W4–5 World model: data collection, ensemble training, surprisal + CUSUM
- [ ] W6–8 Game: adaptation trigger, level editor, disruptions, HUD, ghost path
- [ ] W9–10 Results: 3 experiments, demo video, report

## Experiments
1. Robustness: success rate under disruption, with vs. without surprisal trigger
2. Foresight: fall prediction lead time and accuracy
3. Transfer: success rate retained from Isaac Lab to Unity physics

## Stretch (only after core loop works)
Level Scout AI (world model screens candidate levels) · MuJoCo sim-to-sim · C++ ROS 2 inference node
