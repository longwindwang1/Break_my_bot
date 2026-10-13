"""Load configs/robot_interface.yaml and check it against an Isaac Lab run's params/env.yaml.

The baseline policies were trained on the stock Isaac Lab Go2 tasks, so for now the interface file is
validated against what Isaac Lab actually used rather than injected into the env config. Once we
register our own task (first deviation from the stock config), its env cfg should read this file.

Pure Python + PyYAML: importable without launching Isaac Sim.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERFACE_PATH = REPO_ROOT / "configs" / "robot_interface.yaml"

# Keys of an ObservationGroupCfg that are group settings, not observation terms.
_OBS_GROUP_KEYS = {"concatenate_terms", "concatenate_dim", "enable_corruption", "history_length", "flatten_history_dim"}


def load_interface(path: str | Path = INTERFACE_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class _EnvYamlLoader(yaml.SafeLoader):
    """SafeLoader that tolerates the python/* tags Isaac Lab writes (tuples, slices, ...)."""


def _construct_python_tag(loader, tag_suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_EnvYamlLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python_tag)


def load_env_yaml(params_dir: str | Path) -> dict:
    with open(Path(params_dir) / "env.yaml") as f:
        return yaml.load(f, Loader=_EnvYamlLoader)


def resolve_joint_regex(patterns: dict[str, float], joint_names: list[str]) -> list[float]:
    """Resolve an Isaac Lab {regex: value} joint dict to a per-joint list (each joint must match exactly once)."""
    values = []
    for name in joint_names:
        matches = [v for pat, v in patterns.items() if re.fullmatch(pat, name)]
        if len(matches) != 1:
            raise ValueError(f"joint {name!r} matched {len(matches)} patterns in {list(patterns)}")
        values.append(float(matches[0]))
    return values


def _close(a, b, tol=1e-6) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def check_against_env_yaml(params_dir: str | Path, interface: dict | None = None) -> list[str]:
    """Return a list of human-readable mismatches between the interface file and a run's env.yaml."""
    itf = interface or load_interface()
    env = load_env_yaml(params_dir)
    errors: list[str] = []

    def expect(field: str, ours, theirs):
        if not _close(ours, theirs):
            errors.append(f"{field}: interface={ours!r} env.yaml={theirs!r}")

    t = itf["timing"]
    expect("timing.physics_dt", t["physics_dt"], env["sim"]["dt"])
    expect("timing.decimation", t["decimation"], env["decimation"])
    expect("timing.policy_dt", t["policy_dt"], env["sim"]["dt"] * env["decimation"])
    expect("timing.episode_length_s", t["episode_length_s"], env["episode_length_s"])

    robot = env["scene"]["robot"]
    expect(
        "default_joint_pos",
        itf["default_joint_pos"],
        resolve_joint_regex(robot["init_state"]["joint_pos"], itf["joint_names"]),
    )
    expect("init_base_height", itf["init_base_height"], robot["init_state"]["pos"][2])

    act_cfg = env["actions"]["joint_pos"]
    expect("action.scale", itf["action"]["scale"], act_cfg["scale"])
    if not act_cfg["use_default_offset"]:
        errors.append("action.offset: env.yaml does not use default joint offset")
    expect("action.clip", itf["action"]["clip"], act_cfg["clip"])

    actuators = robot["actuators"]
    if list(actuators) != ["base_legs"] or not actuators["base_legs"]["class_type"].endswith(":DCMotor"):
        errors.append(f"actuator: expected a single DCMotor group, got {list(actuators)}")
    else:
        a = actuators["base_legs"]
        for key in ["stiffness", "damping", "effort_limit", "saturation_effort", "velocity_limit", "friction"]:
            expect(f"actuator.{key}", itf["actuator"][key], a[key])

    obs_cfg = env["observations"]["policy"]
    env_terms = [k for k, v in obs_cfg.items() if k not in _OBS_GROUP_KEYS and v is not None]
    is_rough = "height_scan" in env_terms
    our_terms = [term for term in itf["observation"]["terms"] if is_rough or not term.get("rough_only")]
    if [term["name"] for term in our_terms] != env_terms:
        errors.append(f"observation.terms order: interface={[t['name'] for t in our_terms]} env.yaml={env_terms}")
    for term in our_terms:
        env_term = obs_cfg.get(term["name"])
        if env_term is None:
            continue
        if not env_term["func"].endswith(":" + term["func"]):
            errors.append(f"observation.{term['name']}.func: interface={term['func']} env.yaml={env_term['func']}")
        expect(f"observation.{term['name']}.scale", term.get("scale"), env_term["scale"])
        expect(f"observation.{term['name']}.clip", term.get("clip"), env_term["clip"])
    if is_rough:
        hs_params = obs_cfg["height_scan"]["params"]
        expect("height_scan.offset", itf["height_scan"]["offset"], hs_params.get("offset", 0.5))
        scanner = env["scene"]["height_scanner"]
        hs = itf["height_scan"]
        expect("height_scan.sensor_offset", hs["sensor_offset"], scanner["offset"]["pos"])
        expect("height_scan.grid_size", hs["grid_size"], scanner["pattern_cfg"]["size"])
        expect("height_scan.resolution", hs["resolution"], scanner["pattern_cfg"]["resolution"])
        expect("height_scan.update_period", hs["update_period"], scanner["update_period"])
        if scanner["pattern_cfg"]["ordering"] != hs["ordering"]:
            errors.append(f"height_scan.ordering: interface={hs['ordering']} env.yaml={scanner['pattern_cfg']['ordering']}")
        if scanner["ray_alignment"] != "yaw":
            errors.append(f"height_scan: expected ray_alignment yaw, env.yaml has {scanner['ray_alignment']}")

    cmd = env["commands"]["base_velocity"]
    c = itf["commands"]
    for key in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
        expect(f"commands.{key}", c[key], cmd["ranges"][key])
    expect("commands.heading_command", c["heading_command"], cmd["heading_command"])
    expect("commands.heading_control_stiffness", c["heading_control_stiffness"], cmd["heading_control_stiffness"])
    expect("commands.resampling_time_s", [c["resampling_time_s"]] * 2, cmd["resampling_time_range"])

    return errors


if __name__ == "__main__":
    import sys

    for params_dir in sys.argv[1:]:
        problems = check_against_env_yaml(params_dir)
        print(f"{params_dir}: {'OK' if not problems else ''}")
        for p in problems:
            print(f"  - {p}")
