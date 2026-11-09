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


def expected_obs_groups(itf: dict, variant: str) -> list[dict]:
    """Observation groups the deployed policy of `variant` consumes, in rsl-rl concatenation order:
    [{"name", "history_length", "terms": [term definition from observation.terms (+ "train_func")]}]."""
    term_defs = {t["name"]: t for t in itf["observation"]["terms"]}
    spec = itf["policies"][variant]
    if "groups" not in spec:  # v1: a single "policy" group
        terms = [t for t in itf["observation"]["terms"] if variant == "rough" or not t.get("rough_only")]
        return [{"name": "policy", "history_length": 0, "terms": terms}]
    groups = []
    for g in spec["groups"]:
        terms = []
        for name in g["terms"]:
            term = dict(term_defs[name])
            if name in g.get("train_func", {}):
                term["train_func"] = g["train_func"][name]
            terms.append(term)
        groups.append({"name": g["name"], "history_length": g.get("history_length", 0), "terms": terms})
    return groups


def policy_obs_dim(itf: dict, variant: str) -> int:
    return sum(t["dim"] * max(1, g["history_length"]) for g in expected_obs_groups(itf, variant) for t in g["terms"])


def obs_term_slices(itf: dict) -> dict[str, slice]:
    """Position of every observation term in the (rough-layout) policy observation vector."""
    out, i = {}, 0
    for t in itf["observation"]["terms"]:
        out[t["name"]] = slice(i, i + t["dim"])
        i += t["dim"]
    return out


def wm_input_blocks(itf: dict) -> list[dict]:
    """world_model.input with dims resolved: [{"name", "dim" (total), "history_length", "terms": [{name, dim}]}]."""
    term_dims = {t["name"]: t["dim"] for t in itf["observation"]["terms"]}
    blocks = []
    for b in itf["world_model"]["input"]:
        if "terms" in b:
            terms = [{"name": n, "dim": term_dims[n]} for n in b["terms"]]
            h = b["history_length"]
            blocks.append({"name": b["name"], "history_length": h, "terms": terms, "dim": h * sum(t["dim"] for t in terms)})
        else:
            blocks.append({"name": b["name"], "history_length": 0, "terms": [], "dim": b["dim"]})
    return blocks


def wm_input_dim(itf: dict) -> int:
    return sum(b["dim"] for b in wm_input_blocks(itf))


def wm_state_dim(itf: dict) -> int:
    term_dims = {t["name"]: t["dim"] for t in itf["observation"]["terms"]}
    return sum(term_dims[n] for n in itf["world_model"]["state_terms"])


def detect_variant(observations: dict, itf: dict) -> str:
    """Which policies.<variant> an Isaac Lab observation config (env.yaml dict or configclass) belongs to."""
    names = set(observations) if isinstance(observations, dict) else set(vars(observations))
    for variant, spec in itf["policies"].items():
        if "groups" in spec and {g["name"] for g in spec["groups"]} <= names:
            return variant
    policy = observations["policy"] if isinstance(observations, dict) else observations.policy
    has_scan = (policy.get("height_scan") if isinstance(policy, dict) else getattr(policy, "height_scan", None)) is not None
    return "rough" if has_scan else "flat"


def _normalize_group_from_env_yaml(group: dict) -> dict:
    terms = [
        {"name": k, "func": v["func"].split(":")[-1], "scale": v["scale"], "clip": v["clip"]}
        for k, v in group.items()
        if k not in _OBS_GROUP_KEYS and isinstance(v, dict) and "func" in v
    ]
    return {"history_length": group.get("history_length") or 0, "terms": terms}


def _normalize_group_from_cfg(group) -> dict:
    terms = [
        {"name": k, "func": v.func.__name__, "scale": v.scale, "clip": v.clip}
        for k, v in vars(group).items()
        if v is not None and hasattr(v, "func") and hasattr(v, "noise")
    ]
    return {"history_length": group.history_length or 0, "terms": terms}


def compare_obs_groups(expected: list[dict], actual: dict[str, dict]) -> list[str]:
    """Compare expected_obs_groups() against normalized actual groups {name: {"history_length", "terms"}}."""
    errors = []
    for g in expected:
        act = actual.get(g["name"])
        if act is None:
            errors.append(f"observation group {g['name']!r} missing")
            continue
        names = [t["name"] for t in g["terms"]]
        if names != [t["name"] for t in act["terms"]]:
            errors.append(f"{g['name']}.terms order: interface={names} env={[t['name'] for t in act['terms']]}")
            continue
        if g["history_length"] != act["history_length"]:
            errors.append(f"{g['name']}.history_length: interface={g['history_length']} env={act['history_length']}")
        for ours, theirs in zip(g["terms"], act["terms"]):
            field = f"{g['name']}.{ours['name']}"
            if theirs["func"] != ours.get("train_func", ours["func"]):
                errors.append(f"{field}.func: interface={ours.get('train_func', ours['func'])} env={theirs['func']}")
            if not _close(ours.get("scale"), theirs["scale"]):
                errors.append(f"{field}.scale: interface={ours.get('scale')} env={theirs['scale']}")
            if not _close(ours.get("clip"), theirs["clip"]):
                errors.append(f"{field}.clip: interface={ours.get('clip')} env={theirs['clip']}")
    return errors


def apply_to_env_cfg(env_cfg, variant: str, itf: dict | None = None) -> None:
    """Write the interface values into an Isaac Lab locomotion env cfg (call at the end of __post_init__)
    and check that its observation groups match policies.<variant>. Raises ValueError on mismatch."""
    itf = itf or load_interface()
    t = itf["timing"]
    env_cfg.sim.dt = t["physics_dt"]
    env_cfg.decimation = t["decimation"]
    env_cfg.sim.render_interval = t["decimation"]
    env_cfg.episode_length_s = t["episode_length_s"]

    robot = env_cfg.scene.robot
    robot.init_state.joint_pos = dict(zip(itf["joint_names"], itf["default_joint_pos"]))
    robot.init_state.pos = (0.0, 0.0, itf["init_base_height"])
    act = robot.actuators["base_legs"]
    for key in ["stiffness", "damping", "effort_limit", "saturation_effort", "velocity_limit", "friction"]:
        setattr(act, key, itf["actuator"][key])

    action = env_cfg.actions.joint_pos
    action.scale = itf["action"]["scale"]
    action.use_default_offset = True
    action.clip = itf["action"]["clip"]

    hs = itf["height_scan"]
    scanner = env_cfg.scene.height_scanner
    scanner.offset.pos = tuple(hs["sensor_offset"])
    scanner.ray_alignment = "yaw"
    scanner.pattern_cfg.size = list(hs["grid_size"])
    scanner.pattern_cfg.resolution = hs["resolution"]
    scanner.pattern_cfg.ordering = hs["ordering"]
    scanner.update_period = hs["update_period"]
    env_cfg.scene.contact_forces.update_period = t["physics_dt"]

    f = itf["friction"]
    material = env_cfg.scene.terrain.physics_material
    material.static_friction = f["terrain_static_friction"]
    material.dynamic_friction = f["terrain_dynamic_friction"]
    material.friction_combine_mode = f["combine_mode"]
    env_cfg.sim.physics_material = material

    c = itf["commands"]
    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = (tuple(c[k]) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z"))
    cmd.heading_command = c["heading_command"]
    cmd.heading_control_stiffness = c["heading_control_stiffness"]
    cmd.resampling_time_range = (c["resampling_time_s"], c["resampling_time_s"])

    actual = {
        name: _normalize_group_from_cfg(group)
        for name, group in vars(env_cfg.observations).items()
        if group is not None and hasattr(group, "concatenate_terms")
    }
    errors = compare_obs_groups(expected_obs_groups(itf, variant), actual)
    if errors:
        raise ValueError("observation config does not match robot_interface.yaml:\n  " + "\n  ".join(errors))


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

    variant = detect_variant(env["observations"], itf)
    env_groups = {
        name: _normalize_group_from_env_yaml(group)
        for name, group in env["observations"].items()
        if isinstance(group, dict)
    }
    errors += compare_obs_groups(expected_obs_groups(itf, variant), env_groups)
    obs_cfg = env["observations"]["policy"]
    is_rough = obs_cfg.get("height_scan") is not None
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

    material = env["scene"]["terrain"]["physics_material"]
    f = itf["friction"]
    expect("friction.terrain_static_friction", f["terrain_static_friction"], material["static_friction"])
    expect("friction.terrain_dynamic_friction", f["terrain_dynamic_friction"], material["dynamic_friction"])
    if material["friction_combine_mode"] != f["combine_mode"]:
        errors.append(f"friction.combine_mode: interface={f['combine_mode']} env.yaml={material['friction_combine_mode']}")

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
