"""Robust Go2 policy v2: stock Go2 rough task + friction / payload / COM / gain / push randomization,
pit terrain with a deceptive height scan, and teacher-student observation groups.

Robot/timing/action/actuator/scan values come from configs/robot_interface.yaml (apply_to_env_cfg).
"""

import copy

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

from isaac.interface import apply_to_env_cfg

from . import mdp

HISTORY_LENGTH = 25
# Friction curriculum: lower bound 0.6 -> 0.05 over the first ~1500 PPO iterations (24 env steps each);
# v2 learns ~40% slower than v1 in debug runs (experiments/v2_debug/README.md).
FRICTION_CURRICULUM_STEPS = 36_000
FEET = ".*_foot"


##
# Observations
##


@configclass
class PolicyCfg(ObsGroup):
    """Student / deployed policy input (same terms as v1 rough; the scan may lie during training)."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=mdp.last_action)
    height_scan = ObsTerm(
        func=mdp.height_scan_deceptive,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class ProprioHistoryCfg(ObsGroup):
    """Last HISTORY_LENGTH steps of proprioception (no command), for inferring friction / payload."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = HISTORY_LENGTH
        self.flatten_history_dim = True


@configclass
class TeacherCfg(PolicyCfg):
    """Teacher (and critic) input: the policy terms -- including the deceived scan, so the teacher never
    acts on information the student cannot have -- plus sim-only privileged quantities."""

    foot_friction = ObsTerm(func=mdp.foot_friction)  # [4] mu in [0.05, 1.25]
    added_base_mass = ObsTerm(func=mdp.added_base_mass, scale=0.2)  # [1] kg -> ~[-0.2, 1]
    base_com_offset = ObsTerm(func=mdp.base_com_offset, scale=20.0)  # [3] m -> ~[-1, 1]
    actuator_gain_scale = ObsTerm(func=mdp.actuator_gain_scale, scale=5.0)  # [24] (gain/nominal - 1) -> ~[-1, 1]
    scan_deception_active = ObsTerm(func=mdp.scan_deception_active)  # [1]
    foot_contact_forces = ObsTerm(
        func=mdp.foot_contact_forces,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)},
        scale=0.01,
        clip=(-5.0, 5.0),
    )  # [12] N -> ~[-1, 1] at stance


@configclass
class ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()
    proprio_history: ProprioHistoryCfg = ProprioHistoryCfg()
    teacher: TeacherCfg = TeacherCfg()


##
# Events
##


@configclass
class EventCfg:
    # NOTE: UnitreeGo2RoughEnvCfg.__post_init__ edits add_base_mass / base_external_force_torque /
    # reset_base / reset_robot_joints and sets push_robot / base_com to None. Our own terms therefore use
    # other names, and the stock-named ones exist only to be edited there and disabled afterwards.

    # startup: everything except the feet keeps the stock fixed material (feet are overwritten on reset)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(  # stock name, disabled in Go2V2EnvCfg (replaced by payload_mass)
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.0, 0.0),
            "operation": "add",
        },
    )
    payload_mass = EventTerm(
        func=mdp.randomize_base_mass_cached,
        mode="startup",
        params={"mass_range": (-1.0, 3.0), "asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    payload_com = EventTerm(
        func=mdp.randomize_base_com_cached,
        mode="startup",
        params={
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)},
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )

    # reset
    foot_friction_reset = EventTerm(
        func=mdp.randomize_foot_friction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
            "level_range": (0.05, 1.25),
            "num_levels": 64,
            "lower_start": 0.6,
            "lower_end": 0.05,
            "upper": 1.25,
            "curriculum_steps": FRICTION_CURRICULUM_STEPS,
            "start_step": 0,
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    scan_deception = EventTerm(
        func=mdp.schedule_scan_deception,
        mode="reset",
        params={"prob": 0.3, "duration_s": (1.0, 4.0), "pit_sub_terrain": "pits"},
    )
    base_external_force_torque = EventTerm(  # stock name, disabled in Go2V2EnvCfg
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "force_range": (0.0, 0.0), "torque_range": (0.0, 0.0)},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    # interval
    foot_friction_change = EventTerm(
        func=mdp.randomize_foot_friction,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
            "level_range": (0.05, 1.25),
            "num_levels": 64,
            "lower_start": 0.6,
            "lower_end": 0.05,
            "upper": 1.25,
            "curriculum_steps": FRICTION_CURRICULUM_STEPS,
            "start_step": 0,
            "single_foot_prob": 0.5,  # one foot steps onto a patch vs. the whole surface changes
        },
    )
    push = EventTerm(  # mild in training (stock locomotion values); eval tests up to 1.0 m/s every 3-5 s
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


##
# Environment
##


@configclass
class Go2V2EnvCfg(UnitreeGo2RoughEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.add_base_mass = None  # replaced by payload_mass (caches the value for the teacher)
        self.events.base_external_force_torque = None

        # terrain: add pits with floors (negative + positive blocks); the deceptive scan targets these columns
        gen = copy.deepcopy(self.scene.terrain.terrain_generator)
        gen.sub_terrains["pits"] = terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.15,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.3, 0.8),
            obstacle_height_range=(0.04, 0.16),
            num_obstacles=30,
            platform_width=1.5,
            border_width=0.25,
        )
        self.scene.terrain.terrain_generator = gen

        # rewards (stock Go2 rough terms kept)
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
                "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
            },
        )
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_above_terrain_l2,
            weight=-5.0,
            params={"target_height": 0.32, "sensor_cfg": SceneEntityCfg("height_scanner")},
        )

        # curriculum logging
        self.curriculum.friction = CurrTerm(func=mdp.friction_stats)

        apply_to_env_cfg(self, "v2")


@configclass
class Go2V2EnvCfg_DISTILL(Go2V2EnvCfg):
    """Same randomization, friction curriculum already finished (a new process restarts the step counter)."""

    def __post_init__(self):
        super().__post_init__()
        for term in (self.events.foot_friction_reset, self.events.foot_friction_change):
            term.params["start_step"] = FRICTION_CURRICULUM_STEPS


@configclass
class Go2V2EnvCfg_PLAY(Go2V2EnvCfg_DISTILL):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        for group in (self.observations.policy, self.observations.proprio_history, self.observations.teacher):
            group.enable_corruption = False
        self.events.scan_deception = None
        self.events.push = None
        self.curriculum.friction = None
