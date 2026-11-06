"""RSL-RL configs for robust policy v2: PPO teacher (stage A) and student distillation (stage B).

Both stages share experiment_name, so the distillation run finds the teacher run under
logs/rsl_rl/bmb_go2_v2/. Always pass --load_run and --checkpoint explicitly: Isaac Lab otherwise picks the
alphabetically latest run, which may be a student.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

EXPERIMENT_NAME = "bmb_go2_v2"
HIDDEN_DIMS = [512, 256, 128]


@configclass
class Go2V2TeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = EXPERIMENT_NAME
    obs_groups = {"policy": ["teacher"], "critic": ["teacher"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=HIDDEN_DIMS,
        critic_hidden_dims=HIDDEN_DIMS,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2V2DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    # Rollout storage keeps every obs group; 24 steps x 4096 envs x ~1700 floats ~= 0.7 GB.
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = EXPERIMENT_NAME
    obs_groups = {"policy": ["policy", "proprio_history"], "teacher": ["teacher"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,  # must match the teacher's actor_obs_normalization
        student_hidden_dims=HIDDEN_DIMS,
        teacher_hidden_dims=HIDDEN_DIMS,
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=12,  # num_steps_per_env * num_learning_epochs must be a multiple of this
        max_grad_norm=1.0,
    )
