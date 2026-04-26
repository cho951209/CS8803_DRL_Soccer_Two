import os
import pickle

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 4
BASE_PORT = 52000
STAGE0_SWITCH_BASE_REWARD = 1.8
STAGE1_ROTATE_BASE_REWARD = 1.8
USE_REWARD_SHAPING = True
# REWARD_SHAPING_PRESET = "all_features"
REWARD_SHAPING_PRESET = "previous"
REWARD_SHAPING_PRESETS = {
    "previous": {
        "ball_progress_weight": 0.05,
        "ball_velocity_toward_goal_weight": 0.0,
        "player_to_ball_weight": 0.02,
        "movement_weight": 0.0,
        "step_penalty": -0.0005,
    },
    "all_features": {
        "ball_progress_weight": 0.05,
        "ball_velocity_toward_goal_weight": 0.01,
        "player_to_ball_weight": 0.02,
        "movement_weight": 0.001,
        "step_penalty": -0.0005,
    },
}
REWARD_SHAPING_CONFIG = dict(REWARD_SHAPING_PRESETS[REWARD_SHAPING_PRESET])

# Module-level state (lives in trainer actor process after tune.run starts)
_curriculum_stage = 0  # 0 = vs CEIA baseline, 1 = self-play
_initialized = False
_ceia_weights = None  # ceia_baseline_agent weights for frozen benchmark policies


def policy_mapping_fn(agent_id, *args, **kwargs):
    if _curriculum_stage == 0:
        if agent_id in [0, 1]:  # our team
            return "default"
        return "ceia"
    else:  # stage 1: self-play
        if agent_id == 0:
            return "default"
        else:
            return np.random.choice(
                ["default", "opponent_1", "opponent_2", "opponent_3", "ceia"],
                size=1,
                p=[0.40, 0.20, 0.15, 0.15, 0.10],
            )[0]


def default_controlled_agent_ids():
    if _curriculum_stage == 0:
        return (0, 1)
    return (0,)


class CurriculumSelfPlayCallback(DefaultCallbacks):
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        episode.user_data["original_default_reward"] = 0.0
        episode.user_data["shaped_default_reward"] = 0.0

    def on_episode_step(
        self, *, worker, base_env, episode, env_index, policies=None, **kwargs
    ):
        for agent_id in default_controlled_agent_ids():
            info = episode.last_info_for(agent_id)
            if not info:
                continue
            episode.user_data["original_default_reward"] += info.get("base_reward", 0.0)
            episode.user_data["shaped_default_reward"] += info.get("shaped_reward", 0.0)

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        episode.custom_metrics["original_default_reward"] = episode.user_data.get(
            "original_default_reward", 0.0
        )
        episode.custom_metrics["shaped_default_reward"] = episode.user_data.get(
            "shaped_default_reward", 0.0
        )

    def on_train_result(self, **info):
        global _curriculum_stage, _initialized, _ceia_weights

        trainer = info["trainer"]
        reward_mean = info["result"]["episode_reward_mean"]
        custom_metrics = info["result"].get("custom_metrics", {})
        base_reward_mean = custom_metrics.get("original_default_reward_mean", reward_mean)

        if not _initialized:
            init_weights = {}
            if _ceia_weights is not None:
                print(
                    "==== Initializing CEIA benchmark and self-play pool from ceia_baseline_agent checkpoint ===="
                )
                init_weights["ceia"] = _ceia_weights
                init_weights["opponent_1"] = _ceia_weights
                init_weights["opponent_2"] = _ceia_weights
                init_weights["opponent_3"] = _ceia_weights
            if init_weights:
                trainer.set_weights(init_weights)
            _initialized = True

        if _curriculum_stage == 0:
            print(
                f"[Stage 0: vs CEIA Baseline] shaped_reward_mean={reward_mean:.3f} "
                f"base_reward_mean={base_reward_mean:.3f}"
            )
            if base_reward_mean > STAGE0_SWITCH_BASE_REWARD:
                print(
                    f"==== base_reward_mean > {STAGE0_SWITCH_BASE_REWARD:.1f} — switching to self-play! ===="
                )
                _curriculum_stage = 1
                learned_default = trainer.get_weights(["default"])["default"]
                trainer.set_weights(
                    {
                        "opponent_1": learned_default,
                        "opponent_2": learned_default,
                        "opponent_3": learned_default,
                    }
                )

                def _set_stage_1(_):
                    import train_ray_minwoo

                    train_ray_minwoo._curriculum_stage = 1

                trainer.workers.foreach_worker(_set_stage_1)
        else:
            print(
                f"[Stage 1: Self-Play] shaped_reward_mean={reward_mean:.3f} "
                f"base_reward_mean={base_reward_mean:.3f}"
            )
            if base_reward_mean > STAGE1_ROTATE_BASE_REWARD:
                print("---- Rotating opponent snapshots ----")
                trainer.set_weights(
                    {
                        "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                        "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                        "opponent_1": trainer.get_weights(["default"])["default"],
                    }
                )


def load_weights_from_module(agent_module_name: str):
    """Load policy weights from any agent module that exposes CHECKPOINT_PATH, ALGORITHM, POLICY_NAME."""
    agent_ray = __import__(f"{agent_module_name}.agent_ray", fromlist=["agent_ray"])
    checkpoint_path = agent_ray.CHECKPOINT_PATH
    algorithm = agent_ray.ALGORITHM
    policy_name = agent_ray.POLICY_NAME

    if not os.path.exists(checkpoint_path):
        print(f"WARNING: checkpoint not found at {checkpoint_path}")
        return None

    config_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    config["num_workers"] = 0
    config["num_gpus"] = 0
    tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
    config["env"] = "DummyEnv"

    cls = get_trainable_cls(algorithm)
    tmp_trainer = cls(env="DummyEnv", config=config)
    tmp_trainer.restore(checkpoint_path)
    weights = tmp_trainer.get_weights([policy_name])[policy_name]
    tmp_trainer.stop()

    print(f"Loaded weights from {agent_module_name} ({checkpoint_path})")
    return weights


if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"base_port": BASE_PORT})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    print("Loading ceia_baseline_agent weights (benchmark opponent)...")
    _ceia_weights = load_weights_from_module("ceia_baseline_agent")

    analysis = tune.run(
        "PPO",
        name=f"PPO_curriculum_selfplay_{REWARD_SHAPING_PRESET}",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 10,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CurriculumSelfPlayCallback,
            # RL setup
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "ceia": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "base_port": BASE_PORT,
                "reward_shaping": REWARD_SHAPING_CONFIG if USE_REWARD_SHAPING else None,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": 30000000, "time_total_s": 1000000},
        checkpoint_freq=40,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
