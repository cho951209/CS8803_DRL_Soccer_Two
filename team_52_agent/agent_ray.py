import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface


ALGORITHM = "PPO"
POLICY_NAME = "default"  # this may be useful when training with selfplay


def _find_latest_checkpoint(experiment_name: str):
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_root = os.path.join(agent_dir, "ray_results", experiment_name)
    if not os.path.isdir(experiment_root):
        return None

    trial_dirs = [
        os.path.join(experiment_root, dirname)
        for dirname in os.listdir(experiment_root)
        if dirname.startswith("PPO_Soccer_")
    ]
    trial_dirs = [trial_dir for trial_dir in trial_dirs if os.path.isdir(trial_dir)]
    if not trial_dirs:
        return None

    latest_checkpoint = None
    latest_checkpoint_step = -1
    for trial_dir in sorted(trial_dirs):
        checkpoint_dirs = [
            os.path.join(trial_dir, dirname)
            for dirname in os.listdir(trial_dir)
            if dirname.startswith("checkpoint_")
        ]
        for checkpoint_dir in checkpoint_dirs:
            checkpoint_name = os.path.basename(checkpoint_dir).replace("checkpoint_", "")
            try:
                checkpoint_step = int(checkpoint_name)
            except ValueError:
                continue
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint-{checkpoint_step}"
            )
            if os.path.isfile(checkpoint_path) and checkpoint_step > latest_checkpoint_step:
                latest_checkpoint = checkpoint_path
                latest_checkpoint_step = checkpoint_step

    return latest_checkpoint


CHECKPOINT_PATH = _find_latest_checkpoint(
    "PPO_curriculum_selfplay_previous"
) or _find_latest_checkpoint("PPO_curriculum_selfplay_w_o_reward_shaping")

# CHECKPOINT_PATH = _find_latest_checkpoint(
#     "PPO_curriculum_selfplay_w_o_reward_shaping"
# ) or _find_latest_checkpoint("PPO_curriculum_selfplay_reward_shaping")


class RayAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env):
        """Initialize the RayAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        self.name = "TEAM_52_RAY_AGENT"
        ray.init(ignore_reinit_error=True)

        if CHECKPOINT_PATH is None:
            raise ValueError(
                "Could not find a packaged checkpoint in team_52_agent/ray_results."
            )

        # Load configuration from checkpoint file.
        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            # Try parent directory.
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")

        # Load the config from pickled.
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            # If no config in given checkpoint -> Error.
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory!"
            )

        # no need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0

        # create a dummy env since it's required but we only care about the policy
        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"

        # create the Trainer from config
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        # load state from checkpoint
        agent.restore(CHECKPOINT_PATH)
        # get policy for evaluation
        self.policy = agent.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        actions = {}
        for player_id in observation:
            # compute_single_action returns a tuple of (action, action_info, ...)
            # as we only need the action, we discard the other elements
            actions[player_id], *_ = self.policy.compute_single_action(
                observation[player_id], explore=False
            )
        return actions
