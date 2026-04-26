from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos
import numpy as np


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class RewardShapingWrapper(gym.core.Wrapper):
    def __init__(self, env, config=None):
        super().__init__(env)
        self.config = config or {}
        self._last_ball_x = None
        self._last_player_ball_dist = {}

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_ball_x = None
        self._last_player_ball_dist = {}
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        shaped_reward = self._apply_shaping(reward, info)
        info = self._attach_reward_metadata(reward, shaped_reward, info)
        return obs, shaped_reward, done, info

    def _attach_reward_metadata(self, base_reward, shaped_reward, info):
        if isinstance(base_reward, dict):
            enriched_info = dict(info)
            for agent_id, reward_value in base_reward.items():
                agent_info = dict(enriched_info.get(agent_id, {}))
                shaped_value = float(shaped_reward[agent_id])
                base_value = float(reward_value)
                agent_info["base_reward"] = base_value
                agent_info["shaped_reward"] = shaped_value
                agent_info["shaping_bonus"] = shaped_value - base_value
                enriched_info[agent_id] = agent_info
            return enriched_info

        enriched_info = dict(info)
        base_value = float(base_reward)
        shaped_value = float(shaped_reward)
        enriched_info["base_reward"] = base_value
        enriched_info["shaped_reward"] = shaped_value
        enriched_info["shaping_bonus"] = shaped_value - base_value
        return enriched_info

    def _apply_shaping(self, reward, info):
        if isinstance(reward, dict):
            return self._shape_multiagent_reward(reward, info)
        return self._shape_single_agent_reward(reward, info)

    def _shape_single_agent_reward(self, reward, info):
        shaped_reward = float(reward)
        shaped_reward += self.config.get("step_penalty", 0.0)

        player_info = info.get("player_info", {})
        ball_info = info.get("ball_info", {})
        shaped_reward += self._ball_progress_bonus(ball_info, team_sign=1.0)
        shaped_reward += self._ball_velocity_bonus(ball_info, team_sign=1.0)
        shaped_reward += self._player_ball_bonus(
            player_key=0,
            player_info=player_info,
            ball_info=ball_info,
        )
        shaped_reward += self._movement_bonus(player_info)
        return shaped_reward

    def _shape_multiagent_reward(self, reward, info):
        shaped_reward = {
            agent_id: float(agent_reward) for agent_id, agent_reward in reward.items()
        }

        for agent_id, agent_info in info.items():
            team_sign = 1.0 if agent_id in (0, 1) else -1.0
            shaped_reward[agent_id] += self.config.get("step_penalty", 0.0)
            player_info = agent_info.get("player_info", {})
            ball_info = agent_info.get("ball_info", {})
            shaped_reward[agent_id] += self._ball_progress_bonus(
                ball_info, team_sign=team_sign
            )
            shaped_reward[agent_id] += self._ball_velocity_bonus(
                ball_info, team_sign=team_sign
            )
            shaped_reward[agent_id] += self._player_ball_bonus(
                player_key=agent_id,
                player_info=player_info,
                ball_info=ball_info,
            )
            shaped_reward[agent_id] += self._movement_bonus(player_info)

        return shaped_reward

    def _ball_progress_bonus(self, ball_info, team_sign):
        weight = self.config.get("ball_progress_weight", 0.0)
        if not weight or "position" not in ball_info:
            return 0.0

        ball_x = float(ball_info["position"][0])
        if self._last_ball_x is None:
            self._last_ball_x = ball_x
            return 0.0

        delta_x = ball_x - self._last_ball_x
        self._last_ball_x = ball_x
        return weight * team_sign * delta_x

    def _ball_velocity_bonus(self, ball_info, team_sign):
        weight = self.config.get("ball_velocity_toward_goal_weight", 0.0)
        if not weight or "velocity" not in ball_info:
            return 0.0
        return weight * team_sign * float(ball_info["velocity"][0])

    def _player_ball_bonus(self, player_key, player_info, ball_info):
        weight = self.config.get("player_to_ball_weight", 0.0)
        if not weight or "position" not in player_info or "position" not in ball_info:
            return 0.0

        player_pos = np.asarray(player_info["position"], dtype=np.float32)
        ball_pos = np.asarray(ball_info["position"], dtype=np.float32)
        distance = float(np.linalg.norm(player_pos - ball_pos))
        last_distance = self._last_player_ball_dist.get(player_key)
        self._last_player_ball_dist[player_key] = distance
        if last_distance is None:
            return 0.0
        return weight * (last_distance - distance)

    def _movement_bonus(self, player_info):
        weight = self.config.get("movement_weight", 0.0)
        if not weight or "velocity" not in player_info:
            return 0.0
        velocity = np.asarray(player_info["velocity"], dtype=np.float32)
        return weight * float(np.linalg.norm(velocity))


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env_kwargs = dict(env_config)
    reward_shaping = env_kwargs.pop("reward_shaping", None)
    env = soccer_twos.make(**env_kwargs)
    if reward_shaping:
        env = RewardShapingWrapper(env, reward_shaping)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
