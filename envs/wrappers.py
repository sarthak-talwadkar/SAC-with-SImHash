"""Environment construction.

- `make_env` builds a Gymnasium env, rescales continuous actions to [-1, 1]
  (so the actor never needs to know each env's true action range), and seeds it.
- `SparseMuJoCoReward` converts a dense locomotion reward into a sparse goal
  reward (+1 only once the agent runs past a forward-distance threshold),
  giving us genuine hard-exploration tasks for the comparison.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import RescaleAction


class SparseMuJoCoReward(gym.Wrapper):
    """Sparse reward for MuJoCo locomotion: +1 per step while the agent's
    forward x-position is past `threshold`, else 0. The dense shaping (forward
    velocity, control cost) is discarded, so the agent must *discover* how to
    travel before it sees any reward at all.
    """

    def __init__(self, env, threshold: float = 5.0):
        super().__init__(env)
        self.threshold = float(threshold)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        x = info.get("x_position", None)
        reward = 1.0 if (x is not None and x >= self.threshold) else 0.0
        return obs, reward, terminated, truncated, info


def make_env(env_id: str, seed: int = 0, sparse: bool = False,
             sparse_threshold: float = 5.0, render_mode=None) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    if sparse:
        env = SparseMuJoCoReward(env, threshold=sparse_threshold)
    # Map the agent's [-1, 1] actions onto the env's real action range.
    if isinstance(env.action_space, gym.spaces.Box):
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
