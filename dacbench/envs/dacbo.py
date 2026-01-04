"""DACBO Env."""

from __future__ import annotations

import numpy as np
from dacboenv.dacboenv import DACBOEnv as DEnv

from dacbench.abstract_env import AbstractEnv


class DACBOEnv(AbstractEnv):
    """DACBO env."""

    def __init__(self, config):
        """Init DACBO env."""
        self._env = DEnv(task_ids=config["instance_set"][0], **config)
        self.reset()
        config[
            "cutoff"
        ] = np.inf  # Not used. DACBO handles BO runs (i.e. episodes) internally
        config["observation_space"] = self._env.observation_space
        config["action_space"] = self._env.action_space
        super().__init__(config)

    def step(self, action):
        """Takes one env step."""
        state, reward, terminated, truncated, info = self._env.step(action)
        if truncated:  # Reset BO loop, select new instance
            self._env.reset()
        return state, reward, terminated, truncated, info

    def reset(self, seed=None):
        """Resets the internal env."""
        return self._env.reset()
