"""DACBOEnv Benchmark."""

from __future__ import annotations

from pathlib import Path

import dacboenv
import numpy as np
import pandas as pd
from dacboenv.env.action import AcqParameterActionSpace

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs.dacbo import DACBOEnv

INFO = {
    "identifier": "DACBO",
    "name": "DACBO",
    "reward": "Incumbent cost",
    "state_description": [
        obs.name for obs in dacboenv.env.observation.ALL_OBSERVATIONS
    ],
}

DACBO_DEFAULTS = objdict(
    {
        "reward_range": [-np.inf, np.inf],
        "seed": 0,
        "instance_set_path": "bbob_2_default.csv",
        "observation_keys": None,
        "action_space_class": AcqParameterActionSpace,
        "action_space_kwargs": None,
        "reward_keys": None,
        "inner_seeds": None,
        "benchmark_info": INFO,
    }
)


class DACBOBenchmark(AbstractBenchmark):
    """DACBOEnv benchmark."""

    def __init__(self, config_path=None, config=None):
        """Init DACBOEnv benchmark."""
        super().__init__(config_path, config)

        if not self.config:
            self.config = objdict(DACBO_DEFAULTS.copy())

        for key in DACBO_DEFAULTS:
            if key not in self.config:
                self.config[key] = DACBO_DEFAULTS[key]

    def get_environment(self):
        """Returns the internal env."""
        return DACBOEnv(self.config)

    def read_instance_set(self):
        """Reads the instance set."""
        assert self.config.instance_set_path
        if Path(self.config.instance_set_path).is_file():
            path = self.config.instance_set_path
        else:
            path = (
                Path(__file__).resolve().parent
                / "../instance_sets/dacbo/"
                / self.config.instance_set_path
            )

        instance_df = pd.read_csv(path)
        self.config["instance_set"] = {
            0: instance_df["task_id"].tolist()
        }  # Instance selection is handled by the internal env

        assert len(self.config["instance_set"][0]) > 0, "ERROR: empty instance set"
