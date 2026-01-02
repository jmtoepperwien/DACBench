"""DACBOEnv Benchmark."""

from __future__ import annotations

from pathlib import Path

import dacboenv
import numpy as np
import yaml
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
        "instance_set_path": "bbob_2_default.yaml",
        "observation_keys": [
            "ubr_difference",
            "acq_value_EI",
            "acq_value_PI",
            "previous_param",
        ],
        "action_space_class": AcqParameterActionSpace,
        "action_space_kwargs": None,
        "reward_keys": None,
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

        with open(path) as f:
            instance_data = yaml.safe_load(f)
        print(instance_data)
        self.config["instance_set"] = {
            0: instance_data["task_ids"]
        }  # Instance selection is handled by the internal env
        self.config["inner_seeds"] = instance_data.get("inner_seeds", None)

        assert len(self.config["instance_set"][0]) > 0, "ERROR: empty instance set"
