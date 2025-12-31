"""DACBOEnv Benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs.dacbo import DACBOEnv

INFO = {
    "identifier": "DACBO",
    "name": "DACBO",
    "reward": "DACBO Rewards",
    "state_description": [
        "incumbent_change_observation",
        "trials_passed_observation",
        "trials_left_observation",
        "ubr_observation",
        "modelfit_observation",
        "dimensions_observation",
        "continuous_hp_observation",
        "categorical_hp_observation",
        "ordinal_hp_observation",
        "int_hp_observation",
        "tsp_observation",
        "knn_entropy_observation",
        "skewness_observation",
        "kurtosis_observation",
        "mean_observation",
        "std_observation",
        "variability_observation",
    ],
}

DACBO_DEFAULTS = objdict(
    {
        "reward_range": (float("-inf"), float("inf")),
        "seed": 0,
        "instance_set_path": "bbob_2_default.csv",
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
