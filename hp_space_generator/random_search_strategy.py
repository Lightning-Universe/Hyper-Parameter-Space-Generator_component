# Random Search Strategy using Ray Tune
import collections
from numbers import Number
from typing import Any, Dict

from ray import tune

from hp_space_generator.search_strategy import SearchStrategy


class RandomSearchStrategy(SearchStrategy):
    def run(self, run_id: int, hp_config_dict: Dict[str, Any], *args, **kwargs):
        # To ensure that we don't preprocessing again, we need to pass the pre-processed dict in the run method maybe?
        # Currently for each run in the given num_runs, pre-processing will happen each time!
        # Maybe do this in __init__()?
        if self.should_preprocess:
            hp_config_dict = self.preprocess(hp_config_dict)
        self.runs.extend(self.generate_runs(run_id, hp_config_dict))

    def preprocess(self, hp_dict):
        # Pre-process incoming hp_dict into a dict with values which can be supported by `generate_runs`
        # Feel free to override this function based on your convenience
        preprocessed_hp_dict = {}
        for key, val in hp_dict.items():
            if isinstance(val, collections.abc.Iterable):
                if len(val) == 2 and isinstance(val[0], Number) and isinstance(val[1], Number):
                    preprocessed_hp_dict[key] = tune.uniform(val[0], val[1])
                else:
                    preprocessed_hp_dict[key] = tune.choice(val)
            else:
                preprocessed_hp_dict[key] = tune.choice(val)
        return preprocessed_hp_dict

    def generate_runs(self, run_id: int, hp_dict: Dict[str, Any]):
        runs = []
        model_config = {}
        for key, domain in hp_dict.items():
            if hasattr(domain, "sample"):
                model_config[key] = domain.sample()
            else:
                model_config[key] = domain
        runs.append({"id": run_id, "model_config": model_config})
        return runs
