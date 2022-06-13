# Random Search Strategy using Ray Tune
from typing import Dict, Any

from ray import tune

from hp_space_generator import SearchStrategy


class RandomSearchStrategy(SearchStrategy):
    def run(self, run_id: int, hp_config_dict: Dict[str, Any], *args, **kwargs):
        # To ensure that we don't preprocessing again, we need to pass the pre-processed dict in the run method maybe?
        # Currently for each run in the given num_runs, pre-processing will happen each time!
        # Maybe do this in __init__()?
        if self.should_preprocess:
            hp_config_dict = self.preprocess(hp_config_dict)
        self.runs.extend(self.generate_runs(run_id, hp_config_dict))

    def preprocess(self, hp_dict):
        # Pre-process incoming hp_dict into a dict with values which can be supported by _generate_runs
        # This currently assumes you are using a Random Strategy, feel free to override it with something else

        preprocessed_hp_dict = {}
        for key, val in hp_dict.items():
            if key == "backbone":
                if isinstance(val, str):
                    preprocessed_hp_dict[key] = val
                elif isinstance(val, list):
                    preprocessed_hp_dict[key] = tune.choice(val) 
                else:
                    raise TypeError(f"Expected either List or a string for backbone but got {type(val)}")
            elif key == "learning_rate":
                if not isinstance(val, list) or len(val) != 2:
                    raise ValueError(f"Expected a list of two numbers (float/int) but got {val}")
                preprocessed_hp_dict[key] = tune.uniform(val[0], val[1])
            else:
                preprocessed_hp_dict[key] = val
        return preprocessed_hp_dict

    def generate_runs(self, run_id: int, hp_dict: Dict[str, Any]):
        runs = []
        model_config = {}
        for key, domain in hp_dict.items():
            if hasattr(domain, "sample"):
                model_config[key] = domain.sample()
            else:
                model_config[key] = domain
        runs.append(
            {"id": run_id, "model_config": model_config}
        )
        return runs
