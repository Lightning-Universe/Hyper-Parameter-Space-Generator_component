# Random Search Strategy using Ray Tune
from typing import Dict, Any

from ray import tune

from flash_hpo import SearchStrategy


class RandomSearchStrategy(SearchStrategy):
    def run(self, num_runs: int, hpo_config_dict: Dict[str, Any], *args, **kwargs):
        preprocessed_dict = RandomSearchStrategy.preprocess(hpo_config_dict)
        RandomSearchStrategy.generate_runs(num_runs, preprocessed_dict)

    @staticmethod
    def preprocess(hpo_dict):
        # Pre-process incoming hpo_dict into a dict with values which can be supported by _generate_runs
        # This currently assumes you are using a Random Strategy, feel free to override it with something else
        
        preprocessed_hpo_dict = {}
        for key, val in hpo_dict.items():
            if key == "backbone":
                if isinstance(val, str):
                    preprocessed_hpo_dict[key] = val
                elif isinstance(val, list):
                    preprocessed_hpo_dict[key] = tune.choice(val) 
                else:
                    raise TypeError(f"Expected either List or a string for backbone but got {type(val)}")
            elif key == "learning_rate":
                if not isinstance(val, list) or len(val) != 2:
                    raise ValueError(f"Expected a list of two numbers (float/int) but got {val}")
                preprocessed_hpo_dict[key] = tune.uniform(val[0], val[1])
        return preprocessed_hpo_dict

    @staticmethod
    def generate_runs(num_runs: int, hpo_dict: Dict[str, Any]):
        runs = []
        for run_id in range(num_runs):
            model_config = {}
            for key, domain in hpo_dict.items():
                if hasattr(domain, "sample"):
                    model_config[key] = domain.sample()
                else:
                    model_config[key] = domain
            runs.append(
                {"id": run_id, "model_config": model_config}
            )
        return runs
