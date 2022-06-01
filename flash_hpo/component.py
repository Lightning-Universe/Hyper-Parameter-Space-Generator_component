from typing import Dict, Any, List, Optional
from ray import tune

from lightning import LightningWork


class FlashHPO(LightningWork):
    """
    The HPO Component is used to suggest a list of configurations (hyper-parameters) to run with some config
    from the user for any task.

    This component doesn't come with a default UI. Please consider adding a UI yourself based on your task and needs.
    """

    def __init__(self, run_once=True, *args, **kwargs):
        super().__init__(*args, run_once=run_once, **kwargs)

        self.generated_runs: Optional[List[Dict[str, Any]]] = None
        self.hpo_dict = {}
        self.num_runs = 0

    def run(self, hpo_dict: dict, num_runs: int=1, *args, **kwargs):
        if self.generated_runs is not None:
            # clean-up
            self.generated_runs = None

        self.hpo_dict = hpo_dict
        self.num_runs = num_runs

        hpo_dict = FlashHPO.preprocess_hpo_dict(hpo_dict)
        self.generated_runs = FlashHPO._generate_runs(num_runs, hpo_dict, *args, **kwargs)

    @staticmethod
    def preprocess_hpo_dict(hpo_dict):
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
    def _generate_runs(num_runs: int, hpo_dict: Dict[str, Any], *args, **kwargs):
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
