from typing import Dict, Any

from flash_hpo import SearchStrategy
from ray import tune


class GridSearchStrategy(SearchStrategy):
    def run(self, run_id: int, hpo_config_dict: Dict[str, Any], metric: str, mode: str, preprocess=True, **kwargs):
        try:
            estimator = kwargs["estimator"]
        except KeyError:
            raise KeyError("estimator is required for GridSearchStrategy")

        if preprocess:
            hpo_config_dict = self.preprocess(hpo_config_dict)
        # estimator will be the function you need to run for each job
        analysis = tune.run(estimator, config=hpo_config_dict)
        self.runs.extend(self.generate_runs(run_id, analysis, metric, mode))

    def preprocess(self, hpo_dict):
        preprocessed_hpo_dict = {}
        for key, val in hpo_dict.items():
            if key == "backbone":
                if isinstance(val, str):
                    preprocessed_hpo_dict[key] = val
                elif isinstance(val, list):
                    preprocessed_hpo_dict[key] = tune.grid_search(val)
                else:
                    raise ValueError("Only list/str allowed to be passed as backbone")
            elif key == "learning_rate":
                if isinstance(val, (int, float)):
                    preprocessed_hpo_dict[key] = val
                    continue
                if not isinstance(val, list):
                    raise ValueError("Expected a list of numbers (int/float)")
                preprocessed_hpo_dict[key] = tune.grid_search(val)
            else:
               raise ValueError("Either define the preprocess function for the config type, or pass preprocess=false to the run method")
        return preprocessed_hpo_dict

    def generate_runs(self, run_id: int, analysis_obj, metric: str, mode: str):
        runs = []
        model_config = analysis_obj.get_best_config(metric=metric, mode=mode)
        runs.append(
            {"id": run_id, "model_config": model_config}
        )
        return runs
