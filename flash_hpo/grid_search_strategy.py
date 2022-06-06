from typing import Dict, Any

from regex import W

from flash_hpo import SearchStrategy
from hyperopt import hp
from ray import tune

from ray.tune.suggest.hyperopt import HyperOptSearch


class GridSearchStrategy(SearchStrategy):
    def run(self, run_id: int, hpo_config_dict: Dict[str, Any], metric="mean_accuracy", *args, **kwargs):
        try:
            estimator = kwargs['estimator']
        except KeyError:
            raise KeyError("estimator is required for GridSearchStrategy")
        # estimator will be the function you need to run for each job
        preprocessed_dict = self.preprocess(hpo_config_dict)
        hyperopt = HyperOptSearch(preprocessed_dict, metric=metric, mode="max")
        analysis = tune.run(estimator, search_alg=hyperopt, num_samples=1, stop={"training_iteration": 1})
        self.runs.extend(self.generate_runs(run_id, preprocessed_dict, analysis, metric))

    def preprocess(self, hpo_dict):
        preprocessed_hpo_dict = {}
        for key, val in hpo_dict.items():
            if key == "backbone":
                preprocessed_hpo_dict[key] = hp.choice(key, val)
            elif key == "learning_rate":
                if not isinstance(val, list) or len(val) != 2:
                    raise ValueError(f"Expected a list of two numbers (float/int) but got {val}")
                preprocessed_hpo_dict[key] = hp.uniform(key, val[0], val[1])
        return preprocessed_hpo_dict

    def generate_runs(self, run_id: int, hpo_config_dict: dict, analysis_obj, metric: str):
        runs = []
        model_config = analysis_obj.get_best_config(metric=metric, mode="max")
        runs.append(
            {"id": run_id, "model_config": model_config}
        )
        return runs