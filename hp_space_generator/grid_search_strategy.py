from typing import Any, Dict

from sklearn.model_selection import ParameterGrid

from hp_space_generator.search_strategy import SearchStrategy


class GridSearchStrategy(SearchStrategy):
    def __init__(self, should_preprocess=False, *args, **kwargs):
        super().__init__(should_preprocess=should_preprocess, *args, **kwargs)

    def run(self, hp_config_dict: Dict[str, Any], run_id=-1):
        if self.should_preprocess:
            hp_config_dict = self.preprocess(hp_config_dict)

        self.runs.extend(self.generate_runs(run_id, hp_config_dict))

    def preprocess(self, hpe_dict):
        """We don't need to perform any preprocessing here, assuming correct configuration is passed.

        For what is supported, consult: https://scikit-
        learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html
        """
        return hpe_dict

    def generate_runs(self, run_id: int, model_config: dict):
        runs = []
        config_dict = {"id": run_id}
        param_grid = list(ParameterGrid(model_config))
        for ind, val in enumerate(param_grid):
            config_dict[f"Space Index: {ind}"] = val
        runs.append(config_dict)
        return runs
