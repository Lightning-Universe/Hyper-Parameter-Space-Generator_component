from typing import Dict, Any

from lightning import LightningWork


# TODO: Maybe create a method which makes sure that the given format for the hpe Config Dictionary is correct
class SearchStrategy(LightningWork):
    def __init__(self, should_preprocess=True, *args, **kwargs):
        super().__init__(run_once=True, parallel=True, *args, **kwargs)
        self.should_preprocess = should_preprocess
        self.runs = []

    def run(self, hpe_config_dict: Dict[str, Any], num_runs=1, *args, **kwargs):
        raise NotImplementedError("You need to implement the run method")

    def preprocess(hpe_config_dict, *args, **kwargs):
        raise NotImplementedError("You need to implement preprocess(hpe_config_dict) method")

    def generate_runs(num_runs: int, hpe_config_dict: Dict[str, Any], *args, **kwargs):
        raise NotImplementedError("You need to implement generate_runs(num_runs, hpe_config_dict) method")
