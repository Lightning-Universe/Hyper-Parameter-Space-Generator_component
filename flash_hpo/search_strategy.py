from typing import Dict, Any

from lightning import LightningWork


# TODO: Maybe create a method which makes sure that the given format for the HPO Config Dictionary is correct
class SearchStrategy(LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.hpo_config_dict = None
        # self.num_runs = None
        self.runs = []

    def run(self, num_runs: int, hpo_config_dict: Dict[str, Any], *args, **kwargs):
        # self.num_runs = num_runs
        # self.hpo_config_dict = None
        # preprocessed_dict = SearchStrategy.preprocess(hpo_config_dict, *args, **kwargs)
        # self.runs = SearchStrategy.generate_runs(num_runs, preprocessed_dict, *args, **kwargs)
        raise NotImplementedError("You need to implement the run method")

    @staticmethod
    def preprocess(hpo_config_dict, *args, **kwargs):
        raise NotImplementedError("You need to implement preprocess(hpo_config_dict) method")

    @staticmethod
    def generate_runs(num_runs: int, hpo_config_dict: Dict[str, Any], *args, **kwargs):
        raise NotImplementedError("You need to implement generate_runs(num_runs, hpo_config_dict) method")
