from flash_hpo import FlashHPO, RandomSearchStrategy, GridSearchStrategy

from lightning import LightningWork

class DoSomethingExtra(LightningWork):
    def __init__(self):
        super().__init__(run_once=True)
        self.hpo_list = []

    def run(self, hpo_list):
        self.hpo_list.extend(hpo_list)


def test_flash_hpo_random():
    # Test the Flash HPO Component for text classification
    hpo_config = {
        "backbone": ["prajjwal1/bert-tiny", "pra1/bert-medium"],
        "learning_rate": [0.00001, 0.01],
    }

    hpo = FlashHPO()
    work_obj = DoSomethingExtra()
    hpo.run(hpo_config, num_runs=2, work=work_obj, strategy=RandomSearchStrategy()) 
    assert len(work_obj.hpo_list) != 0, "Didn't generate results..."

def test_flash_hpo_grid():
    # Test the Flash HPO Component for text classification
    hpo_config = {
        "backbone": ["prajjwal1/bert-tiny", "pra1/bert-medium"],
        "learning_rate": [0.00001, 0.01],
    }

    hpo = FlashHPO()
    work_obj = DoSomethingExtra()
    hpo.run(hpo_config, num_runs=2, work=work_obj, strategy=GridSearchStrategy(should_preprocess=False)) 
    assert len(work_obj.hpo_list) != 0, "Didn't generate results..."
