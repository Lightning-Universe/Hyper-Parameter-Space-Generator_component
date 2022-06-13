from lightning_hp_engine import LightningHPEngine, RandomSearchStrategy, GridSearchStrategy

from lightning import LightningWork

class DoSomethingExtra(LightningWork):
    def __init__(self):
        super().__init__(run_once=True)
        self.hpe_list = []

    def run(self, hpe_list):
        self.hpe_list.extend(hpe_list)


def test_lightning_hp_engine_random():
    # Test the HP Engine Component for text classification
    hpe_config = {
        "backbone": ["prajjwal1/bert-tiny", "pra1/bert-medium"],
        "learning_rate": [0.00001, 0.01],
    }

    hpe = LightningHPEngine()
    work_obj = DoSomethingExtra()
    hpe.run(hpe_config, num_runs=2, work=work_obj, strategy=RandomSearchStrategy())
    assert len(work_obj.hpe_list) != 0, "Didn't generate results..."

def test_lightning_hp_engine_grid():
    # Test the HP Engine Component for text classification
    hpe_config = {
        "backbone": ["prajjwal1/bert-tiny", "pra1/bert-medium"],
        "learning_rate": [0.00001, 0.01],
    }

    hpe = LightningHPEngine()
    work_obj = DoSomethingExtra()
    hpe.run(hpe_config, num_runs=2, work=work_obj, strategy=GridSearchStrategy(should_preprocess=False))
    assert len(work_obj.hpe_list) != 0, "Didn't generate results..."
