import lightning as L

from hp_space_generator import GridSearchStrategy, HPSpaceGenerator, RandomSearchStrategy


class DoSomethingExtra(L.LightningWork):
    def __init__(self):
        super().__init__(run_once=True)
        self.hp_list = []

    def run(self, hp_list):
        self.hp_list.extend(hp_list)


def test_hp_space_generator_random():
    # Test the HP  Component for text classification
    hp_config = {
        "backbone": ["prajjwal1/bert-tiny", "pra1/bert-medium"],
        "learning_rate": [0.00001, 0.01],
    }

    space_generator = HPSpaceGenerator()
    work_obj = DoSomethingExtra()
    space_generator.run(hp_config, num_runs=2, work=work_obj, strategy=RandomSearchStrategy())
    assert len(work_obj.hp_list) != 0, "Didn't generate results..."


def test_hp_space_generator_grid():
    # Test the HP Space Generator Component for text classification
    hp_config = {
        "backbone": ["prajjwal1/bert-tiny", "pra1/bert-medium"],
        "learning_rate": [0.00001, 0.01],
    }

    space_generator = HPSpaceGenerator()
    work_obj = DoSomethingExtra()
    space_generator.run(hp_config, num_runs=2, work=work_obj, strategy=GridSearchStrategy(should_preprocess=False))
    assert len(work_obj.hp_list) != 0, "Didn't generate results..."
