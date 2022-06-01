from typing import List, Any

from lightning import LightningFlow, LightningApp, LightningWork

from flash_hpo import FlashHPO, RandomSearchStrategy


class DoSomethingExtra(LightningWork):
    def run(self, hpo_list: List[Any]):
        pass


class HPOComponent(LightningFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hpo = FlashHPO()

    def run(self):
        hpo_config = {
            "backbone": "prajjwal1/bert-tiny",
            "learning_rate": [0.000001, 0.1],
        }
        # work_cls allows you to use the hyper-paramaters from the given num_runs
        # basically .run() method is called with the HPOs from the HPO component after this
        self.hpo.run(hpo_dict=hpo_config, num_runs=2, work_cls=DoSomethingExtra, strategy_cls=RandomSearchStrategy)

        if self.hpo.results is not None:
            with open("strategy.txt", "w+") as _file:
                _file.write(str(self.hpo.results[0]))
        # if self.hpo.generated_runs is not None:
        #     with open("output.txt", "a") as _file:
        #         _file.write(str(self.hpo.generated_runs[0]))

# To launch the HPO Component
app = LightningApp(HPOComponent(), debug=True)
