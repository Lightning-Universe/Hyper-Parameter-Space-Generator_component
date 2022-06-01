from typing import List, Any

from lightning import LightningFlow, LightningApp, LightningWork
from lightning.frontend import StreamlitFrontend
from lightning.utilities.state import AppState

import streamlit as st
import pandas as pd

from flash_hpo import FlashHPO, RandomSearchStrategy


class DoSomethingExtra(LightningWork):
    def __init__(self):
        super().__init__()
        self.hpo_list = []

    def run(self, hpo_list: List[Any], *args, **kwargs):
        self.hpo_list = hpo_list

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_fn)


def render_fn(state: AppState):
    st.title("Hyper Parameters from the given runs")

    # if len(state.hpo_list) == 0:
    #     # Empty!!
    #     st.title("Configuring...")
    # else:
    #     # for hpo in state.hpo_list:
    #     # st.write(pd.DataFrame(state.hpo_list))
    st.write(type(state.hpo_list))
        

class HPOComponent(LightningFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hpo = FlashHPO()
        self.work = DoSomethingExtra()

    def run(self):
        hpo_config = {
            "backbone": "prajjwal1/bert-tiny",
            "learning_rate": [0.000001, 0.1],
        }
        # work_cls allows you to use the hyper-paramaters from the given num_runs
        # basically .run() method is called with the HPOs from the HPO component after this
        # self.hpo.run(hpo_dict=hpo_config, num_runs=2, work_cls=DoSomethingExtra, strategy_cls=RandomSearchStrategy)
        self.hpo.run(hpo_dict=hpo_config, num_runs=2, work=self.work, strategy_cls=RandomSearchStrategy)

        # if len(self.hpo.results) > 0:
        #     with open("strategy.txt", "a") as _file:
        #         _file.write(str(self.hpo.results))
        # if self.hpo.generated_runs is not None:
        #     with open("output.txt", "a") as _file:
        #         _file.write(str(self.hpo.generated_runs[0]))

    def configure_layout(self):
        return {"name": "HPO Output", "content": self.work}


# To launch the HPO Component
app = LightningApp(HPOComponent(), debug=True)
