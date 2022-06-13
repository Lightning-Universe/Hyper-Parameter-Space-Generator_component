<div align="center">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lai.png" width="200px">

A lightning component to run a Hyper-Parameter engine on a given config for RandomSearch and GridSearch strategies.

______________________________________________________________________

</div>

## Install

Use these instructions to install:

```bash
git clone https://github.com/PyTorchLightning/lightning-hp-engine.git
cd lightning-hp-engine
pip install -r requirements.txt
pip install -e .
```

## Use the component

Expected config:

```python
{
    "backbone": <List of strings/string>,
    "learning_rate": <List of 2 numbers (int/float)>,
    "... any other config ...": <vals>,
}
```

Methods which can be overridden will be included later.

```python
from typing import List, Any

from lightning import LightningFlow, LightningApp, LightningWork
from lightning.frontend import StreamlitFrontend
from lightning.utilities.state import AppState


from lightning_hp_engine import LightningHPEngine
from lightning_hp_engine import RandomSearchStrategy, GridSearchStrategy


class DoSomethingExtra(LightningWork):
    def __init__(self):
        super().__init__(run_once=True)
        self.hpe_list = []

    def run(self, hpe_list: List[Any]):
        self.hpe_list.extend(hpe_list)
        with open("work_output.txt", "a") as _file:
            _file.write(str(len(self.hpe_list)))


def _render_fn(state: AppState):
    import streamlit as st
    import pandas as pd

    st.title("Hyper Parameters from the given runs")

    if state.data is None:
        st.write("We are working on receiving the data... hold on!")
        return

    st.table(pd.DataFrame(state.data))


class Visualizer(LightningFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None

    def run(self, data):
        self.data = data

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_fn)


class HPEComponent(LightningFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hpe = LightningHPEngine()
        self.work_random_search = DoSomethingExtra()
        self.work_grid_search = DoSomethingExtra()
        self.visualize = Visualizer()
        self.results = None

    def run(self):
        # If you want, you can write your own preprocess functions, and pass should_preprocess=False
        # in the class instantiation.

        hpe_config = {
            "backbone": ["prajjwal1/bert-tiny", "pra1/bert-medium"],
            "learning_rate": [0.000001, 0.1],
        }

        # work_cls allows you to use the hyper-paramaters from the given num_runs
        # basically .run() method is called with the hpes from the HP Engine component after this

        self.hpe.run(hpe_dict=hpe_config, num_runs=5,
                     work=self.work_random_search, strategy=RandomSearchStrategy(should_preprocess=True))

        # self.hpe.run(hpe_dict=hpe_config, num_runs=5,
        #              strategy=GridSearchStrategy(should_preprocess=False), work=self.work_grid_search)

        if self.work_random_search.has_succeeded:
            self.visualize.run(self.hpe.results)

    def configure_layout(self):
        return {"name": "hpe Output", "content": self.visualize}


# To launch the hpe Component
app = LightningApp(HPEComponent(), debug=True)
```
