<div align="center">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lai.png" width="200px">

A lightning component to generate Hyper Parameter Space on a given config for Random Search and Grid Search strategies.

______________________________________________________________________

</div>

## Install

Use these instructions to install:

```bash
git clone https://github.com/PyTorchLightning/LAI-Hyper-Parameter-Space-Generator.git
cd LAI-Hyper-Parameter-Space-Generator
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

import lightning as L
from lightning.app.frontend import StreamlitFrontend
from lightning.app.utilities.state import AppState

from hp_space_generator import HPSpaceGenerator
from hp_space_generator import RandomSearchStrategy, GridSearchStrategy


class DoSomethingExtra(L.LightningWork):
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


class Visualizer(L.LightningFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None

    def run(self, data):
        self.data = data

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_fn)


class HPComponent(L.LightningFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.space_generator = HPSpaceGenerator()
        self.work_random_search = DoSomethingExtra()
        self.work_grid_search = DoSomethingExtra()
        self.visualize = Visualizer()
        self.results = None

    def run(self):
        # If you want, you can write your own preprocess functions, and pass should_preprocess=False
        # in the class instantiation.

        hp_config = {
            "backbone": ["prajjwal1/bert-tiny", "pra1/bert-medium"],
            "learning_rate": [0.000001, 0.1],
        }

        # work_cls allows you to use the hyper-paramaters from the given num_runs

        self.space_generator.run(hp_dict=hp_config, num_runs=5,
                     work=self.work_random_search, strategy=RandomSearchStrategy(should_preprocess=True))

        # self.space_generator.run(hp_dict=hp_config, num_runs=5,
        #              strategy=GridSearchStrategy(should_preprocess=False), work=self.work_grid_search)

        if self.work_random_search.has_succeeded:
            self.visualize.run(self.space_generator.results)

    def configure_layout(self):
        return {"name": "generated Hyper Parameter Space", "content": self.visualize}


# To launch the hpe Component
app = L.LightningApp(HPComponent(), debug=True)
```
