<div align="center">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lai.png" width="200px">

A lightning component to run HPO on given config.

______________________________________________________________________

</div>

## Install

Use these instructions to install:

```bash
git clone https://github.com/PyTorchLightning/LAI-HPO.git
cd LAI-HPO
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


from flash_hpo import FlashHPO, RandomSearchStrategy


class DoSomethingExtra(LightningWork):
    def __init__(self):
        super().__init__(run_once=True)
        self.hpo_list = []

    def run(self, hpo_list: List[Any]):
        self.hpo_list.extend(hpo_list)


def _render_fn(state: AppState):
    import streamlit as st
    import pandas as pd

    st.title("Hyper Parameters from the given runs")

    if state.data is None:
        st.write("We are working on receiving the data... hold on!")
        return

    st.write(pd.DataFrame(state.data))


class Visualizer(LightningFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None

    def run(self, data):
        self.data = data

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_fn)


class HPOComponent(LightningFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hpo = FlashHPO()
        self.work = DoSomethingExtra()
        self.visualize = Visualizer()
        self.results = None

    def run(self):
        hpo_config = {
            "backbone": ["prajjwal1/bert-tiny", "pra1/bert-medium"],
            "learning_rate": [0.000001, 0.1],
        }

        # work_cls allows you to use the hyper-paramaters from the given num_runs
        # basically .run() method is called with the HPOs from the HPO component after this
        # self.hpo.run(hpo_dict=hpo_config, num_runs=2, work_cls=DoSomethingExtra, strategy_cls=RandomSearchStrategy)
        self.hpo.run(hpo_dict=hpo_config, num_runs=6,
                     work=self.work, strategy=RandomSearchStrategy())

        if self.work.has_succeeded:
            self.visualize.run(self.hpo.results)

    def configure_layout(self):
        return {"name": "HPO Output", "content": self.visualize}


# To launch the HPO Component
app = LightningApp(HPOComponent(), debug=True)
```
