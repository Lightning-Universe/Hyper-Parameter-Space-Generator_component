<div align="center">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lai.png" width="200px">

[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai)
![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
[![CI testing](https://github.com/Lightning-Universe/Hyper-Parameter-Space-Generator_component/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-Universe/Hyper-Parameter-Space-Generator_component/actions/workflows/ci-testing.yml)

A lightning component to generate Hyper Parameter Space on a given config for Random Search and Grid Search strategies.

______________________________________________________________________

</div>

## Install

Use these instructions to install:

```bash
git clone https://github.com/Lightning-AI/LAI-Hyper-Parameter-Space-Generator-Component.git
cd LAI-Hyper-Parameter-Space-Generator-Component
pip install -e .
```

## About the Hyper Parameter Space Generator Component

This component currently supports two strategies:

1. Random Search Strategy (`RandomSearchStrategy`) - using Ray library
1. Grid Search Strategy (`GridSearchStrategy`) - using Scikit-Learn library

The component has been designed to allow users command over how they want to preprocess data. You can choose not to preprocess as well.

### Random Search Strategy

The random search strategy uses Ray's tune library to adjust the input parameter config to generate random parameters.

### Grid Search Strategy

The grid search strategy uses [Scikit-Learn's `ParameterGrid`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html) class which generates the grid of parameters. This search strategy has no default preprocessing, but if you need one, feel free to override the method.

### Input Format

Expected config for any strategy:

```python
{
    "backbone": <List of strings/string>,
    "learning_rate": <values>,
    "... any other config ...": <values>,
}
```

## Usage

To get started with using this component, copy the code below and paste it into an `app.py` file. Run the app using: `lightning run app app.py` locally and use `lightning run app app.py --cloud` on the cloud.

```python
from typing import List, Any
import logging

import lightning as L
from lightning.app.frontend import StreamlitFrontend
from lightning.app.utilities.state import AppState

from hp_space_generator import HPSpaceGenerator
from hp_space_generator import RandomSearchStrategy, GridSearchStrategy


class DoSomethingExtra(L.LightningWork):
    def __init__(self):
        super().__init__(cache_calls=True)

    def run(self, hp_list: List[Any]):
        logging.log(f"Entered work with the list: {hp_list}")


# Visualizer class to show the output table on the page
class Visualizer(L.LightningFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None

    def run(self, data):
        self.data = data

    def configure_layout(self):
        return StreamlitFrontend(render_fn=_render_fn)


def _render_fn(state: AppState):
    import streamlit as st
    import pandas as pd

    st.title("Hyper Parameters from the given runs")

    if state.data is None:
        st.write("We are working on receiving the data... hold on!")
        return

    st.table(pd.DataFrame(state.data))


class HPComponent(L.LightningFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.space_generator = HPSpaceGenerator()
        self.work_random_search = DoSomethingExtra()
        self.work_grid_search = DoSomethingExtra()
        self.visualize = Visualizer()
        self.results = None

    def run(self):
        # The input config to create the hyper parameter space around it
        hp_config = {
            "backbone": ["prajjwal1/bert-tiny", "pra1/bert-medium"],
            "learning_rate": [0.000001, 0.1],
        }

        # The hyper-parameter space generated is passed to the work class
        self.space_generator.run(
            hp_dict=hp_config,
            num_runs=5,
            work=self.work_random_search,
            strategy=RandomSearchStrategy(should_preprocess=True),
        )

        # Uncomment the lines below if you want to use GridSearchStrategy
        # self.space_generator.run(hp_dict=hp_config, num_runs=5,
        #              strategy=GridSearchStrategy(should_preprocess=False), work=self.work_grid_search)

        # Use self.work.work_grid_search if GridSearchStrategy is used
        if self.work_random_search.has_succeeded:
            self.visualize.run(self.space_generator.results)

    def configure_layout(self):
        return {"name": "generated Hyper Parameter Space", "content": self.visualize}


# To launch the hpe Component
app = L.LightningApp(HPComponent(), debug=True)
```
