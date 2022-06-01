<div align="center">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lai.png" width="200px">

A lightning component to run HPO on given config.

______________________________________________________________________

</div>

## Install

**NOTE**: Please do not review yet unless and until you are one of the main authors of Auto ML App. This component is in _Heavy WIP_ stage. :)

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

```
from lightning import LightningFlow
from lightning import LightningApp

from flash_hpo import FlashHPO


class HPOComponent(LightningFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hpo = FlashHPO()

    def run(self):
        hpo_config = {
            "backbone": "prajjwal1/bert-tiny",
            "learning_rate": [0.000001, 0.1],
        }
        self.hpo.run(hpo_dict=hpo_config, num_runs=2)

        if self.hpo.generated_runs is not None:
            with open("output.txt", "a") as _file:
                _file.write(str(self.hpo.generated_runs[0]))


# To launch the HPO Component
app = LightningApp(HPOComponent(), debug=True)
```
