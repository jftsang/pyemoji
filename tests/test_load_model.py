import json
from pathlib import Path

from pyemoji.model import Model
import pyemoji.experiments.ising


def test_can_load_json():
    j = (Path(pyemoji.experiments.ising.__file__).parent / "ising.json").read_text(
        encoding="utf-8"
    )
    d = json.loads(j)
    r = Model.model_validate(d)
    print(r)
