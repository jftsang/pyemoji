import json
from pathlib import Path

import pytest

from model import Model


def test_can_load_rules():
    j = (Path(__file__).parent.parent.parent / "models" / "ising.json").read_text()
    d = json.loads(j)
    r = Model.model_validate(d)
    print(r)
