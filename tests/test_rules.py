import json
from pathlib import Path

import pytest

from pyemoji.model import Model


def test_can_load_rules():
    j = (Path(__file__).parent.parent / "experiments" / "ising.json").read_text(encoding="utf-8")
    d = json.loads(j)
    r = Model.model_validate(d)
    print(r)
