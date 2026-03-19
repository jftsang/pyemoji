import json
from pathlib import Path

import pytest

from rules import Rules


def test_can_load_rules():
    j = (Path(__file__).parent.parent.parent / "models" / "ising.json").read_text()
    d = json.loads(j)
    r = Rules.from_dict(d)
    print(r)
