import json
from pathlib import Path

import pyemoji
from pyemoji.model import Model


def test_can_load_json(subtests):
    """Test that we can load the original JSON models from ncase/sim."""
    ncase_models = (
        Path(pyemoji.__file__).parent.parent  # project top level directory
        / "ncase"
        / "sim"
        / "models"
    )

    for f in ncase_models.iterdir():
        with subtests.test(f.name):
            j = f.read_text(encoding="utf-8")
            d = json.loads(j)
            r = Model.model_validate(d)
