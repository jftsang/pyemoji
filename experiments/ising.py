import json
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from pyemoji.model import Model
from pyemoji.simulator import Simulator
from pyemoji.video import imgen, run


class IsingSim(Simulator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pop_history = []

    def post_step(self):
        t = self.time
        p = self.populations()
        self.pop_history.append({"t": t, **p})

    def should_stop(self) -> bool:
        return self.populations()["up"] <= 10 or self.time > 2000

    def post_stop(self):
        df = pd.DataFrame.from_records(self.pop_history)

        ax = plt.gca()
        ax.plot(df["t"], df["down"] / 1024)
        ax.plot(df["t"], df["up"] / 1024)
        # ax.plot(df["t"], 128 * 128 * np.exp(-0.01 * df["t"]), "--", label="model")
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        plt.show()


def main(argv: str | None = None) -> None:
    rules = Path(__file__).parent / "ising.json"
    d = json.loads(rules.read_text())
    rules = Model.model_validate(d)

    simulator = IsingSim(rules)

    states = tqdm(simulator.run(), total=2000)
    g = imgen(states)
    run(g, fps_cap=None)


if __name__ == "__main__":
    main()
