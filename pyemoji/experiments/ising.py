import json
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from pyemoji.file_writers import PopulationFileWriter
from pyemoji.model import Model
from pyemoji.simulator import Simulator
from pyemoji.visualization.images import ImageMaker


class IsingSim(Simulator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pop_history = []

    def post_step(self):
        t = self.time
        p = self.populations()
        self.pop_history.append({"t": t, **p})

    def should_stop(self) -> bool:
        return self.time > 200

    def post_stop(self):
        df = pd.DataFrame.from_records(self.pop_history)

        fig, axs = plt.subplots(2, 1)
        ax = axs[0]
        ax.plot(df["t"], df["down"] / self.grid.size)
        ax.plot(df["t"], df["up"] / self.grid.size)
        # ax.plot(df["t"], 128 * 128 * np.exp(-0.01 * df["t"]), "--", label="model")
        ax.set_ylim(-0.1, 1.1)
        ax.legend()

        ax = axs[1]
        img = ImageMaker().from_simulator(self)
        ax.imshow(img)

        plt.show()


def main(argv: str | None = None) -> None:
    rules = Path(__file__).parent / "ising.json"
    d = json.loads(rules.read_text(encoding="utf-8"))
    rules = Model.model_validate(d)

    simulator = IsingSim(rules)
    writer = PopulationFileWriter(simulator, "ising.population.csv")
    simulator.writers.append(writer)

    with writer:
        states = tqdm(simulator.run(), total=500)
        for _ in states:
            pass
        # g = imgen(states)
        # run(g, fps_cap=None)


if __name__ == "__main__":
    main()
