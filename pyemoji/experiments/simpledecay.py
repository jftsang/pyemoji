import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pyemoji.actions import GoToStateAction, IfRandomAction
from pyemoji.model import Model, State, WorldRules
from pyemoji.simulator import Simulator

downstate = State(id=0, name="down", icon="⚫️", actions=[])
upstate = State(id=1, name="up", icon="🔴", actions=[])

decay = GoToStateAction(destState=downstate)
mightdecay = IfRandomAction(probability=0.01, actions=[decay])

upstate.actions.append(mightdecay)

rules = Model(
    states=[downstate, upstate],
    world=WorldRules(
        neighborhood="moore",
        proportions={
            downstate.id: 0,
            upstate.id: 100,
        },
        height=29,
        width=31,
    ),
)


class SimpleDecaySim(Simulator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pop_history = []
        self.tmax = 2000

    def post_step(self):
        t = self.time
        p = self.populations()
        self.pop_history.append({"t": t, **p})

    def should_stop(self) -> bool:
        return self.populations()["up"] == 0 or self.time > self.tmax

    def produce_plots(self):
        df = pd.DataFrame.from_records(simulator.pop_history)

        ax = plt.gca()
        ax.plot(df["t"], df["up"], label="actual population")
        ax.plot(
            df["t"],
            self.grid.size * np.exp(-mightdecay.probability * df["t"]),
            "--",
            label="exponential decay",
        )
        ax.set_yscale("log")
        ax.legend()
        plt.show()

    def finalize(self):
        self.produce_plots()
        super().finalize()


simulator = SimpleDecaySim(rules)

if __name__ == "__main__":
    for _ in tqdm(simulator.run(), total=simulator.tmax):
        pass
