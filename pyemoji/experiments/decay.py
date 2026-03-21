import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pyemoji.actions import GoToStateAction, IfNeighborAction, IfRandomAction
from pyemoji.model import Model, State, WorldRules
from pyemoji.simulator import Simulator

downstate = State(id=0, name="down", icon="⚫️", actions=[])
upstate = State(id=1, icon="🔴", name="up", actions=[])

decay = GoToStateAction(stateID=0)
mightdecay = IfRandomAction(probability=0.01, actions=[decay])

# Progressively more likely to decay if you have neighbours who have
# already decayed
act = upstate
for num in range(1, 9):
    prev = act
    act = IfNeighborAction(sign=">=", num=num, stateID=0, actions=[mightdecay])
    prev.actions.append(act)


upstate.actions.append(mightdecay)

rules = Model(
    states=[downstate, upstate],
    world=WorldRules(
        neighborhood="moore",
        proportions=[{"stateID": 0, "parts": 0}, {"stateID": 1, "parts": 100}],
        size={"width": 31, "height": 29},
    ),
)


class DecaySim(Simulator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pop_history = []

    def post_step(self):
        t = self.time
        p = self.populations()
        self.pop_history.append({"t": t, **p})

    def should_stop(self) -> bool:
        return self.populations()["up"] <= 100 or self.time > 2

    def post_stop(self):
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


simulator = DecaySim(rules)

if __name__ == "__main__":
    print(d := simulator.dump())

    for _ in tqdm(simulator.run(), total=2):
        pass

    print(simulator.dump())

    simulator.load(d)
    print(simulator.dump())
