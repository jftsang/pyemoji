import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from actions import GoToStateAction, IfNeighborAction, IfRandomAction, MoveToAction
from model import Model, State, WorldRules
from simulator import Simulator

downstate = State(id=0, name="down", icon="⚫️", actions=[])
upstate = State(id=1, icon="🔴", name="up", actions=[])

walk = MoveToAction(dest="neighbors", spotStateID=0, leaveStateID=0)
upstate.actions.append(walk)

rules = Model(
    states=[downstate, upstate],
    world=WorldRules(
        neighborhood="moore",
        proportions=[{"stateID": 0, "parts": 13 * 17 - 3}, {"stateID": 1, "parts": 3}],
        size={"width": 13, "height": 17},
    ),
)


class RandomWalkSimulator(Simulator):
    def setup_ics(self):
        super().setup_ics()
        for i in range(self.height):
            for j in range(self.width):
                agent = Agent(i, j, simulator=self)

                state: State = random.choices(self.model.states, weights=probs)[0]  # type: ignore

                agent.force_state(state)
                self.grid[i, j] = agent


simulator = Simulator(rules)

if __name__ == "__main__":
    pops = []

    tmax = 100
    for s in tqdm(simulator.run(), total=tmax):
        t = s.time
        p = s.populations()
        pops.append({"t": t, **p})

        if t > tmax:
            break

    df = pd.DataFrame.from_records(pops)

    print(df)
    ax = plt.gca()
    ax.plot(df["t"], df["up"])
    # ax.plot(df["t"], 31 * 29 * np.exp(-0.001 * df["t"]), "--", label="model")
    # ax.set_yscale("log")
    # ax.legend()
    plt.show()
