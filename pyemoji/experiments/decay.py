import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pyemoji.actions import GoToStateAction, IfNeighborAction, IfRandomAction
from pyemoji.model import Model, State, WorldRules
from pyemoji.simulator import Simulator

downstate = State(id=0, name="down", icon="⚫️", actions=[])
upstate = State(id=1, icon="🔴", name="up", actions=[])

# decay = GoToStateAction(stateID=0)
# mightdecay = IfRandomAction(probability=0.01, actions=[decay])

# Progressively more likely to decay if you have neighbours who have
# already decayed
# act = upstate
# for num in range(1, 9):
#    prev = act
#    act = IfNeighborAction(sign=">=", num=num, stateID=0, actions=[mightdecay])
#    prev.actions.append(act)

# Additional probability of decay if you have a neighbour who has
# already decayed.

base_rate = 0.01
additional_rate = 0.01
neighbors_needed_for_assistance = 4

decay = GoToStateAction(stateID=0)
base_decay = IfRandomAction(probability=base_rate, actions=[decay])
assisted_decay = IfNeighborAction(
    sign=">=",
    num=neighbors_needed_for_assistance,
    stateID=0,
    actions=[IfRandomAction(probability=additional_rate, actions=[decay])],
)
upstate.actions.append(base_decay)
upstate.actions.append(assisted_decay)

rules = Model(
    states=[downstate, upstate],
    world=WorldRules(
        neighborhood="neumann",
        proportions=[{"stateID": 0, "parts": 0}, {"stateID": 1, "parts": 100}],
        height=103,
        width=101,
        # height=29,
        # width=31,
    ),
)


class DecaySim(Simulator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pop_history = []
        self.tmax = 1000

    def post_step(self):
        t = self.time
        p = self.populations()
        self.pop_history.append({"t": t, **p})

    def should_stop(self) -> bool:
        return self.populations()["up"] <= 50 or self.time > self.tmax

    def finalize(self):
        df = pd.DataFrame.from_records(simulator.pop_history)
        t = df["t"].to_numpy()
        pop = df["up"].to_numpy()

        # Number predicted by exponential decay
        exp_decay = self.grid.size * np.exp(-base_rate * t)

        ax = plt.gca()
        ax.plot(t, pop, "r-", label="actual population")
        ax.plot(
            t,
            exp_decay,
            "k-",
            label="exponential decay (base rate)",
        )
        ax.plot(
            t,
            self.grid.size * np.exp(-(base_rate + additional_rate) * t),
            "k:",
            label="exponential decay (always assisted)",
        )

        # Mean field theory predicts...
        delta = additional_rate / base_rate
        r0 = base_rate
        # mft_p = np.exp(-base_rate * t) / (1 - delta * np.exp(-base_rate * t))
        mft_p = (np.exp(4 * r0 * t) + delta * (np.exp(4 * r0 * t) - 1)) ** -0.25
        ax.plot(t, self.grid.size * mft_p, "k-.", label="mean field theory")

        ax.set_ylim(50 * 0.95, self.grid.size * 1.05)
        ax.set_yscale("log")
        ax.legend()
        plt.show()


simulator = DecaySim(rules)

if __name__ == "__main__":
    for _ in tqdm(simulator.run(), total=simulator.tmax):
        pass
