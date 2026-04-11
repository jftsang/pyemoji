import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
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

tmax = 300

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
        # height=59,
        # width=61,
        height=29,
        width=31,
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
        # return self.populations()["up"] <= 10 or self.time > tmax
        return self.time >= tmax

    def finalize(self):
        pass


def main():
    ax = plt.gca()
    simulator: DecaySim
    n_experiments = 20
    t = np.arange(0, tmax)
    results = np.empty((n_experiments, len(t)))
    for sid in range(n_experiments):
        simulator = DecaySim(rules)

        for _ in tqdm(simulator.run(), total=tmax):
            pass
        df = pd.DataFrame.from_records(simulator.pop_history)
        # t = df["t"].to_numpy()
        pop = df["up"].to_numpy()
        results[sid, :] = pop

        # cmap = plt.get_cmap("Oranges")
        # ax.plot(t, pop, "-", c=cmap(sid / n_experiments), lw=0.8)

    mu = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    ax.fill_between(t, mu - std, mu + std, alpha=0.2, color="r")
    ax.plot(t, mu, "r-")
    ax.plot(t, mu + std, "r--")
    ax.plot(t, mu - std, "r--")

    gs = rules.world.height * rules.world.width
    exp_decay = gs * np.exp(-base_rate * t)
    ax.plot(t, exp_decay, "k-", label="exponential decay (base rate)")
    ax.plot(
        t,
        gs * np.exp(-(base_rate + additional_rate) * t),
        "k:",
        label="exponential decay (always assisted)",
    )

    # Mean field theory predicts...
    delta = additional_rate / base_rate
    r0 = base_rate

    def mft(_, p):
        return -r0 * p * (1 + delta * (1 - p) ** 4)

    sol = solve_ivp(mft, [0, tmax], [1], t_eval=np.arange(0, tmax))

    ax.plot(sol.t, gs * sol.y[0], "k-.", label="mean field theory")

    ax.set_ylim(50, gs * 1.05)
    ax.set_yscale("log")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
