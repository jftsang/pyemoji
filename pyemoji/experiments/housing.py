import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from pyemoji.actions import (
    IfNeighborAction,
    MoveToAction,
    IfRandomAction,
)
from pyemoji.model import Model, State, WorldRules
from pyemoji.simulator import Simulator
from pyemoji.visualization.pygame import PygameVisualizer

empty = State(id=0, name="empty", icon="", actions=[])
abandoned = State(id=1, name="unused building", icon="🏚️", actions=[])
occupied = State(id=2, name="occupied building", icon="🏠", actions=[])
person = State(id=3, name="person", icon="🚶", actions=[])
mobile_person = State(id=4, name="mobile person", icon="🚴", actions=[])


rules = Model(
    states=[empty, abandoned, occupied, person, mobile_person],
    world=WorldRules(
        neighborhood="moore",
        proportions={
            empty.id: 60,
            abandoned.id: 10,
            occupied.id: 0,
            person.id: 10,
            mobile_person.id: 0,
        },
        height=37,
        width=41,
    ),
)


move_in = IfNeighborAction(
    sign=">",
    num=0,
    neighborState=abandoned,
    actions=[
        MoveToAction(
            dest="neighbors",
            destState=abandoned,
            resultState=occupied,
            leaveState=empty,
        ),
    ],
)

slow_people_move = IfNeighborAction(
    sign="=",
    num=0,
    neighborState=abandoned,
    actions=[
        IfRandomAction(
            probability=0.2,
            actions=[MoveToAction(dest="neighbors", destState=empty, leaveState=empty)],
        )
    ],
)

fast_people_move = IfNeighborAction(
    sign="=",
    num=0,
    neighborState=abandoned,
    actions=[MoveToAction(dest="anywhere", destState=empty, leaveState=empty)],
)

person.actions = [move_in, slow_people_move]
mobile_person.actions = [move_in, fast_people_move]


class HousingSim(Simulator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pop_history = []
        self.tmax = 200
        self.pbar = tqdm(total=self.tmax)

    def pre_step(self):
        t = self.time
        p = self.populations()
        self.pop_history.append({"t": t, **p})

    def post_step(self):
        self.pbar.update(1)

    def post_stop(self):
        t = self.time
        p = self.populations()
        self.pop_history.append({"t": t, **p})

    def should_stop(self) -> bool:
        return self.time > self.tmax

    def produce_plots(self):
        df = pd.DataFrame.from_records(self.pop_history)
        fig, ax = plt.subplots(1, 1)
        for s in self.model.states[1:]:
            ax.plot(df["t"], df[s.name], label=s.name)

        ax.legend()
        self.fig = fig

        # breakpoint()

    def finalize(self):
        self.produce_plots()
        super().finalize()


simulator = HousingSim(rules)

if __name__ == "__main__":
    states = simulator.run()
    vi = PygameVisualizer.render(states, cell_size=20)
    vi.run()

    simulator.fig.show()

    plt.show()
