from pyemoji.actions import IfNeighborAction, GoToStateAction, MoveToAction
from tqdm.auto import tqdm

from pyemoji.model import Model, State, WorldRules
from pyemoji.simulator import Simulator
from pyemoji.visualization.pygame import PygameVisualizer

empty = State(id=0, name="down", icon="", actions=[])
abandoned = State(id=1, name="empty building", icon="🏚️", actions=[])
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
        GoToStateAction(destState=occupied),
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
    actions=[MoveToAction(dest="neighbors", destState=empty, leaveState=empty)],
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
        self.tmax = 2000
        self.pbar = tqdm(total=self.tmax)

    def post_step(self):
        t = self.time
        p = self.populations()
        self.pop_history.append({"t": t, **p})
        self.pbar.update(1)

    def should_stop(self) -> bool:
        return self.time > self.tmax

    def produce_plots(self): ...

    def finalize(self):
        self.produce_plots()
        super().finalize()


simulator = HousingSim(rules)

if __name__ == "__main__":
    states = simulator.run()
    vi = PygameVisualizer.render(states, cell_size=20)
    vi.run()
