from tqdm.auto import tqdm

from pyemoji.model import Model, State, WorldRules
from pyemoji.simulator import Simulator
from pyemoji.visualization.pygame import render

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
            occupied.id: 10,
            person.id: 10,
            mobile_person.id: 10,
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
        return self.time > self.tmax

    def produce_plots(self): ...

    def finalize(self):
        self.produce_plots()
        super().finalize()


simulator = SimpleDecaySim(rules)

if __name__ == "__main__":
    states = tqdm(simulator.run(), total=simulator.tmax)
    render(states, cell_size=24)
