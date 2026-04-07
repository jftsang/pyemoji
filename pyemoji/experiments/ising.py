import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from pyemoji.actions import GoToStateAction, IfNeighborAction, IfRandomAction
from pyemoji.file_writers import PopulationFileWriter
from pyemoji.model import Model, State, WorldRules
from pyemoji.simulator import Simulator
from pyemoji.visualization.images import ImageMaker

downstate = State(id=0, name="down", icon="⚫️", actions=[])
upstate = State(id=1, icon="🔴", name="up", actions=[])

flipdown = GoToStateAction(stateID=0)
flipup = GoToStateAction(stateID=1)


def maybe(p, a):
    return IfRandomAction(probability=p, actions=[a])


p_assisted_flip = 0.05
p_random_flip = 0.0001

downstate.actions = [
    IfNeighborAction(
        sign=">=", num=5, stateID=1, actions=[maybe(p_assisted_flip, flipup)]
    ),
    maybe(p_random_flip, flipup),
]

upstate.actions = [
    IfNeighborAction(
        sign=">=", num=5, stateID=0, actions=[maybe(p_assisted_flip, flipdown)]
    ),
    maybe(p_random_flip, flipdown),
]

model = Model(
    states=[downstate, upstate],
    world=WorldRules(
        neighborhood="moore",
        proportions=[{"stateID": 0, "parts": 1}, {"stateID": 1, "parts": 1}],
        height=23,
        width=29,
    ),
)


class IsingSim(Simulator):
    def __init__(self, model: Model, tmax=100):
        super().__init__(model)
        self.tmax = tmax
        self.pop_history = []

    def post_step(self):
        super().post_step()
        t = self.time
        p = self.populations()
        self.pop_history.append({"t": t, **p})

    def should_stop(self) -> bool:
        return self.time > self.tmax

    def produce_plots(self):
        df = pd.DataFrame.from_records(self.pop_history)

        fig, axs = plt.subplots(2, 1)
        ax = axs[0]
        ax.plot(df["t"], df["down"] / self.grid.size, "k", label="down")
        ax.plot(df["t"], df["up"] / self.grid.size, "r", label="up")
        # ax.set_ylim(-0.1, 1.1)
        ax.legend()

        ax = axs[1]
        img = ImageMaker().from_simulator(self)
        ax.imshow(img)

        plt.show()

    def finalize(self):
        self.produce_plots()
        super().finalize()


def main() -> None:
    simulator = IsingSim(model, tmax=1000)
    writer = PopulationFileWriter(simulator, "ising.population.csv")
    simulator.writers.append(writer)

    with writer:
        states = tqdm(simulator.run(), total=simulator.tmax)
        for _ in states:
            pass
        # g = imgen(states)
        # run(g, fps_cap=None)


if __name__ == "__main__":
    main()
