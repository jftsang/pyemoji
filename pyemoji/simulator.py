import random
from collections import Counter
from typing import Iterable, Self

import numpy as np
from streamerate import stream  # type: ignore

from agent import Agent
from model import Model, State


class Simulator:
    def __init__(self, model: Model):
        self.model = model
        self.grid = np.empty((self.height, self.width), dtype=object)
        self.time: int = 0

    @property
    def states(self) -> dict[int, State]:
        return self.model.statemap

    @property
    def width(self) -> int:
        return self.model.world.size["width"]

    @property
    def height(self) -> int:
        return self.model.world.size["height"]

    def populations(self) -> dict[str, int]:
        state_names = [state.name for state in self.model.states]
        p = {n: 0 for n in state_names}
        count = Counter(
            stream(self.get_all_agents()).map(lambda agent: agent.state.name).to_list()
        )
        p.update(count)
        return p

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "\n".join(
            (
                "".join(
                    stream(self.grid[i, :])
                    .map(lambda agent: agent.state.icon)
                    .to_list()
                )
            )
            for i in range(self.height)
        )

    def get_all_agents(self) -> list[Agent]:
        return list(self.grid.flatten())

    def get_neighbors(self, agent: Agent):
        x = agent.x
        y = agent.y
        hood = self.model.world.neighborhood
        coords: list[tuple[int, int]]
        if hood == "moore":
            coords = [
                (x - 1, y - 1),
                (x, y - 1),
                (x + 1, y - 1),
                (x - 1, y),
                (x + 1, y),
                (x - 1, y + 1),
                (x, y + 1),
                (x + 1, y + 1),
            ]
        elif hood == "neumann":
            coords = [
                (x, y - 1),
                (x - 1, y),
                (x + 1, y),
                (x, y + 1),
            ]
        else:
            raise ValueError

        def legal(x, y):
            if x < 0:
                return False
            if x >= self.height:
                return False
            if y < 0:
                return False
            if y >= self.width:
                return False
            return True

        coords = stream(coords).starfilter(legal).to_list()
        neighbors = [self.grid[xy[0], xy[1]] for xy in coords]
        return neighbors

    def setup_ics(self):
        self.time = 0
        # set up initial conditions
        probs = [0] * len(self.model.states)
        for sp in self.model.world.proportions:
            sid, p = sp["stateID"], sp["parts"]
            probs[sid] = p
        for i in range(self.height):
            for j in range(self.width):
                agent = Agent(i, j, simulator=self)

                state: State = random.choices(self.model.states, weights=probs)[0]  # type: ignore

                agent.force_state(state)
                self.grid[i, j] = agent

    def step(self):
        self.time += 1
        self.pre_step()
        agents = self.get_all_agents()
        random.shuffle(agents)
        for agent in agents:
            agent.mark_as_not_updated()
        for agent in agents:
            agent.calculate_next_state()
        for agent in agents:
            agent.go_to_next_state()

        self.post_step()

    def pre_step(self):
        # override me
        pass

    def post_step(self):
        # override me
        pass

    def should_stop(self) -> bool:
        # override me
        return False

    def post_stop(self):
        # override me
        pass

    def run(self) -> Iterable[Self]:
        self.setup_ics()
        try:
            while not self.should_stop():
                yield self
                self.step()

        finally:
            self.post_stop()
