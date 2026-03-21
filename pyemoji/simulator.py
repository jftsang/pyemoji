import random
from collections import Counter
from typing import Iterable, Self, Any

import numpy as np
from streamerate import stream  # type: ignore

from pyemoji.agent import Agent
from pyemoji.file_writers import FileWriter
from pyemoji.model import Model, State


class Simulator:
    def __init__(self, model: Model):
        self.model = model
        self.grid = np.empty((self.height, self.width), dtype=object)
        for i in range(self.height):
            for j in range(self.width):
                agent = Agent(i, j, simulator=self)
                self.grid[i, j] = agent

        self.time: int = 0

        self.writers: list[FileWriter] = []

    def dump(self) -> dict[str, Any]:
        if len(self.states) > 15:
            raise ValueError("Too many states to serialize")
        agent2sid = np.vectorize(lambda ag: ag.state.id if ag.state else 0)
        serialized_grid: np.ndarray = agent2sid(self.grid)
        return {
            "time": self.time,
            "grid": "".join(f"{x:x}" for x in serialized_grid.flat),
        }

    def load(self, d):
        self.time = d["time"]
        states = d["grid"]

        for i in range(self.height):
            for j in range(self.width):
                self.grid[i, j].force_state(
                    self.states[int(states[i * self.width + j])]
                )

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
        # make sure we pick up states with zero population
        state_names = [state.name for state in self.model.states]
        count = Counter(
            stream(self.get_all_agents()).map(lambda agent: agent.state.name).to_list()
        )
        return {sn: count.get(sn, 0) for sn in state_names}

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

        def legal(x: int, y: int) -> bool:
            if x < 0:
                return False
            if x >= self.height:
                return False
            if y < 0:
                return False
            if y >= self.width:
                return False
            return True

        coords = (
            stream(coords)
            .starfilter(legal)  # ty:ignore[invalid-argument-type]
            .to_list()
        )
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
                agent = self.grid[i, j]
                state: State = random.choices(self.model.states, weights=probs)[0]  # type: ignore
                agent.force_state(state)

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

    def pre_step(self):
        # override me
        pass

    def post_step(self):
        # override me
        self.write_to_output_files()

    def should_stop(self) -> bool:
        """Termination condition. Gets checked before each step."""
        # override me
        return False

    def post_stop(self):
        # override me
        pass

    def finalize(self):
        # override me
        pass

    def run(self) -> Iterable[Self]:
        self.setup_ics()
        self.write_output_headers()
        try:
            while not self.should_stop():
                yield self
                self.pre_step()
                self.step()
                self.post_step()
            self.post_stop()

        finally:
            self.finalize()

    def write_output_headers(self):
        for writer in self.writers:
            writer.write_header()

    def write_to_output_files(self):
        for writer in self.writers:
            writer.write_state()
