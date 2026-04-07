import random
from collections import Counter
from itertools import product
from typing import Iterable, Self, Any

import numpy as np
from streamerate import stream

from pyemoji.agent import Agent
from pyemoji.file_writers import FileWriter
from pyemoji.model import Model, State


class Simulator:
    def __init__(self, model: Model):
        self.model = model

        # Agents don't actually "move around". When an agent "moves" to
        # another cell, what actually happens is that the agent at the
        # destination cell is updated.
        self.grid = np.empty((self.height, self.width), dtype=object)
        # flat list of agents that will be shuffled
        self._agents = np.empty(self.height * self.width, dtype=object)

        # Initialise with arbitrary agents. Their states will be set
        # later.
        for i, j in product(range(self.height), range(self.width)):
            agent = Agent(i, j, simulator=self)
            self._agents[i * self.width + j] = agent
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

        for i, j in product(range(self.height), range(self.width)):
            self.grid[i, j].force_state(self.states[int(states[i * self.width + j])])

    @property
    def states(self) -> dict[int, State]:
        return self.model.statemap

    @property
    def width(self) -> int:
        return self.model.world.width

    @property
    def height(self) -> int:
        return self.model.world.height

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

    def get_all_agents(self) -> np.ndarray:
        return self._agents

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
        for i, j in product(range(self.height), range(self.width)):
            agent = self.grid[i, j]
            state: State = random.choices(self.model.states, weights=probs)[0]
            agent.force_state(state)

    def step(self):
        self.time += 1
        agents = self.get_all_agents()
        np.random.shuffle(agents)
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
        """Runs after each step. Put stuff like writing to output files
        or keeping track of the system state here.
        """
        # override me
        self.write_to_output_files()

    def should_stop(self) -> bool:
        """Termination condition. Gets checked before each step."""
        # override me
        return False

    def post_stop(self):
        """Runs after the termination condition `self.should_stop` is
        attained, that is, after a successful end to the simulation. Put
        stuff like postprocessing here.
        """
        # override me
        pass

    def handle_error(self, exc: BaseException):
        """Runs if the simulation stops without terminating
        successfully. Default behaviour is to dump the state and then
        reraise the exception.
        """
        print(self.dump())
        raise exc

    def finalize(self):
        """Unconditionally runs when the simulation is stopped, whether
        it terminated successfully or encountered an exception. Put
        stuff like dumping the state or updating files here.
        """
        # override me

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

        # BaseException, not Exception, so that KeyboardInterrupt is
        # caught. It is the responsibility of `handle_error` to decide
        # what to do.
        except BaseException as exc:
            self.handle_error(exc)

        finally:
            self.finalize()

    def write_output_headers(self):
        """Triggers all the writers to write their header rows. This
        should run once at the start of a simulation.
        """
        for writer in self.writers:
            writer.write_header()

    def write_to_output_files(self):
        """Triggers all the writers to write a row. This should run
        after each step.
        """
        for writer in self.writers:
            writer.write_state()
