import random
from collections import Counter
from typing import Iterable

import numpy as np
from streamerate import stream

from rules import Rules, State


class Agent:
    def __init__(self, x, y, rules: Rules):
        self.x: int = x
        self.y: int = y
        self.state: State = rules.default_state

        self.rules: Rules = rules

        self.updated: bool = False
        self.next_state: State = rules.default_state

    def mark_as_not_updated(self):
        self.updated = False

    def calculate_next_state(self):
        if self.updated:
            return
        self.next_state = self.state

    def go_to_next_state(self):
        self.state = self.next_state

    def force_state(self, new_state: State):
        self.state = self.next_state = new_state
        self.updated = True

    def perform_actions(self):
        actions = self.state.actions
        initial_next_state = self.next_state
        for action in actions:
            action.step(self)
            if self.next_state != initial_next_state:
                return


class Simulator:
    def __init__(self, rules: Rules):
        self.rules = rules
        self.grid = np.empty((self.height, self.width), dtype=object)
        self.time: int = 0

    @property
    def states(self) -> dict[int, State]:
        return self.rules.statemap

    @property
    def width(self) -> int:
        return self.rules.world.size["width"]

    @property
    def height(self) -> int:
        return self.rules.world.size["height"]

    def populations(self) -> dict[str, int]:
        return Counter(
            stream(self.get_all_agents()).map(lambda agent: agent.state.name).to_list()
        )

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
        x = agent.x  # horizontal position, second
        y = agent.y  # vertical position, first
        hood = self.rules.world.neighborhood
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
            if x >= self.width:
                return False
            if y < 0:
                return False
            if y >= self.height:
                return False
            return True

        coords = stream(coords).starfilter(legal).to_list()
        neighbors = [self.grid[xy[1], xy[0]] for xy in coords]
        return neighbors

    def setup_ics(self):
        self.time = 0
        # set up initial conditions
        probs = [0] * len(self.rules.states)
        for sp in self.rules.world.proportions:
            sid, p = sp["stateID"], sp["parts"]
            probs[sid] = p
        for i in range(self.height):
            for j in range(self.width):
                agent = Agent(i, j, rules=self.rules)

                state: State = random.choices(self.rules.states, weights=probs)[0]

                agent.force_state(state)
                self.grid[i, j] = agent

        print(self.rules)
        print(self)
        print(self.populations())

    def step(self):
        self.time += 1
        for agent in self.get_all_agents():
            neighbors = self.get_neighbors(agent)
            ...

    def run(self) -> Iterable[tuple[int, np.ndarray, dict[str, int]]]:
        self.setup_ics()
        while True:
            yield self.time, self.grid, self.populations()

            self.step()
