from collections import Counter
from typing import Iterable

import numpy as np

from rules import Rules, State
from streamerate import stream


class Simulator:
    def __init__(self, rules: Rules):
        self.rules = rules
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.time: int = 0

        print(self.rules)
        print(self)
        print(self.populations())

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
        return (
            stream(Counter(self.grid.flatten()).items())
            .mapKeys(lambda sid: self.states[sid].name)
            .to_dict()
        )

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "\n".join(
            ("".join(stream(self.grid[i, :]).map(self.rules.sid2char).to_list()))
            for i in range(self.height)
        )

    def setup_ics(self):
        self.time = 0
        # set up initial conditions

    def step(self):
        self.time += 1
        NotImplemented

    def run(self) -> Iterable[tuple[int, np.ndarray, dict[str, int]]]:
        self.setup_ics()
        while True:
            yield self.time, self.grid, self.populations()

            self.step()
