from collections import Counter

import numpy as np

from rules import Rules
from streamerate import stream


class Simulator:
    def __init__(self, rules: Rules):
        self.rules = rules
        self.grid = np.zeros(
            (self.rules.world.size["height"], self.rules.world.size["width"]), dtype=int
        )
        self.time: int = 0

        print(self)
        print(self.populations())

    def populations(self) -> dict:
        return (
            stream(Counter(self.grid.flatten()).items())
            .mapKeys(lambda sid: self.rules.statemap[sid].name)
            .to_dict()
        )

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "\n".join(
            ("".join(stream(self.grid[i, :]).map(self.rules.sid2char).to_list()))
            for i in range(self.grid.shape[0])
        )
