from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from streamerate import stream

from pyemoji.simulator import Simulator

cmap = plt.get_cmap("Pastel2")
# lookup table
rgb_lut = (cmap(np.arange(cmap.N))[:, :3] * 255).astype(np.uint8)


class ImageMaker:
    def __init__(self):
        self._buf: np.ndarray = np.empty(0)

    def from_simulator(self, s: Simulator) -> Image.Image:
        # reuse an existing buffer if we can
        if self._buf.shape != s.grid.shape:
            self._buf = np.zeros((*s.grid.shape, 3), dtype="uint8")

        sids = np.vectorize(lambda ag: ag.state.id)(s.grid)
        self._buf[:] = rgb_lut[sids % cmap.N]
        return Image.fromarray(self._buf)

    def follow(self, s: Iterable[Simulator]):
        for _s in s:
            yield self.from_simulator(_s)


if __name__ == "__main__":
    from pyemoji.experiments.decay import simulator

    states = simulator.run()
    images = ImageMaker().follow(stream(states).take(100))
    print(next(images))
