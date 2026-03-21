from typing import Iterable

import numpy as np
import pygame
from PIL import Image
from matplotlib import pyplot as plt

from pyemoji.simulator import Simulator


def run(frame_gen, fps_cap=30):
    pygame.init()
    screen = None
    clock = pygame.time.Clock()

    try:
        for frame in frame_gen:
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(frame)
            frame = frame.convert("RGB")

            w, h = frame.size
            scale = int(640 / max(w, h))
            display_size = (w * scale, h * scale)

            if screen is None:

                screen = pygame.display.set_mode(display_size)
                pygame.display.set_caption("Animation")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            surf = pygame.image.fromstring(frame.tobytes(), frame.size, "RGB")
            scaled = pygame.transform.scale(surf, display_size)
            screen.blit(scaled, (0, 0))
            pygame.display.flip()
            if fps_cap:
                clock.tick(fps_cap)

    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


# --- example usage ---
cmap = plt.get_cmap("Pastel2")
# lookup table
rgb_lut = (cmap(np.arange(cmap.N))[:, :3] * 255).astype(np.uint8)


def imgen(states: Iterable[Simulator]):
    arr = None
    for s in states:

        sids = np.vectorize(lambda ag: ag.state.id)(s.grid)
        arr = (
            arr if arr is not None else np.zeros((*s.grid.shape, 3), dtype="uint8")
        )  # only initialize once
        arr[:] = rgb_lut[sids % cmap.N]

        yield Image.fromarray(arr)


# from experiments.randomwalk import simulator
from experiments.decay import simulator

if __name__ == "__main__":
    states = simulator.run()
    g = imgen(states)
    # next(g)
    run(g, fps_cap=10)
