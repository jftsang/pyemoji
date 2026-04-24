from functools import cache
from itertools import product, tee
from typing import Iterable, Callable, Self
import numpy as np
import pygame
import pygame.image
from PIL import Image, ImageDraw, ImageFont
from pyemoji.simulator import Simulator


EMOJI_FONT_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"

emoji_font = ImageFont.truetype(EMOJI_FONT_PATH, 64)


class PygameVisualizer:
    def __init__(
        self,
        states: Iterable[Simulator],
        cell_size: int,
        display_size: tuple[int, int],
        fps_cap: int = 30,
    ):
        self.states = states
        self.cell_size = cell_size
        self.display_size = display_size
        self.fps_cap = fps_cap

    def __hash__(self):
        return hash(id(self))  # FIXME

    @classmethod
    def render(cls, states: Iterable[Simulator], cell_size: int = 30, **kwargs) -> Self:
        [next_state] = tee(states, 1)
        s0: Simulator = next(next_state)
        rows, cols = s0.grid.shape
        display_size = (cols * cell_size, rows * cell_size)
        return cls(states, cell_size, display_size, **kwargs)

    @cache
    def get_glyph(self, icon: str) -> pygame.Surface:
        img = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        bbox = emoji_font.getbbox(icon)
        left, top, right, bottom = bbox
        x = 32 - (left + right) // 2
        y = 32 - (top + bottom) // 2
        draw.text((x, y), icon, font=emoji_font, anchor="la", embedded_color=True)

        img = img.resize(
            (self.cell_size, self.cell_size), resample=Image.Resampling.LANCZOS
        )
        return pygame.image.frombytes(img.tobytes(), img.size, "RGBA")

    def run(
        self,
    ):
        frame_gen = self.imgen()
        pygame.init()
        screen = pygame.display.set_mode(self.display_size)
        pygame.display.set_caption("Animation")
        clock = pygame.time.Clock()
        try:
            for surf in frame_gen:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return

                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        print("space")

                screen.blit(surf, (0, 0))
                pygame.display.flip()
                if self.fps_cap:
                    clock.tick(self.fps_cap)
        finally:
            pygame.quit()

    def imgen(self) -> Iterable[pygame.Surface]:
        agent2icon: Callable[[np.ndarray], np.ndarray] = np.vectorize(
            lambda ag: ag.state.icon
        )

        surface = pygame.Surface(size=self.display_size)
        for s in self.states:
            icons = agent2icon(s.grid)
            surface.fill((255, 255, 255))
            rows, cols = s.grid.shape
            for r, c in product(range(rows), range(cols)):
                glyph = self.get_glyph(icons[r, c])
                surface.blit(glyph, (c * self.cell_size, r * self.cell_size))
            yield surface


if __name__ == "__main__":
    from pyemoji.experiments.simpledecay import simulator

    vi = PygameVisualizer.render(simulator.run(), cell_size=20)
    vi.run()
