from functools import cache
from itertools import product, tee
from typing import Iterable, Callable
import numpy as np
import pygame
import pygame.image
from PIL import Image, ImageDraw, ImageFont
from pyemoji.simulator import Simulator


EMOJI_FONT_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"

emoji_font = ImageFont.truetype(EMOJI_FONT_PATH, 64)


@cache
def get_glyph(icon: str, cell_size: int) -> pygame.Surface:
    img = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    bbox = emoji_font.getbbox(icon)
    left, top, right, bottom = bbox
    x = 32 - (left + right) // 2
    y = 32 - (top + bottom) // 2
    draw.text((x, y), icon, font=emoji_font, anchor="la", embedded_color=True)

    img = img.resize((cell_size, cell_size), resample=Image.Resampling.LANCZOS)
    return pygame.image.frombytes(img.tobytes(), img.size, "RGBA")


def run(
    frame_gen: Iterable[pygame.Surface],
    display_size: tuple[int, int],
    fps_cap: int = 30,
):
    pygame.init()
    screen = pygame.display.set_mode(display_size)
    pygame.display.set_caption("Animation")
    clock = pygame.time.Clock()
    try:
        for surf in frame_gen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            if fps_cap:
                clock.tick(fps_cap)
    finally:
        pygame.quit()


def imgen(
    states: Iterable[Simulator], display_size: tuple[int, int], cell_size: int
) -> Iterable[pygame.Surface]:
    agent2icon: Callable[[np.ndarray], np.ndarray] = np.vectorize(
        lambda ag: ag.state.icon
    )

    surface = pygame.Surface(size=display_size)
    for s in states:
        icons = agent2icon(s.grid)
        surface.fill((255, 255, 255))
        rows, cols = s.grid.shape
        for r, c in product(range(rows), range(cols)):
            glyph = get_glyph(icons[r, c], cell_size)
            surface.blit(glyph, (c * cell_size, r * cell_size))
        yield surface


def render(simulator: Simulator) -> None:
    states: Iterable[Simulator] = simulator.run()
    [next_state] = tee(states, 1)
    s0: Simulator = next(next_state)
    rows, cols = s0.grid.shape
    cell_size = 30
    display_size: tuple[int, int] = (cols * cell_size, rows * cell_size)
    g: Iterable[pygame.Surface] = imgen(states, display_size, cell_size=cell_size)
    run(g, display_size=display_size)


if __name__ == "__main__":
    from pyemoji.experiments.simpledecay import simulator

    render(simulator)
