#
# bunch o' utilities
#
import time
from contextlib import contextmanager

import click
import matplotlib.pyplot as plt
import numpy as np

################################################################################


@contextmanager
def timing(prompt: str):
    start = time.monotonic()
    yield
    click.echo(f"{prompt} in {time.monotonic() - start:.2f} seconds.")


def grid_plot(images: list[np.ndarray],
              grid: tuple[int, int],
              figsize: tuple[float, float] = (19.2, 10.8),
              maximized: bool = True,
              blocked: bool = False):
    plt.figure(figsize=figsize)
    for i in range(len(images)):
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        plt.subplot(len(images) // grid[1], grid[1], i + 1)
        plt.imshow(images[i])
        plt.axis("off")

    if maximized:
        mgr = plt.get_current_fig_manager()
        mgr.window.state('zoomed')

    plt.show(block=blocked)
