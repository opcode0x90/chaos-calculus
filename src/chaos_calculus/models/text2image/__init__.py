from enum import Enum, auto
from typing import Type

import numpy as np
from chaos_calculus.util import grid_plot

###############################################################################


class BatchMode(Enum):
    """Enumeration for describing the batch mode used for image generation."""
    DISABLED = auto()
    FULL = auto()
    CHUNKED = auto()


class Model:
    """Common interface for text-to-image model with different implementations."""

    width: int
    height: int
    batch_mode: BatchMode
    batch_size: int
    args: tuple
    kwargs: dict

    def __init__(self,
                 width=512,
                 height=512,
                 batch_mode: BatchMode = BatchMode.CHUNKED,
                 batch_size: int = 3,
                 *args,
                 **kwargs) -> None:
        self.width = width
        self.height = height
        self.batch_mode = batch_mode
        self.batch_size = batch_size

        self.args = args
        self.kwargs = kwargs

    def generate(self, prompt: str, negative_prompt: str | None = None, batch_size: int = 1) -> list[np.ndarray]:
        """Generate image using given prompts."""
        raise NotImplementedError

    def generate_batch(self,
                       size: int,
                       prompt: str,
                       negative_prompt: str | None = None,
                       batch_mode: BatchMode | None = None,
                       batch_size: int | None = None) -> list[np.ndarray]:
        """Generate a bunch of images using given prompts."""
        images = []
        batches = self.split_batches(size, batch_mode, batch_size)
        print(f"{batches=}")

        for batch_size in batches:
            im = self.generate(prompt, negative_prompt, batch_size)
            images.extend(im)

        return images

    def generate_plot(self,
                      prompt: str,
                      negative_prompt: str | None = None,
                      figsize: tuple[float, float] = (19.2, 10.8),
                      grid: tuple[int, int] | None = None,
                      batch_mode: BatchMode | None = None,
                      batch_size: int | None = None,
                      maximized: bool = True,
                      blocked: bool = False) -> None:
        """Plot generated image on specified grid."""
        if not grid:
            # attempt to generate grid that fills up screen space
            grid = self.make_grid(figsize)

        size = grid[0] * grid[1]
        print(f"{grid=}")
        print(f"{size=}")
        images = self.generate_batch(size, prompt, negative_prompt, batch_mode, batch_size)

        grid_plot(images, grid, figsize, maximized, blocked)

    def make_grid(self, figsize: tuple[float, float] = (19.2, 10.8)) -> tuple[int, int]:
        """Generate a grid that attemps to fill up screen space."""
        rows = (figsize[0] * 100) // self.height
        columns = (figsize[1] * 100) // self.width
        return (int(rows), int(columns))

    def split_batches(self, size: int, batch_mode: BatchMode | None = None, batch_size: int | None = None) -> list[int]:
        """Divide batches according to `batch_size` and configured `batch_mode`."""

        if not batch_mode:
            batch_mode = self.batch_mode
        if not batch_size:
            batch_size = self.batch_size

        if batch_mode == BatchMode.FULL or size <= batch_size:
            batches = [size]
        elif batch_mode == BatchMode.CHUNKED:
            batches = [batch_size] * (size // batch_size)
            if (remainder := size % batch_size):
                batches.append(remainder)
        elif batch_mode == BatchMode.DISABLED:
            batches = [1] * size
        else:
            raise NotImplementedError

        return batches


###############################################################################


# list of registered models
def get_models() -> dict[str, Type[Model]]:
    from .keras import KerasModel
    from .pytorch import PyTorchModel

    return {'keras': KerasModel, 'pytorch': PyTorchModel}
