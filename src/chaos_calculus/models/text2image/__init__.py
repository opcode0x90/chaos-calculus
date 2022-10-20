from enum import Enum, auto

import numpy as np

###############################################################################


class BatchMode(Enum):
    """Enumeration for describing the batch mode for image generation."""
    DISABLED = auto()
    FULL = auto()
    CHUNKED = auto()


class Model:
    """Common interface for text-to-image model with different implementations."""

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

        for batch_size in batches:
            im = self.generate(prompt, negative_prompt, batch_size)
            images.extend(im)

        return images

    def generate_plot(self,
                      prompt: str,
                      negative_prompt: str | None = None,
                      grid: tuple[int, int] | None = None) -> None:
        """Plot generated image on specified grid."""
        pass

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
            batches.append(size % batch_size)
        elif batch_mode == BatchMode.DISABLED:
            batches = [1] * size
        else:
            raise NotImplementedError

        return batches
