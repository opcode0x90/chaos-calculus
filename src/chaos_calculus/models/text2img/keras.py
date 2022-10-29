import keras_cv
import numpy as np
from tensorflow import keras

from . import BatchMode, Model

################################################################################


class KerasModel(Model):
    """Stable Diffusion implementation using keras_cv."""

    prompt = "ultra-detailed. uhd 8k, artstation, cryengine, octane render, unreal engine. a photograph of an astronaut riding a horse"

    def __init__(self,
                 width=512,
                 height=512,
                 batch_mode: BatchMode = BatchMode.CHUNKED,
                 batch_size: int = 3,
                 mixed_fp: bool = False,
                 jit_compile: bool = False,
                 *args,
                 **kwargs) -> None:
        super().__init__(width, height, batch_mode, batch_size, *args, **kwargs)

        if mixed_fp:
            # use mixed precision for speedups
            keras.mixed_precision.set_global_policy("mixed_float16")

        # initialize the model
        self.model = keras_cv.models.StableDiffusion(img_width=width, img_height=height, jit_compile=jit_compile)

    def generate(self, prompt: str, negative_prompt: str | None = None, batch_size: int = 1) -> list[np.ndarray]:
        """Generate image using given prompt."""
        encoded = self.model.encode_text(prompt)
        self.model._get_unconditional_context = lambda: self.model.encode_text(negative_prompt)

        return self.model.generate_image(encoded, batch_size=batch_size)
