import numpy as np
import torch
from diffusers import StableDiffusionPipeline

from . import BatchMode, Model

################################################################################


class PyTorchModel(Model):
    """Stable Diffusion implementation using pytorch. (using diffusers pipeline)"""

    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"

    def __init__(self,
                 width=512,
                 height=512,
                 batch_mode: BatchMode = BatchMode.CHUNKED,
                 batch_size: int = 3,
                 *args,
                 **kwargs) -> None:
        batch_size = 1
        super().__init__(width, height, batch_mode, batch_size, *args, **kwargs)

        # initialize the model
        pipe = StableDiffusionPipeline.from_pretrained(self.model_id, use_auth_token=True)
        pipe = pipe.to(self.device)

        self.pipe = pipe

    def generate(self, prompt: str, negative_prompt: str | None = None, batch_size: int = 1) -> list[np.ndarray]:
        """Generate image using given prompt."""

        with torch.autocast("cuda"):
            # convert PIL image to numpy
            return [np.array(image) for image in self.pipe(prompt, guidance_scale=7.5).images]
