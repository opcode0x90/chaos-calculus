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
                 fp16: bool = False,
                 mixed_fp: bool = False,
                 *args,
                 **kwargs) -> None:
        super().__init__(width, height, batch_mode, batch_size, *args, **kwargs)

        self.fp16 = fp16
        self.mixed_fp = mixed_fp

        # initialize the model
        if fp16:
            # use model with reduced precision
            kwargs |= {'torch_dtype': torch.float16, 'revision': "fp16"}

        pipe = StableDiffusionPipeline.from_pretrained(self.model_id, use_auth_token=True, **kwargs)
        pipe = pipe.to(self.device)

        self.pipe = pipe

    def generate(self, prompt: str, negative_prompt: str | None = None, batch_size: int = 1) -> list[np.ndarray]:
        """Generate image using given prompt."""

        with torch.autocast("cuda", dtype=torch.float16, enabled=self.mixed_fp):
            results = self.pipe(prompt,
                                negative_prompt=negative_prompt,
                                num_images_per_prompt=batch_size,
                                guidance_scale=7.5)

            # convert PIL image to numpy
            return [np.array(image) for image in results.images]
