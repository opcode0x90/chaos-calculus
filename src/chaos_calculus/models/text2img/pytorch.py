import numpy as np
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline, EulerDiscreteScheduler

from . import BatchMode, Model

################################################################################


class PyTorchModel(Model):
    """Legacy Stable Diffusion v1.4 implementation using pytorch and diffusers pipeline."""

    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = "a photo of an astronaut riding a horse on mars"

    def __init__(self,
                 width=512,
                 height=512,
                 batch_mode: BatchMode = BatchMode.CHUNKED,
                 batch_size: int = 3,
                 fp16: bool = False,
                 mixed_fp: bool = False,
                 safemode: bool = True,
                 *args,
                 **kwargs) -> None:
        super().__init__(width, height, batch_mode, batch_size, *args, **kwargs)

        self.fp16 = fp16
        self.mixed_fp = mixed_fp

        if fp16:
            # use model with reduced precision
            kwargs |= {'torch_dtype': torch.float16, 'revision': "fp16"}

        if not safemode:
            # disable built-in safety checker (lots of FP when used with anime models)
            kwargs |= {'safety_checker': None}

        # initialize the model
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


################################################################################


#
# Variants of original Stable Diffusion model
#
class StableDiffusionv15Model(PyTorchModel):
    """Legacy Stable Diffusion model trained with extra steps. https://huggingface.co/runwayml/stable-diffusion-v1-5"""

    model_id = "runwayml/stable-diffusion-v1-5"


class StableDiffusionv2Model(PyTorchModel):
    """Stable Diffusion model version 2. https://huggingface.co/stabilityai/stable-diffusion-2"""

    model_id = "stabilityai/stable-diffusion-2"
    prompt = "a professional photograph of an astronaut riding a horse"

    def __init__(self,
                 width=512,
                 height=512,
                 batch_mode: BatchMode = BatchMode.CHUNKED,
                 batch_size: int = 3,
                 fp16: bool = False,
                 mixed_fp: bool = False,
                 safemode: bool = True,
                 *args,
                 **kwargs) -> None:
        super().__init__(width, height, batch_mode, batch_size, *args, **kwargs)

        self.fp16 = fp16
        self.mixed_fp = mixed_fp

        if fp16:
            # use model with reduced precision
            kwargs |= {'torch_dtype': torch.float16, 'revision': "fp16"}

        if not safemode:
            # disable built-in safety checker (lots of FP when used with anime models)
            kwargs |= {'safety_checker': None}

        scheduler = EulerDiscreteScheduler.from_pretrained(self.model_id,
                                                           subfolder="scheduler",
                                                           prediction_type="v_prediction")
        pipe = DiffusionPipeline.from_pretrained(self.model_id, scheduler=scheduler, use_auth_token=True, **kwargs)
        pipe = pipe.to(self.device)

        self.pipe = pipe


class WaifuModel(PyTorchModel):
    """Stable Diffusion model that has been conditioned on high-quality anime images through fine-tuning.. https://huggingface.co/hakurei/waifu-diffusion"""

    model_id = "hakurei/waifu-diffusion"
    prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"


class CyberpunkEdgerunnerModel(PyTorchModel):
    """Stable Diffusion model finetuned to Cyberpunk Edgerunner anime. https://huggingface.co/DGSpitzer/Cyberpunk-Anime-Diffusion"""

    model_id = "DGSpitzer/Cyberpunk-Anime-Diffusion"
    prompt = "Anime fine details portrait of school girl in front of modern tokyo city landscape on the background deep bokeh, anime masterpiece, 8k, sharp high quality anime, a beautiful perfect face girl in dgs illustration style"
