import time
from contextlib import contextmanager

import click
import keras_cv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

try:
    import gnureadline as readline
except ImportError:
    import readline

################################################################################

# use mixed precision for speedups
keras.mixed_precision.set_global_policy("mixed_float16")

# max pictures per row
MAX_COLUMN = 3
IMAGES = 9

################################################################################


@click.command()
def main():
    click.echo("Initializing Stable Diffusion...")
    with timing("Model initialized"):
        # model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
        model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=True)
        # model = keras_cv.models.StableDiffusion(img_width=640, img_height=640, jit_compile=True)
        # model = keras_cv.models.StableDiffusion(img_width=640, img_height=1080 - 32, jit_compile=True)
    click.echo()

    # prompt = "photograph of an astronaut riding a horse"
    prompt = "ultra-detailed. uhd 8k, artstation, cryengine, octane render, unreal engine. photograph of an astronaut riding a horse"

    while True:
        if not (prompt := get_prompt(prompt)):
            continue

        positive, _, negative = prompt.partition("~")
        positive = positive.strip()
        negative = negative.strip()

        with timing("Image generated"):
            encoded_positive = model.encode_text(positive)
            model._get_unconditional_context = lambda: model.encode_text(negative)

            images = model.generate_image(encoded_positive, batch_size=IMAGES)
            # images = []
            # for _ in range(3):
            #     im = model.generate_image(encoded_positive)
            #     images.extend(im)

        plt.figure(figsize=(19.2, 10.8))
        for i in range(len(images)):
            plt.subplots_adjust(0, 0, 1, 1, 0, 0)
            plt.subplot(len(images) // MAX_COLUMN, MAX_COLUMN, i + 1)
            plt.imshow(images[i])
            plt.axis("off")

        mgr = plt.get_current_fig_manager()
        mgr.window.state('zoomed')
        plt.show()


################################################################################


@contextmanager
def timing(prompt: str):
    start = time.monotonic()
    yield
    click.echo(f"{prompt} in {time.monotonic() - start:.2f} seconds.")


def get_prompt(prefill) -> str:

    def hook():
        readline.insert_text(prefill)
        readline.redisplay()

    readline.set_pre_input_hook(hook)
    prompt = input("Prompt (Ctrl+C and Enter to quit): ")
    readline.set_pre_input_hook()

    return prompt
