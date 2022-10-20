import textwrap

import click
import keras_cv
import matplotlib.pyplot as plt
import tensorflow as tf
from chaos_calculus.repl import Repl
from chaos_calculus.util import timing
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
    banner = textwrap.dedent("""\
           ________                        ______      __           __
          / ____/ /_  ____ _____  _____   / ________ _/ _______  __/ __  _______
         / /   / __ \/ __ `/ __ \/ ___/  / /   / __ `/ / ___/ / / / / / / / ___/
        / /___/ / / / /_/ / /_/ (__  )  / /___/ /_/ / / /__/ /_/ / / /_/ (__  )
        \____/_/ /_/\__,_/\____/____/   \____/\__,_/_/\___/\__,_/_/\__,_/____/

        (Press Ctrl+C and Enter to abort.)
        """)

    # prompt = "photograph of an astronaut riding a horse"
    prompt = "ultra-detailed. uhd 8k, artstation, cryengine, octane render, unreal engine. photograph of an astronaut riding a horse"

    with Repl("Prompt", prefill=prompt, banner=banner) as repl:
        click.echo("Initializing Stable Diffusion...")
        with timing("Model initialized"):
            # model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
            model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=True)
            # model = keras_cv.models.StableDiffusion(img_width=640, img_height=640, jit_compile=True)
            # model = keras_cv.models.StableDiffusion(img_width=640, img_height=1080 - 32, jit_compile=True)

        for prompt in repl:
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
