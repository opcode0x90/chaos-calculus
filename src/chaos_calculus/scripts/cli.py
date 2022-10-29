from email.policy import default

import click
from chaos_calculus.models.text2image import get_models
from chaos_calculus.repl import Repl
from chaos_calculus.util import timing

################################################################################

BANNER = """\
   ________                        ______      __           __
  / ____/ /_  ____ _____  _____   / ________ _/ _______  __/ __  _______
 / /   / __ \/ __ `/ __ \/ ___/  / /   / __ `/ / ___/ / / / / / / / ___/
/ /___/ / / / /_/ / /_/ (__  )  / /___/ /_/ / / /__/ /_/ / / /_/ (__  )
\____/_/ /_/\__,_/\____/____/   \____/\__,_/_/\___/\__,_/_/\__,_/____/

(Use Ctrl+C plus Return to exit.)
"""

# max pictures per row
MAX_COLUMN = 3
IMAGES = 9

# list of available models
MODELS = get_models()

################################################################################


@click.command()
@click.option('--backend',
              type=click.Choice(list(MODELS.keys()), case_sensitive=False),
              default='keras',
              show_default=True,
              help="Specify which backend for Stable Diffusion to load.")
@click.option('--width', type=int, default=512, show_default=True, help="Width of generated image.")
@click.option('--height', type=int, default=512, show_default=True, help="Height of generated image.")
@click.option('--batch-size',
              type=int,
              default=3,
              help="Number of images to generate per batch. Reduce batch size if experiencing out of memory error.")
@click.option('--mixed-fp/--no-mixed-fp', is_flag=True, default=True, show_default=True, help="Enable mixed precision.")
@click.option('--jit-compile/--no-jit-compile',
              is_flag=True,
              default=True,
              show_default=True,
              help="Enable JIT compile.")
@click.option('--fp16',
              is_flag=True,
              default=False,
              show_default=True,
              help="Load model in reduced precision mode (float16). This will consume less GPU memory.")
@click.option(
    '--safemode/--no-safemode',
    is_flag=True,
    default=True,
    show_default=True,
    help=
    "Enable built-in safety checker. This will generate a warning when potentially NSFW image is created and replace them with black image."
)
def main(backend: str, width: int, height: int, batch_size: int, mixed_fp: bool, jit_compile: bool, fp16: bool,
         safemode: bool):

    if not (Model := MODELS.get(backend)):
        raise click.UsageError(f"Error occured while loading backend: `{backend}`")

    # get example prompt for the model
    prompt = Model.prompt

    with Repl("Prompt", prefill=prompt, banner=BANNER) as repl:
        click.echo("Initializing Stable Diffusion...")
        with timing("Model initialized"):
            model = Model(width=width,
                          height=height,
                          batch_size=batch_size,
                          mixed_fp=mixed_fp,
                          jit_compile=jit_compile,
                          fp16=fp16,
                          safemode=safemode)

        for prompt in repl:
            positive, _, negative = prompt.partition("~")
            positive = positive.strip()
            negative = negative.strip()

            with timing("Image generated"):
                model.generate_plot(positive, negative)


################################################################################
