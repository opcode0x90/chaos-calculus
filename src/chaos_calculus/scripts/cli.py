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

(Press Ctrl+C and Enter to abort.)
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
              help="Specify which backend for Stable Diffusion to load.")
def main(backend: str):
    # prompt = "photograph of an astronaut riding a horse"
    prompt = "ultra-detailed. uhd 8k, artstation, cryengine, octane render, unreal engine. a photograph of an astronaut riding a horse"

    with Repl("Prompt", prefill=prompt, banner=BANNER) as repl:
        click.echo("Initializing Stable Diffusion...")
        with timing("Model initialized"):
            Model = MODELS[backend]
            model = Model(width=512, height=512, fp16=True, mixed_fp=True, jit_compile=True)

        for prompt in repl:
            positive, _, negative = prompt.partition("~")
            positive = positive.strip()
            negative = negative.strip()

            with timing("Image generated"):
                model.generate_plot(positive, negative)


################################################################################
