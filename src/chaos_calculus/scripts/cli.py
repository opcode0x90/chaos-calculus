import textwrap

import click
from chaos_calculus.models.text2image.keras import KerasModel
from chaos_calculus.repl import Repl
from chaos_calculus.util import timing

################################################################################

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
            model = KerasModel(width=512, height=512, batch_size=9, mixed_fp=True, jit_compile=True)

        for prompt in repl:
            positive, _, negative = prompt.partition("~")
            positive = positive.strip()
            negative = negative.strip()

            with timing("Image generated"):
                model.generate_plot(positive, negative)


################################################################################
