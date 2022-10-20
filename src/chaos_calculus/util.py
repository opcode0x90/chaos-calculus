#
# bunch o' utilities
#
import time
from contextlib import contextmanager

import click

################################################################################


@contextmanager
def timing(prompt: str):
    start = time.monotonic()
    yield
    click.echo(f"{prompt} in {time.monotonic() - start:.2f} seconds.")
