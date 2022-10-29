from typing import Optional

import click

try:
    # macOS
    import gnureadline as readline
except ImportError:
    # everyone else
    import readline

################################################################################


class Repl:
    """Basic implementation of read-eval-print-loop interface."""

    def __init__(self,
                 prompt: str,
                 prefill: Optional[str] = None,
                 banner: Optional[str] = None,
                 nl_before: bool = True):
        self.prompt = prompt
        self.prefill = prefill
        self.banner = banner
        self.nl_before = nl_before

    def __enter__(self):
        if self.banner:
            # display banner
            click.echo(self.banner)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # ignore
        pass

    def __iter__(self):
        return self

    def __next__(self) -> str:

        def _hook():
            readline.insert_text(self.prefill)
            readline.redisplay()

        if self.prefill:
            # use readline to prefill the answers
            readline.set_pre_input_hook(_hook)

        if self.nl_before:
            # append newline before prompt
            click.echo()

        # prompt for input
        value = click.prompt(self.prompt, type=str, prompt_suffix="> ")

        # clear the input hook for next prompt
        readline.set_pre_input_hook()

        # reuse the value for next prefill
        self.prefill = value

        return value
