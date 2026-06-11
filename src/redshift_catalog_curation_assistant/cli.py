import sys
from pathlib import Path
import click

from . import io as rc_io
from . import inspect as rc_inspect


@click.group()
def cli():
    """Redshift Catalog Curation Assistant CLI."""


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
def inspect(config):
    """Run inspection using CONFIG YAML."""
    rc_inspect.run_inspect(Path(config))


@cli.command()
@click.option("--help-only", is_flag=True, default=False, help="Show help and exit")
def version(help_only):
    """Show package version."""
    try:
        from ._version import __version__

        click.echo(__version__)
    except Exception:
        click.echo("unknown")


def main():
    cli()


if __name__ == "__main__":
    main()
