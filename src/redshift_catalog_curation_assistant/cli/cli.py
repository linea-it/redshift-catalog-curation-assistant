import ast
from pathlib import Path

import click

from .. import __version__
from .. import fits as rc_fits
from .. import inspect as rc_inspect


@click.group()
def cli():
    """Redshift Catalog Curation Assistant CLI."""


@cli.command()
@click.argument("config", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--path",
    "input_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    help="Catalog file or partitioned Parquet directory to inspect.",
)
@click.option(
    "--survey-name",
    help="Survey name to use for report output. Defaults to the input filename stem.",
)
@click.option(
    "--fits-hdu",
    default=1,
    show_default=True,
    help="FITS HDU to read when inspecting FITS files.",
)
@click.option(
    "--unique-limit",
    default=10,
    show_default=True,
    help="Maximum unique categorical values to report.",
)
@click.option(
    "--column-name",
    "column_names",
    multiple=True,
    help="Column name for a headerless file. Repeat once per column.",
)
@click.option(
    "--column-names",
    "column_names_list",
    help="Column names for a headerless file as a Python-style list or comma-separated string.",
)
@click.option(
    "--column-selection",
    "column_selection",
    multiple=True,
    help="Column to include in the inspection report. Repeat once per selected column.",
)
@click.option(
    "--column-selection-list",
    "column_selection_list",
    help="Columns to include in the report as a Python-style list or comma-separated string.",
)
@click.option(
    "--dask-threshold-mb",
    default=100.0,
    show_default=True,
    help="Use Dask for supported non-FITS files at or above this size. Use 0 to disable.",
)
@click.option(
    "--dask-cluster",
    default="local",
    show_default=True,
    help="Dask cluster config: 'local' or a JSON/Python dict matching the executor schema.",
)
def inspect(
    config,
    input_path,
    survey_name,
    fits_hdu,
    unique_limit,
    column_names,
    column_names_list,
    column_selection,
    column_selection_list,
    dask_threshold_mb,
    dask_cluster,
):
    """Run inspection using CONFIG YAML or default settings from --path."""
    if bool(config) == bool(input_path):
        raise click.UsageError("Provide exactly one of CONFIG or --path.")

    if config:
        cfg = rc_inspect.load_config(Path(config))
        cfg.setdefault("dask_threshold_mb", dask_threshold_mb)
        cfg.setdefault("dask_cluster", _parse_dask_cluster_option(dask_cluster))
    else:
        path = Path(input_path)
        cfg = {
            "input_file": str(path),
            "survey_name": survey_name or path.stem,
            "fits_hdu": fits_hdu,
            "unique_limit": unique_limit,
            "dask_threshold_mb": dask_threshold_mb,
            "dask_cluster": _parse_dask_cluster_option(dask_cluster),
        }
        parsed_column_names = _parse_column_names_options(column_names, column_names_list)
        if parsed_column_names:
            cfg["column_names"] = parsed_column_names
        parsed_column_selection = _parse_column_selection_options(column_selection, column_selection_list)
        if parsed_column_selection:
            cfg["column_selection"] = parsed_column_selection

    try:
        outdir = rc_inspect.run_inspect_config(cfg)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Wrote inspection reports to {outdir}")


def _parse_column_names_options(column_names: tuple[str, ...], column_names_list: str | None) -> list[str]:
    return _parse_repeated_or_list_option(
        values=column_names,
        values_list=column_names_list,
        repeated_option="--column-name",
        list_option="--column-names",
        example='["RA", "Dec", "z"]',
    )


def _parse_column_selection_options(
    column_selection: tuple[str, ...], column_selection_list: str | None
) -> list[str]:
    return _parse_repeated_or_list_option(
        values=column_selection,
        values_list=column_selection_list,
        repeated_option="--column-selection",
        list_option="--column-selection-list",
        example='["RA", "Dec", "z"]',
    )


def _parse_repeated_or_list_option(
    values: tuple[str, ...],
    values_list: str | None,
    repeated_option: str,
    list_option: str,
    example: str,
) -> list[str]:
    if values and values_list:
        raise click.UsageError(f"Use either {repeated_option} repeatedly or {list_option} once, not both.")
    if values:
        return [value.strip() for value in values if value.strip()]
    if not values_list:
        return []

    try:
        parsed = ast.literal_eval(values_list)
    except (SyntaxError, ValueError):
        parsed = [value.strip() for value in values_list.split(",")]

    if not isinstance(parsed, list | tuple) or not all(isinstance(value, str) for value in parsed):
        raise click.BadParameter(f"Expected a list of strings, e.g. '{example}'.")
    return [value.strip() for value in parsed if value.strip()]


def _parse_dask_cluster_option(value: str) -> dict:
    value = value.strip()
    if value == "local":
        return {"name": value}
    if value == "slurm":
        raise click.BadParameter("SLURM Dask clusters require a dict literal with args for --dask-cluster.")

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise click.BadParameter("Expected 'local' or a dict literal for --dask-cluster.") from exc

    if not isinstance(parsed, dict):
        raise click.BadParameter("Expected 'local' or a dict literal for --dask-cluster.")
    return parsed


@cli.command("inspect-fits")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--max-columns",
    default=12,
    show_default=True,
    help="Maximum number of table columns to preview.",
)
def inspect_fits(path, max_columns):
    """Describe HDUs in a FITS file."""
    try:
        summaries = rc_fits.describe_fits(Path(path), max_columns=max_columns)
    except rc_fits.LargeCompressedFitsError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(rc_fits.format_fits_description(summaries))


def _planned_command(command: str, config: str) -> None:
    click.echo(f"{command} is planned for a later phase. Config received: {config}")


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
def curate(config):
    """Run local curation using CONFIG YAML."""
    _planned_command("curate", config)


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
def qa(config):
    """Run quality-assurance checks using CONFIG YAML."""
    _planned_command("qa", config)


@cli.command("validate-flags")
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
def validate_flags(config):
    """Validate flag translation rules using CONFIG YAML."""
    _planned_command("validate-flags", config)


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
def run(config):
    """Run the configured local workflow."""
    _planned_command("run", config)


@cli.command()
def version():
    """Show package version."""
    click.echo(__version__)


def main():
    """Run the command-line interface."""
    cli()


if __name__ == "__main__":
    main()
