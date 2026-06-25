import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}


def load_qa_config(path: Path) -> dict[str, Any]:
    """Load a YAML QA notebook configuration file."""
    with open(path, "r") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def dry_run_qa_config(config: dict[str, Any]) -> Path:
    """Validate a QA notebook config and return the planned output path."""
    validated = _validate_qa_config(config)
    return _output_notebook(validated)


def generate_qa_notebook(config: dict[str, Any]) -> Path:
    """Generate a QA notebook from CONFIG and return the notebook path."""
    validated = _validate_qa_config(config)
    output_notebook = _output_notebook(validated)
    output_notebook.parent.mkdir(parents=True, exist_ok=True)
    notebook = _build_notebook(validated)
    with open(output_notebook, "w") as handle:
        json.dump(notebook, handle, indent=2)
        handle.write("\n")
    return output_notebook


def _validate_qa_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("QA config must be a YAML mapping.")
    if not config:
        raise ValueError("QA config is empty. Set input_file and output_notebook or output_dir.")

    title = config.get("title", "QA Notebook")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string when provided.")

    subtitle = config.get("subtitle", "Basic dataset characterization")
    if subtitle is not None and not isinstance(subtitle, str):
        raise ValueError("subtitle must be a string or null.")

    last_verified_run = config.get("last_verified_run")
    if last_verified_run is not None and not isinstance(last_verified_run, str):
        raise ValueError("last_verified_run must be a string when provided.")

    input_file = config.get("input_file")
    if input_file is None:
        raise ValueError("QA config requires input_file with a local curated catalog path.")
    if not isinstance(input_file, str | Path) or not str(input_file).strip():
        raise ValueError("input_file must be a non-empty path string.")

    input_format = str(config.get("input_format", "parquet")).lower()
    if input_format not in {"parquet", "csv", "hats"}:
        raise ValueError("input_format must be one of: parquet, csv, hats.")
    if "include_absolute_input_path" in config and not isinstance(
        config["include_absolute_input_path"], bool
    ):
        raise ValueError("include_absolute_input_path must be true or false.")

    output_notebook = config.get("output_notebook")
    output_dir = config.get("output_dir")
    if output_notebook is not None and output_dir is not None:
        raise ValueError("QA config accepts either output_notebook or output_dir, not both.")
    if output_notebook is not None and (
        not isinstance(output_notebook, str | Path) or not str(output_notebook).strip()
    ):
        raise ValueError("output_notebook must be a non-empty path string when provided.")
    if output_dir is not None and (not isinstance(output_dir, str | Path) or not str(output_dir).strip()):
        raise ValueError("output_dir must be a non-empty path string when provided.")

    header_images = config.get("header_images", [])
    if header_images is None:
        header_images = []
    if not isinstance(header_images, list | tuple):
        raise ValueError("header_images must be a list with at most two image paths or mappings.")
    if len(header_images) > 2:
        raise ValueError("header_images accepts at most two images.")
    for index, image in enumerate(header_images):
        _validate_header_image(image, index)

    for key in ("summary", "acknowledgments"):
        values = config.get(key, [])
        if values is None:
            continue
        if not isinstance(values, list | tuple) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"{key} must be a list of strings when provided.")

    plots = config.get("plots", {})
    if plots is None:
        plots = {}
    if not isinstance(plots, dict):
        raise ValueError("plots must be a mapping when provided.")
    for key in ("spatial", "redshift", "quality", "redshift_error"):
        value = plots.get(key)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"plots.{key} must be a mapping when provided.")
    spatial = plots.get("spatial")
    if spatial is not None:
        _validate_footprints(_configured_footprints(spatial))

    return config


def _validate_header_image(image: Any, index: int) -> None:
    if isinstance(image, str):
        if not image.strip():
            raise ValueError(f"header_images[{index}] must not be empty.")
        return
    if not isinstance(image, dict):
        raise ValueError(f"header_images[{index}] must be a path string or mapping.")
    src = image.get("src", image.get("path", image.get("url")))
    if not isinstance(src, str) or not src.strip():
        raise ValueError(f"header_images[{index}] requires src, path, or url.")
    if "width" in image and (not isinstance(image["width"], int) or image["width"] <= 0):
        raise ValueError(f"header_images[{index}].width must be a positive integer.")


def _configured_footprints(spatial: dict[str, Any]) -> list[dict[str, Any]]:
    footprints = spatial.get("footprints")
    if footprints is None and spatial.get("footprint_path"):
        footprints = [
            {
                "path": spatial["footprint_path"],
                "label": spatial.get("footprint_label", "Footprint"),
            }
        ]
    if footprints is None:
        return []
    if not isinstance(footprints, list | tuple):
        raise ValueError("plots.spatial.footprints must be a list of footprint mappings.")

    normalized = []
    for index, footprint in enumerate(footprints):
        if isinstance(footprint, str):
            footprint = {"path": footprint}
        if not isinstance(footprint, dict):
            raise ValueError(f"plots.spatial.footprints[{index}] must be a path string or mapping.")
        path = footprint.get("path")
        if not isinstance(path, str | Path) or not str(path).strip():
            raise ValueError(f"plots.spatial.footprints[{index}] requires a non-empty path.")
        normalized.append(footprint)
    return normalized


def _validate_footprints(footprints: list[dict[str, Any]]) -> None:
    for index, footprint in enumerate(footprints):
        path = Path(footprint["path"])
        if not path.exists():
            raise ValueError(_footprint_error(path, f"file does not exist for footprints[{index}]."))
        columns = set(pd.read_csv(path, nrows=0).columns)
        footprint_format = _footprint_format(columns, path=path)
        configured_format = footprint.get("format")
        if configured_format is not None and configured_format != footprint_format:
            raise ValueError(
                _footprint_error(
                    path,
                    f"configured format {configured_format!r} does not match detected format "
                    f"{footprint_format!r}.",
                )
            )


def _footprint_format(columns: set[str], path: Path | None = None) -> str:
    region_columns = {"region_id", "ring_type", "vertex_id", "ra_deg", "dec_deg"}
    limit_columns = {"ra_center", "dec_limit"}
    if region_columns.issubset(columns):
        return "region_vertices"
    if limit_columns.issubset(columns):
        return "declination_limit"
    raise ValueError(_footprint_error(path, f"unsupported columns: {', '.join(sorted(columns))}."))


def _footprint_error(path: Path | None, detail: str) -> str:
    prefix = f"Invalid footprint CSV {path}: " if path else "Invalid footprint CSV: "
    return (
        prefix + detail + " Expected either columns region_id, ring_type, vertex_id, ra_deg, dec_deg "
        "for one or more polygon/ring curves, or columns ra_center, dec_limit for a single "
        "RA-sampled declination-limit curve."
    )


def _output_notebook(config: dict[str, Any]) -> Path:
    if config.get("output_notebook"):
        return Path(config["output_notebook"])
    output_dir = Path(config.get("output_dir", "reports/qa"))
    return output_dir / "qa_notebook.ipynb"


def _build_notebook(config: dict[str, Any]) -> dict[str, Any]:
    cells = [
        _markdown_cell(_header_source(config)),
        _markdown_cell(
            "--- \n\n"
            "This notebook contains simple plots and statistics for a quick characterization "
            "of the data product.\n\n"
            "--- \n"
        ),
        _markdown_cell("## Summary"),
        _markdown_cell(_summary_source(config)),
    ]

    acknowledgments = config.get("acknowledgments") or []
    if acknowledgments:
        cells.extend(
            [
                _markdown_cell("### Acknowledgments"),
                _markdown_cell("\n\n".join(f"*{item}*" for item in acknowledgments)),
            ]
        )

    cells.extend(
        [
            _markdown_cell("## Imports and settings"),
            _markdown_cell("Imports."),
            _code_cell(_imports_source(config)),
            _markdown_cell("Pandas configuration."),
            _code_cell("pd.set_option('display.max_rows', 10)\npd.set_option('display.max_columns', 40)"),
        ]
    )

    cells.extend(_local_data_cells(config))

    cells.extend(
        [
            _markdown_cell("First rows."),
            _code_cell("df.head()"),
            _markdown_cell("Total number of rows."),
            _code_cell("len(df)"),
            _markdown_cell("Total number of columns."),
            _code_cell("len(df.columns.to_list())"),
            _markdown_cell("## Basic Statistics "),
            _code_cell("df.describe()"),
        ]
    )
    cells.extend(_plot_cells(config))

    return {
        "cells": cells,
        "metadata": config.get("metadata", NOTEBOOK_METADATA),
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _header_source(config: dict[str, Any]) -> str:
    images = "\n".join(_header_image_html(image) for image in config.get("header_images", []) or [])
    title = config.get("title", "QA Notebook").strip()
    subtitle = config.get("subtitle", "Basic dataset characterization")
    last_verified_run = _last_verified_run(config)

    parts = []
    if images:
        parts.append(images)
        parts.append("")
    parts.append(f"# {title}")
    if subtitle:
        parts.append(f"<font size=4> {subtitle} </font>")
    parts.append("")
    parts.append(f"Last verified run: **{last_verified_run}**")
    return "\n".join(parts)


def _last_verified_run(config: dict[str, Any]) -> str:
    return config.get("last_verified_run") or date.today().isoformat()


def _header_image_html(image: str | dict[str, Any]) -> str:
    if isinstance(image, str):
        src = image.strip()
        image_config: dict[str, Any] = {}
    else:
        image_src = image.get("src", image.get("path", image.get("url")))
        if not isinstance(image_src, str):  # pragma: no cover - guarded by config validation
            raise ValueError("header image requires src, path, or url.")
        src = image_src.strip()
        image_config = image

    align = image_config.get("align", "left")
    width = image_config.get("width")
    style = image_config.get("style")
    attrs = [f'align="{align}"', f'src = "{src}"']
    if width:
        attrs.append(f"width={width}")
    if style:
        attrs.append(f'style="{style}"')
    return "<img " + " ".join(attrs) + ">"


def _summary_source(config: dict[str, Any]) -> str:
    summary = config.get("summary") or []
    return "\n".join(f"* {item}" for item in summary) if summary else "_No summary metadata configured._"


def _imports_source(config: dict[str, Any]) -> str:
    imports = [
        "# General",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from matplotlib.colors import LogNorm",
        "import seaborn as sns",
        "from IPython.display import display, Markdown",
        "",
    ]
    if str(config.get("input_format", "parquet")).lower() == "hats":
        imports.extend(["# HATS", "import lsdb"])
    return "\n".join(imports)


def _local_data_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    input_file = _notebook_path(config, config["input_file"])
    input_format = str(config.get("input_format", "parquet")).lower()
    if input_format == "parquet":
        read_source = f"df = pd.read_parquet({input_file!r})"
    elif input_format == "csv":
        read_source = f"df = pd.read_csv({input_file!r})"
    elif input_format == "hats":
        read_source = f"catalog = lsdb.open_catalog({input_file!r})\ndf = catalog.compute()"
    else:  # pragma: no cover - guarded by config validation
        raise ValueError("input_format must be one of: parquet, csv, hats.")
    return [
        _markdown_cell("## Basic product information"),
        _markdown_cell("Retrieve data from local curated catalog."),
        _code_cell(read_source),
    ]


def _notebook_path(config: dict[str, Any], path: str | Path) -> str:
    input_file = Path(path)
    if config.get("include_absolute_input_path", True):
        return str(input_file.resolve())
    return input_file.name


def _plot_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    plots = config.get("plots") or {}
    cells: list[dict[str, Any]] = []

    spatial = plots.get("spatial")
    if spatial:
        cells.extend(
            [
                _markdown_cell("## Plots\n### Spatial distribution\n"),
                _code_cell(_spatial_plot_source(config, spatial)),
            ]
        )
    else:
        cells.append(_markdown_cell("## Plots"))

    redshift = plots.get("redshift")
    if redshift:
        cells.extend(
            [
                _markdown_cell("### Redshift distribution\nGlobal redshift distribution.\n"),
                _code_cell(_hist_source(redshift, default_label="redshift")),
            ]
        )

    quality = plots.get("quality")
    if quality:
        cells.extend(
            [
                _markdown_cell("### Quality Flags\n\n" + quality.get("description", "")),
                _code_cell(_quality_source(quality)),
            ]
        )

    redshift_error = plots.get("redshift_error")
    if redshift_error:
        cells.extend(
            [
                _markdown_cell("### Redshift error distribution\n\nGlobal redshift error distribution."),
                _code_cell(_hist_source(redshift_error, default_label="redshift error")),
            ]
        )

    return cells


def _spatial_plot_source(config: dict[str, Any], spatial: dict[str, Any]) -> str:
    title = spatial.get("title", f"{config.get('title', 'Catalog')} - Spatial distribution")
    ra_column = spatial.get("ra_column", "ra")
    dec_column = spatial.get("dec_column", "dec")
    footprint_code = _footprint_plot_source(config, _configured_footprints(spatial))

    return (
        "def ra_to_mollweide_x(ra_deg):\n"
        '    """\n'
        "    Convert RA in degrees to Mollweide x coordinate using the astronomical convention.\n"
        "    RA = 0 deg is at the center and RA increases to the left.\n"
        '    """\n'
        "    ra_centered = ((np.asarray(ra_deg) + 180) % 360) - 180\n"
        "    return -np.deg2rad(ra_centered)\n\n"
        "def plot_wrapped_curve(ax, ra_deg, dec_deg, **plot_kwargs):\n"
        '    """Plot a curve in Mollweide coordinates, split at RA wrap jumps."""\n'
        "    x = ra_to_mollweide_x(ra_deg)\n"
        "    y = np.deg2rad(dec_deg)\n\n"
        "    jump_idx = np.where(np.abs(np.diff(x)) > np.pi)[0]\n"
        "    start = 0\n"
        "    first_segment = True\n\n"
        "    for j in jump_idx:\n"
        "        end = j + 1\n"
        "        if end - start > 1:\n"
        "            if first_segment:\n"
        "                ax.plot(x[start:end], y[start:end], **plot_kwargs)\n"
        "                first_segment = False\n"
        "            else:\n"
        "                ax.plot(\n"
        "                    x[start:end],\n"
        "                    y[start:end],\n"
        '                    **{k: v for k, v in plot_kwargs.items() if k != "label"},\n'
        "                )\n"
        "        start = end\n\n"
        "    if len(x) - start > 1:\n"
        "        if first_segment:\n"
        "            ax.plot(x[start:], y[start:], **plot_kwargs)\n"
        "        else:\n"
        "            ax.plot(\n"
        "                x[start:],\n"
        "                y[start:],\n"
        '                **{k: v for k, v in plot_kwargs.items() if k != "label"},\n'
        "            )\n\n"
        f"x_points = ra_to_mollweide_x(df[{ra_column!r}].values)\n"
        f"y_points = np.deg2rad(df[{dec_column!r}].values)\n"
        "valid = np.isfinite(x_points) & np.isfinite(y_points)\n"
        "x_points = x_points[valid]\n"
        "y_points = y_points[valid]\n\n"
        "xbins = np.linspace(-np.pi, np.pi, 180)\n"
        "ybins = np.linspace(-np.pi / 2, np.pi / 2, 90)\n"
        "H, xedges, yedges = np.histogram2d(x_points, y_points, bins=[xbins, ybins])\n"
        "H = np.ma.masked_where(H == 0, H)\n\n"
        "fig = plt.figure(figsize=(16, 8))\n"
        'ax = fig.add_subplot(111, projection="mollweide")\n'
        "mesh = ax.pcolormesh(\n"
        '    xedges, yedges, H.T, norm=LogNorm(), shading="auto", cmap="viridis", zorder=1\n'
        ")\n"
        "cbar = fig.colorbar(mesh, ax=ax, pad=0.05)\n"
        'cbar.set_label("Number of objects")\n'
        "ax.grid(False)\n\n"
        "dec_grid = np.deg2rad(np.linspace(-90, 90, 500))\n"
        "for grid_ra_deg in np.arange(-150, 181, 30):\n"
        "    ax.plot(\n"
        "        np.full_like(dec_grid, np.deg2rad(grid_ra_deg)),\n"
        "        dec_grid,\n"
        '        color="gray",\n'
        "        linewidth=0.6,\n"
        "        alpha=0.5,\n"
        "        zorder=2,\n"
        "    )\n\n"
        "ra_grid = np.deg2rad(np.linspace(-180, 180, 800))\n"
        "for grid_dec_deg in np.arange(-75, 76, 15):\n"
        "    ax.plot(\n"
        "        ra_grid,\n"
        "        np.full_like(ra_grid, np.deg2rad(grid_dec_deg)),\n"
        '        color="gray",\n'
        "        linewidth=0.6,\n"
        "        alpha=0.5,\n"
        "        zorder=2,\n"
        "    )\n"
        f"{footprint_code}"
        "\n"
        "tick_degs = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])\n"
        "tick_labels = [\n"
        '    "150 deg", "120 deg", "90 deg", "60 deg", "30 deg", "0 deg",\n'
        '    "330 deg", "300 deg", "270 deg", "240 deg", "210 deg",\n'
        "]\n"
        "ax.set_xticks(np.deg2rad(tick_degs))\n"
        "ax.set_xticklabels(tick_labels)\n\n"
        f"ax.set_xlabel({ra_column!r})\n"
        f"ax.set_ylabel({dec_column!r})\n"
        f"ax.set_title({title!r})\n"
        "if ax.get_legend_handles_labels()[0]:\n"
        '    ax.legend(loc="upper right")\n'
        "plt.show()"
    )


def _footprint_plot_source(config: dict[str, Any], footprints: list[dict[str, Any]]) -> str:
    if not footprints:
        return ""

    parts = ["\n"]
    for index, footprint in enumerate(footprints):
        path = Path(footprint["path"])
        columns = set(pd.read_csv(path, nrows=0).columns)
        footprint_format = footprint.get("format") or _footprint_format(columns, path=path)
        label = footprint.get("label", f"Footprint {index + 1}")
        color = footprint.get("color", _default_footprint_color(index))
        linewidth = footprint.get("linewidth", 1)
        variable = f"footprint_{index}"
        first_variable = f"first_curve_{index}"

        notebook_path = _notebook_path(config, path)
        parts.append(f"{variable} = pd.read_csv({notebook_path!r})\n")
        if footprint_format == "region_vertices":
            parts.append(
                f"{first_variable} = True\n"
                f'for _, footprint_region in {variable}.groupby("region_id"):\n'
                '    ext = footprint_region[footprint_region["ring_type"] == "exterior"].sort_values(\n'
                '        "vertex_id"\n'
                "    )\n"
                "    plot_wrapped_curve(\n"
                "        ax,\n"
                '        ext["ra_deg"].values,\n'
                '        ext["dec_deg"].values,\n'
                f"        linewidth={linewidth!r},\n"
                f"        color={color!r},\n"
                "        zorder=3,\n"
                f'        label={label!r} if {first_variable} else "_nolegend_",\n'
                "    )\n"
                f"    {first_variable} = False\n"
            )
        elif footprint_format == "declination_limit":
            parts.append(
                f'{variable} = {variable}.sort_values("ra_center")\n'
                "plot_wrapped_curve(\n"
                "    ax,\n"
                f'    {variable}["ra_center"].values,\n'
                f'    {variable}["dec_limit"].values,\n'
                f"    linewidth={linewidth!r},\n"
                f"    color={color!r},\n"
                "    zorder=3,\n"
                f"    label={label!r},\n"
                ")\n"
            )
    return "".join(parts)


def _default_footprint_color(index: int) -> str:
    colors = ["red", "orange", "cyan", "magenta", "white"]
    return colors[index % len(colors)]


def _hist_source(plot: dict[str, Any], default_label: str) -> str:
    column = plot.get("column", default_label)
    value_range = plot.get("range")
    bins = int(plot.get("bins", 50))
    if value_range:
        min_value, max_value = value_range
        data_expr = f"df[(df[{column!r}] >= {min_value}) & (df[{column!r}] <= {max_value})]"
        xlim = f"plt.xlim({min_value}, {max_value})\n"
    else:
        data_expr = "df"
        xlim = ""
    return (
        "plt.figure(figsize=(8, 6))\n"
        "sns.histplot(\n"
        f"    data={data_expr},\n"
        f"    x={column!r},\n"
        "    kde=True,\n"
        f"    bins={bins},\n"
        ")\n"
        f"plt.xlabel({column!r})\n"
        'plt.ylabel("Count")\n'
        f"{xlim}"
        "plt.tight_layout()\n"
        "plt.show()"
    )


def _quality_source(quality: dict[str, Any]) -> str:
    column = quality.get("column", "quality")
    return (
        "plt.figure(figsize=(8, 6))\n"
        "sns.countplot(\n"
        "    data=df,\n"
        f"    x={column!r},\n"
        f"    order=sorted(df[{column!r}].dropna().unique()),\n"
        ")\n\n"
        f"plt.xlabel({column!r})\n"
        'plt.ylabel("Count")\n'
        "plt.xticks(rotation=0)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )


def _markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(source),
    }


def _code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(source),
    }


def _source_lines(source: str) -> list[str]:
    lines = source.splitlines(keepends=True)
    return lines or [""]
