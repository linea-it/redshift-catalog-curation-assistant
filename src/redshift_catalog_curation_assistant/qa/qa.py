import json
import os
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..executor import dask_cluster_config

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
DEFAULT_LARGE_INPUT_THRESHOLD_MB = 100


def load_qa_config(path: Path) -> dict[str, Any]:
    """Load a YAML QA notebook configuration file."""
    with open(path, "r") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def dry_run_qa_config(config: dict[str, Any]) -> Path:
    """Validate a QA notebook config and return the planned output path."""
    validated = _validate_qa_config(config)
    return _output_notebook(validated)


def run_qa_config(config: dict[str, Any]) -> dict[str, Path]:
    """Generate configured QA artifacts and return their paths."""
    output_notebook = generate_qa_notebook(config)
    artifacts = {"notebook": output_notebook}
    if config.get("generate_html", False):
        artifacts["html"] = execute_qa_notebook_to_html(config, output_notebook)
    return artifacts


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


def execute_qa_notebook_to_html(config: dict[str, Any], notebook_path: Path | None = None) -> Path:
    """Execute a generated QA notebook in memory and export the executed HTML."""
    validated = _validate_qa_config(config)
    _validate_notebook_runtime_paths(validated)

    output_notebook = Path(notebook_path) if notebook_path is not None else _output_notebook(validated)
    if not output_notebook.exists():
        raise ValueError(f"Cannot generate QA HTML because notebook does not exist: {output_notebook}")

    output_html = _output_html(validated)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    try:
        import nbformat
        from jupyter_client import AsyncKernelManager
        from nbclient import NotebookClient
        from nbconvert import HTMLExporter
    except ImportError as exc:  # pragma: no cover - dependency metadata should prevent this
        raise ValueError(
            "QA HTML generation requires nbformat, nbclient, and nbconvert. "
            "Install the project with QA HTML dependencies enabled."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="qa-ipython-") as ipython_dir:
        execution_env = os.environ.copy()
        execution_env["IPYTHONDIR"] = ipython_dir
        execution_env["JPY_PARENT_PID"] = "1"
        execution_env.pop("JPY_INTERRUPT_EVENT", None)
        execution_env.pop("IPY_INTERRUPT_EVENT", None)
        resources = {"metadata": {"path": str(output_notebook.parent.resolve())}}
        kernel_name = validated.get("html_kernel_name", "python3")
        timeout = validated.get("html_execution_timeout", 600)

        notebook = nbformat.read(output_notebook, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name=kernel_name,
            resources=resources,
            shutdown_kernel="immediate",
        )
        client.km = AsyncKernelManager(kernel_name=kernel_name)
        client.km.transport = "ipc"
        try:
            client.execute(
                env=execution_env,
                independent=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            notebook = nbformat.read(output_notebook, as_version=4)
            client = NotebookClient(
                notebook,
                timeout=timeout,
                kernel_name=kernel_name,
                resources=resources,
                shutdown_kernel="immediate",
            )
            client.execute(
                env=execution_env,
                independent=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    body, _ = HTMLExporter().from_notebook_node(notebook)
    output_html.write_text(body, encoding="utf-8")
    return output_html


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

    pzs_configured = any(
        config.get(key) is not None for key in ("pzs_prod_name", "pzs_token_path", "pzs_host")
    )
    input_file = config.get("input_file")
    if input_file is None and not pzs_configured:
        raise ValueError("QA config requires input_file with a local curated catalog path.")
    if input_file is not None and (not isinstance(input_file, str | Path) or not str(input_file).strip()):
        raise ValueError("input_file must be a non-empty path string.")

    input_format = config.get("input_format")
    if input_format is not None and str(input_format).lower() not in {"parquet", "csv", "hats"}:
        raise ValueError("input_format must be one of: parquet, csv, hats.")
    if pzs_configured and input_file is None and input_format is not None:
        raise ValueError("PZ Server input_format override requires an input_file override as well.")
    pzs_values = {key: config.get(key) for key in ("pzs_prod_name", "pzs_token_path", "pzs_host")}
    if any(value is not None for value in pzs_values.values()):
        for key, value in pzs_values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string when using PZ Server.")
        product_name = str(pzs_values["pzs_prod_name"])
        if Path(product_name).name != product_name or product_name in {".", ".."}:
            raise ValueError("pzs_prod_name must be a product ID, not a filesystem path.")
        download_dir = config.get("pzs_download_dir", "./downloaded_data")
        if not isinstance(download_dir, str | Path) or not str(download_dir).strip():
            raise ValueError("pzs_download_dir must be a non-empty path string when provided.")
        if "pzs_overwrite" in config and not isinstance(config["pzs_overwrite"], bool):
            raise ValueError("pzs_overwrite must be true or false.")
    if "include_absolute_input_path" in config and not isinstance(
        config["include_absolute_input_path"], bool
    ):
        raise ValueError("include_absolute_input_path must be true or false.")
    threshold = config.get("large_input_threshold_mb", DEFAULT_LARGE_INPUT_THRESHOLD_MB)
    if isinstance(threshold, bool) or not isinstance(threshold, int | float) or threshold <= 0:
        raise ValueError("large_input_threshold_mb must be a positive number.")
    if "force_compute" in config and not isinstance(config["force_compute"], bool):
        raise ValueError("force_compute must be true or false.")
    if (
        "dask_cluster" in config
        and config["dask_cluster"] is not None
        and not isinstance(config["dask_cluster"], dict)
    ):
        raise ValueError("dask_cluster must be a mapping when provided.")
    dask_cluster_config(config)
    if "generate_html" in config and not isinstance(config["generate_html"], bool):
        raise ValueError("generate_html must be true or false.")
    output_html = config.get("output_html")
    if output_html is not None and (not isinstance(output_html, str | Path) or not str(output_html).strip()):
        raise ValueError("output_html must be a non-empty path string when provided.")
    html_kernel_name = config.get("html_kernel_name")
    if html_kernel_name is not None and (
        not isinstance(html_kernel_name, str) or not html_kernel_name.strip()
    ):
        raise ValueError("html_kernel_name must be a non-empty string when provided.")
    html_execution_timeout = config.get("html_execution_timeout")
    if html_execution_timeout is not None and (
        not isinstance(html_execution_timeout, int | float) or html_execution_timeout <= 0
    ):
        raise ValueError("html_execution_timeout must be a positive number when provided.")

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
    quality = plots.get("quality")
    if quality is not None:
        _validate_label_rotation(quality, "plots.quality")
    categorical = plots.get("categorical", [])
    if categorical is None:
        categorical = []
    if not isinstance(categorical, list | tuple):
        raise ValueError("plots.categorical must be a list of mappings when provided.")
    for index, plot in enumerate(categorical):
        if not isinstance(plot, dict):
            raise ValueError(f"plots.categorical[{index}] must be a mapping.")
        column = plot.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ValueError(f"plots.categorical[{index}].column must be a non-empty string.")
        _validate_label_rotation(plot, f"plots.categorical[{index}]")
    spatial = plots.get("spatial")
    if spatial is not None:
        _validate_footprints(_configured_footprints(spatial))

    return config


def _validate_label_rotation(plot: dict[str, Any], path: str) -> None:
    rotation = plot.get("label_rotation")
    if rotation is not None and (isinstance(rotation, bool) or not isinstance(rotation, int | float)):
        raise ValueError(f"{path}.label_rotation must be a number when provided.")


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


def _output_html(config: dict[str, Any]) -> Path:
    if config.get("output_html"):
        return Path(config["output_html"])
    return _output_notebook(config).with_suffix(".html")


def _validate_notebook_runtime_paths(config: dict[str, Any]) -> None:
    input_file = Path(config["input_file"]) if config.get("input_file") else None
    if not _uses_pzserver(config) and input_file is not None and not input_file.exists():
        raise ValueError(f"Cannot generate QA HTML because input_file does not exist: {input_file}")

    plots = config.get("plots") or {}
    spatial = plots.get("spatial")
    if spatial:
        _validate_footprints(_configured_footprints(spatial))


def _input_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _qa_data_mode(config: dict[str, Any]) -> str:
    if _pzs_runtime_input(config):
        return "auto"
    if config.get("force_compute", False):
        return "forced_in_memory"
    input_size = _input_size_bytes(Path(config["input_file"]))
    threshold = float(config.get("large_input_threshold_mb", DEFAULT_LARGE_INPUT_THRESHOLD_MB))
    if input_size is not None and input_size > threshold * 1024 * 1024:
        return "lazy"
    return "in_memory"


def _data_mode_source(config: dict[str, Any]) -> str:
    mode = _qa_data_mode(config)
    if mode == "auto":
        return "**Data access mode:** resolved at runtime after PZ Server download"
    input_size = _input_size_bytes(Path(config["input_file"])) if config.get("input_file") else None
    size_label = "unknown" if input_size is None else f"{input_size / (1024 * 1024):.2f} MB"
    threshold = float(config.get("large_input_threshold_mb", DEFAULT_LARGE_INPUT_THRESHOLD_MB))
    labels = {
        "in_memory": "in-memory",
        "forced_in_memory": "forced in-memory",
        "lazy": "lazy partition aggregation",
    }
    source = (
        f"**Data access mode:** {labels[mode]}  \n"
        f"**Input size:** {size_label}  \n"
        f"**Lazy threshold:** {threshold:g} MB"
    )
    if mode == "lazy":
        source += f"  \n**Dask executor:** {dask_cluster_config(config)['name']}"
    return source


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
        ]
    )
    if _uses_pzserver(config):
        cells.extend(
            [
                _markdown_cell("PZ Server configuration."),
                _code_cell(_pzserver_configuration_source(config)),
            ]
        )
    cells.extend(
        [
            _markdown_cell("Pandas configuration."),
            _code_cell("pd.set_option('display.max_rows', 10)\npd.set_option('display.max_columns', 40)"),
        ]
    )

    cells.append(_markdown_cell(_data_mode_source(config)))
    cells.extend(_dask_setup_cells(config))
    cells.extend(_local_data_cells(config))
    cells.extend(_basic_information_cells(config))
    cells.extend(_data_quality_cells(config))
    cells.extend(_plot_cells(config))
    cells.extend(_dask_cleanup_cells(config))
    for index, cell in enumerate(cells):
        cell.setdefault("id", f"qa-cell-{index:03d}")

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
        "from IPython.display import display, HTML, Markdown",
        "",
    ]
    if _uses_pzserver(config):
        imports.extend(
            [
                "# PZ Server",
                "from pathlib import Path",
                "import shutil",
                "import zipfile",
                "from pzserver import PzServer",
                "",
            ]
        )
    plots = config.get("plots") or {}
    has_plotting = any(
        plots.get(key) for key in ("spatial", "redshift", "quality", "redshift_error", "categorical")
    )
    if has_plotting:
        imports.extend(["# Plotting", "import matplotlib.pyplot as plt"])
    if plots.get("spatial"):
        imports.append("from matplotlib.colors import LogNorm")
    if any(plots.get(key) for key in ("redshift", "quality", "redshift_error", "categorical")):
        imports.append("import seaborn as sns")
    if has_plotting:
        imports.append("")
    input_format = str(config.get("input_format", "parquet")).lower()
    if _qa_data_mode(config) in {"lazy", "auto"} and (
        input_format in {"parquet", "csv"} or _pzs_runtime_input(config)
    ):
        imports.extend(["# Lazy tabular access", "import dask.dataframe as dd", ""])
    if _qa_data_mode(config) in {"lazy", "auto"}:
        imports.extend(
            [
                "# Distributed execution",
                "import atexit",
                "from dask.distributed import Client",
                "from redshift_catalog_curation_assistant.executor import create_dask_cluster",
                "",
            ]
        )
    if input_format == "hats" or _pzs_runtime_input(config):
        imports.extend(["# HATS", "import lsdb"])
    if _qa_data_mode(config) in {"lazy", "auto"}:
        imports.extend(["import warnings", "", _lazy_helpers_source()])
    return "\n".join(imports)


def _qa_dask_logs_dir(config: dict[str, Any]) -> Path | None:
    cluster_config = dask_cluster_config(config)
    configured = cluster_config.get("logs_dir")
    if configured:
        return Path(configured)
    if str(cluster_config.get("name", "local")).lower() == "slurm":
        return _output_notebook(config).parent / "logs"
    return None


def _dask_setup_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    if _qa_data_mode(config) != "lazy":
        return []
    cluster_config = dask_cluster_config(config)
    logs_dir = _qa_dask_logs_dir(config)
    logs_value = _notebook_path(config, logs_dir) if logs_dir is not None else None
    source = (
        f"qa_cluster_config = {cluster_config!r}\n"
        f"qa_cluster = create_dask_cluster(qa_cluster_config, logs_dir={logs_value!r})\n"
        "qa_client = Client(qa_cluster)\n"
        "qa_cluster_closed = False\n\n"
        "def close_qa_dask_cluster():\n"
        "    global qa_cluster_closed\n"
        "    if not qa_cluster_closed:\n"
        "        qa_client.close()\n"
        "        qa_cluster.close()\n"
        "        qa_cluster_closed = True\n\n"
        "atexit.register(close_qa_dask_cluster)\n"
        "qa_client"
    )
    return [
        _markdown_cell("## Distributed execution"),
        _markdown_cell("Create the configured Dask cluster for lazy QA operations."),
        _code_cell(source),
    ]


def _dask_cleanup_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    if _qa_data_mode(config) == "auto":
        return [
            _markdown_cell("## Cleanup"),
            _markdown_cell("Close the QA Dask client and cluster when lazy execution was used."),
            _code_cell(
                "if globals().get('qa_cluster_closed') is False:\n"
                "    close_qa_dask_cluster()\n"
                "    atexit.unregister(close_qa_dask_cluster)"
            ),
        ]
    if _qa_data_mode(config) != "lazy":
        return []
    return [
        _markdown_cell("## Cleanup"),
        _markdown_cell("Close the QA Dask client and cluster."),
        _code_cell("close_qa_dask_cluster()\natexit.unregister(close_qa_dask_cluster)"),
    ]


def _local_data_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    input_file = _notebook_path(config, config["input_file"]) if config.get("input_file") else None
    input_format = str(config.get("input_format", "parquet")).lower()
    mode = _qa_data_mode(config)
    if _pzs_runtime_input(config):
        threshold = float(config.get("large_input_threshold_mb", DEFAULT_LARGE_INPUT_THRESHOLD_MB))
        force_compute = bool(config.get("force_compute", False))
        logs_dir = _qa_dask_logs_dir(config)
        logs_value = _notebook_path(config, logs_dir) if logs_dir is not None else None
        runtime_input_source = (
            "qa_input_file, qa_input_format = candidates[0]\n"
            if _pzs_auto_input(config)
            else f"qa_input_file = {input_file!r}\nqa_input_format = {input_format!r}\n"
        )
        read_source = runtime_input_source + (
            "def qa_path_size_bytes(path):\n"
            "    path = Path(path)\n"
            "    if path.is_file():\n"
            "        return path.stat().st_size\n"
            "    return sum(item.stat().st_size for item in path.rglob('*') if item.is_file())\n\n"
            f"qa_threshold_mb = {threshold!r}\n"
            f"qa_force_compute = {force_compute!r}\n"
            "qa_input_size_bytes = qa_path_size_bytes(qa_input_file)\n"
            "if qa_force_compute:\n"
            "    qa_access_mode = 'forced_in_memory'\n"
            "elif qa_input_size_bytes > qa_threshold_mb * 1024 * 1024:\n"
            "    qa_access_mode = 'lazy'\n"
            "else:\n"
            "    qa_access_mode = 'in_memory'\n"
            "qa_cluster_closed = True\n"
            "if qa_access_mode == 'lazy':\n"
            f"    qa_cluster_config = {dask_cluster_config(config)!r}\n"
            f"    qa_cluster = create_dask_cluster(qa_cluster_config, logs_dir={logs_value!r})\n"
            "    qa_client = Client(qa_cluster)\n"
            "    qa_cluster_closed = False\n\n"
            "    def close_qa_dask_cluster():\n"
            "        global qa_cluster_closed\n"
            "        if not qa_cluster_closed:\n"
            "            qa_client.close()\n"
            "            qa_cluster.close()\n"
            "            qa_cluster_closed = True\n\n"
            "    atexit.register(close_qa_dask_cluster)\n\n"
            "def qa_open_data(columns=None):\n"
            "    if qa_access_mode == 'lazy':\n"
            "        if qa_input_format == 'parquet':\n"
            "            return dd.read_parquet(qa_input_file, columns=columns)\n"
            "        if qa_input_format == 'csv':\n"
            "            return dd.read_csv(qa_input_file, usecols=columns)\n"
            "        return lsdb.open_catalog(qa_input_file, columns=columns)\n"
            "    if qa_input_format == 'parquet':\n"
            "        return pd.read_parquet(qa_input_file, columns=columns)\n"
            "    if qa_input_format == 'csv':\n"
            "        return pd.read_csv(qa_input_file, usecols=columns)\n"
            "    catalog = lsdb.open_catalog(qa_input_file, columns=columns)\n"
            "    return catalog.compute()\n\n"
            "qa_runtime_summary = pd.Series({\n"
            "    'input_file': str(qa_input_file),\n"
            "    'input_format': qa_input_format,\n"
            "    'input_size_mb': qa_input_size_bytes / (1024 * 1024),\n"
            "    'access_mode': qa_access_mode,\n"
            "})\n"
            "qa_runtime_summary['input_size_mb'] = f\"{qa_runtime_summary['input_size_mb']:.2f} MB\"\n"
            "if qa_access_mode == 'lazy':\n"
            f"    qa_runtime_summary['dask_executor'] = {dask_cluster_config(config)['name']!r}\n"
            "else:\n"
            "    qa_runtime_summary['dask_executor'] = 'not used'\n\n"
            "qa_data = qa_open_data()\n"
            "qa_input_file, qa_input_format, qa_access_mode"
        )
    elif mode == "lazy" and input_format == "parquet":
        read_source = f"df = dd.read_parquet({input_file!r})"
    elif mode == "lazy" and input_format == "csv":
        read_source = f"df = dd.read_csv({input_file!r})"
    elif mode == "lazy" and input_format == "hats":
        read_source = f"catalog = lsdb.open_catalog({input_file!r})"
    elif input_format == "parquet":
        read_source = f"df = pd.read_parquet({input_file!r})"
    elif input_format == "csv":
        read_source = f"df = pd.read_csv({input_file!r})"
    elif input_format == "hats":
        read_source = f"catalog = lsdb.open_catalog({input_file!r})\ndf = catalog.compute()"
    else:  # pragma: no cover - guarded by config validation
        raise ValueError("input_format must be one of: parquet, csv, hats.")
    if mode == "forced_in_memory":
        read_source = (
            "import warnings\n"
            'warnings.warn("force_compute=True: loading the complete QA input into memory.")\n' + read_source
        )
    cells = [_markdown_cell("## Basic product information")]
    if _uses_pzserver(config):
        cells.extend(
            [
                _markdown_cell("Download and unzip the configured PZ Server product."),
                _code_cell(_pzserver_download_source(config)),
            ]
        )
    cells.extend(
        [
            _markdown_cell("Open the curated catalog."),
            _code_cell(read_source),
        ]
    )
    return cells


def _uses_pzserver(config: dict[str, Any]) -> bool:
    return config.get("pzs_prod_name") is not None


def _pzs_auto_input(config: dict[str, Any]) -> bool:
    return _uses_pzserver(config) and not config.get("input_file") and not config.get("input_format")


def _pzs_runtime_input(config: dict[str, Any]) -> bool:
    if not _uses_pzserver(config):
        return False
    if _pzs_auto_input(config):
        return True
    input_file = config.get("input_file")
    return input_file is not None and not Path(input_file).exists()


def _pzserver_configuration_source(config: dict[str, Any]) -> str:
    token_path = _notebook_path(config, config["pzs_token_path"])
    return (
        f"pzs_token_path = Path({token_path!r})\n"
        "token = pzs_token_path.read_text(encoding='utf-8').strip()\n"
        f"pz_server = PzServer(token=token, host={config['pzs_host']!r})"
    )


def _pzserver_download_source(config: dict[str, Any]) -> str:
    download_dir = _notebook_path(config, config.get("pzs_download_dir", "./downloaded_data"))
    return (
        f"prod_name = {config['pzs_prod_name']!r}\n"
        f"download_root = Path({download_dir!r})\n"
        "download_path = download_root / prod_name\n"
        f"pzs_overwrite = {config.get('pzs_overwrite', False)!r}\n"
        "if pzs_overwrite and download_path.exists():\n"
        "    shutil.rmtree(download_path)\n"
        "download_path.mkdir(parents=True, exist_ok=True)\n"
        "pz_server.download_product(product_id=prod_name, save_in=download_path)\n\n"
        "downloaded_archives = [\n"
        "    path for path in download_path.iterdir()\n"
        "    if path.is_file() and zipfile.is_zipfile(path)\n"
        "]\n"
        "if not downloaded_archives:\n"
        "    raise FileNotFoundError(\n"
        "        f'PZ Server product {prod_name!r} did not produce a ZIP archive in {download_path}'\n"
        "    )\n"
        "archive_path = max(downloaded_archives, key=lambda path: path.stat().st_mtime)\n"
        "with zipfile.ZipFile(archive_path) as archive:\n"
        "    archive_members = [\n"
        "        download_path / name for name in archive.namelist() if not name.endswith('/')\n"
        "    ]\n"
        "    download_root_resolved = download_path.resolve()\n"
        "    if any(download_root_resolved not in member.resolve().parents for member in archive_members):\n"
        "        raise ValueError(f'Unsafe path in PZ Server ZIP archive: {archive_path}')\n"
        "    archive.extractall(download_path)\n"
        + (_pzserver_detection_source() if _pzs_auto_input(config) else "")
    )


def _pzserver_detection_source() -> str:
    return """

def read_properties(path):
    properties = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            properties[key.strip()] = value.strip()
    return properties

collection_candidates = sorted({
    path.parent for path in archive_members if path.name == 'collection.properties'
})
hats_candidates = sorted({
    path.parent
    for path in archive_members
    if path.name == 'hats.properties'
    and read_properties(path).get('dataproduct_type', 'object') == 'object'
})
if collection_candidates:
    candidates = [('hats', path) for path in collection_candidates]
elif hats_candidates:
    candidates = [('hats', path) for path in hats_candidates]
else:
    parquet_files = sorted({path for path in archive_members if path.suffix.lower() == '.parquet'})
    parquet_parents = sorted({path.parent for path in parquet_files})
    if len(parquet_files) == 1:
        candidates = [('parquet', parquet_files[0])]
    elif len(parquet_parents) == 1:
        candidates = [('parquet', parquet_parents[0])]
    else:
        candidates = [('parquet', path) for path in parquet_parents]
    if not candidates:
        candidates = [
            ('csv', path) for path in archive_members if path.suffix.lower() == '.csv'
        ]

if len(candidates) != 1:
    raise ValueError(
        f'Expected exactly one HATS, Parquet, or CSV input in {archive_path}; found {candidates}'
    )
qa_input_format, qa_input_file = candidates[0]
"""


def _basic_information_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    if _pzs_runtime_input(config):
        return [
            _markdown_cell("Detected input and runtime access mode."),
            _code_cell("qa_runtime_summary"),
            _markdown_cell("First rows."),
            _code_cell("qa_data.head()"),
            _markdown_cell("Total number of rows."),
            _code_cell(
                "if qa_access_mode == 'lazy':\n"
                "    qa_total_rows = qa_row_count(qa_data)\n"
                "else:\n"
                "    qa_total_rows = len(qa_data)\n"
                "qa_total_rows"
            ),
            _markdown_cell("Total number of columns."),
            _code_cell("len(qa_data.columns)"),
            _markdown_cell("## Basic Statistics "),
            _code_cell(
                "if qa_access_mode == 'lazy':\n"
                "    if qa_input_format == 'hats':\n"
                "        catalog_statistics = qa_data.aggregate_column_statistics()\n"
                "    else:\n"
                "        catalog_statistics = qa_data.describe().compute()\n"
                "elif qa_input_format == 'hats':\n"
                "    catalog_statistics = lsdb.open_catalog(qa_input_file).aggregate_column_statistics()\n"
                "else:\n"
                "    catalog_statistics = qa_data.describe()\n"
                "display(HTML(\n"
                "    '<div style=\"max-height: 520px; overflow: auto;\">'\n"
                "    + catalog_statistics.to_html(max_rows=None, max_cols=None)\n"
                "    + '</div>'\n"
                "))\n"
                "del catalog_statistics"
            ),
        ]
    if _qa_data_mode(config) != "lazy":
        if str(config.get("input_format", "parquet")).lower() == "hats":
            describe_source = (
                "catalog_statistics = catalog.aggregate_column_statistics()\n"
                "display(HTML(\n"
                "    '<div style=\"max-height: 520px; overflow: auto;\">'\n"
                "    + catalog_statistics.to_html(max_rows=None, max_cols=None)\n"
                "    + '</div>'\n"
                "))\n"
                "del catalog_statistics"
            )
        else:
            describe_source = "df.describe()"
        return [
            _markdown_cell("First rows."),
            _code_cell("df.head()"),
            _markdown_cell("Total number of rows."),
            _code_cell("qa_total_rows = len(df)\nqa_total_rows"),
            _markdown_cell("Total number of columns."),
            _code_cell("len(df.columns.to_list())"),
            _markdown_cell("## Basic Statistics "),
            _code_cell(describe_source),
        ]

    input_format = str(config.get("input_format", "parquet")).lower()
    if input_format == "hats":
        input_file = _notebook_path(config, config["input_file"])
        head_source = "catalog.head()"
        row_count_source = (
            "count_column = catalog.columns[0]\n"
            f"count_data = lsdb.open_catalog({input_file!r}, columns=[count_column])\n"
            "qa_total_rows = qa_row_count(count_data)\n"
            "del count_data\n"
            "qa_total_rows"
        )
        column_count_source = "len(catalog.columns)"
        describe_source = (
            "catalog_statistics = catalog.aggregate_column_statistics()\n"
            "display(HTML(\n"
            "    '<div style=\"max-height: 520px; overflow: auto;\">'\n"
            "    + catalog_statistics.to_html(max_rows=None, max_cols=None)\n"
            "    + '</div>'\n"
            "))\n"
            "del catalog_statistics"
        )
    else:
        head_source = "df.head()"
        row_count_source = "qa_total_rows = qa_row_count(df)\nqa_total_rows"
        column_count_source = "len(df.columns)"
        describe_source = "df.describe().compute()"

    return [
        _markdown_cell("First rows."),
        _code_cell(head_source),
        _markdown_cell("Total number of rows."),
        _code_cell(row_count_source),
        _markdown_cell("Total number of columns."),
        _code_cell(column_count_source),
        _markdown_cell("## Basic Statistics "),
        _code_cell(describe_source),
    ]


def _data_quality_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _markdown_cell("## Missing values"),
        _markdown_cell("Null count and percentage for every catalog column."),
        _code_cell(_missing_values_source(config)),
        _markdown_cell("## Data quality warnings"),
        _markdown_cell(
            "Objective checks for empty or unusable data, invalid coordinates, and configured plots."
        ),
        _code_cell(_warnings_source(config)),
    ]


def _missing_values_source(config: dict[str, Any]) -> str:
    if _pzs_runtime_input(config):
        counts_source = (
            "if qa_access_mode == 'lazy':\n"
            "    qa_missing_count_values = qa_missing_counts(qa_data)\n"
            "else:\n"
            '    qa_missing_count_values = qa_data.isna().sum().astype("int64")'
        )
    elif _qa_data_mode(config) == "lazy":
        source_name = "catalog" if str(config.get("input_format", "parquet")).lower() == "hats" else "df"
        counts_source = f"qa_missing_count_values = qa_missing_counts({source_name})"
    else:
        counts_source = 'qa_missing_count_values = df.isna().sum().astype("int64")'
    return (
        f"{counts_source}\n"
        "qa_missing_values = pd.DataFrame({\n"
        '    "Column": qa_missing_count_values.index,\n'
        '    "Missing": qa_missing_count_values.to_numpy(),\n'
        "})\n"
        'qa_missing_values["Missing %"] = np.where(\n'
        "    qa_total_rows > 0,\n"
        '    100 * qa_missing_values["Missing"] / qa_total_rows,\n'
        "    0.0,\n"
        ")\n"
        "display(HTML(\n"
        "    '<div style=\"max-height: 520px; overflow: auto;\">'\n"
        '    + qa_missing_values.to_html(index=False, max_rows=None, float_format=lambda x: f"{x:.2f}")\n'
        "    + '</div>'\n"
        "))"
    )


def _warning_columns(config: dict[str, Any]) -> list[str]:
    plots = config.get("plots") or {}
    columns: list[str] = []
    spatial = plots.get("spatial")
    if spatial:
        columns.extend([spatial.get("ra_column", "ra"), spatial.get("dec_column", "dec")])
    for key, default in (("redshift", "redshift"), ("redshift_error", "redshift error")):
        if plots.get(key):
            columns.append(plots[key].get("column", default))
    if plots.get("quality"):
        columns.append(plots["quality"].get("column", "quality"))
    columns.extend(plot["column"] for plot in plots.get("categorical", []) or [])
    return list(dict.fromkeys(columns))


def _unusable_footprint_warnings(config: dict[str, Any]) -> list[str]:
    spatial = (config.get("plots") or {}).get("spatial")
    if not spatial:
        return []
    warnings_found = []
    for index, footprint in enumerate(_configured_footprints(spatial)):
        path = Path(footprint["path"])
        data = pd.read_csv(path)
        footprint_format = footprint.get("format") or _footprint_format(set(data.columns), path=path)
        if footprint_format == "region_vertices":
            usable = any(
                int(mask.sum()) >= 2
                for mask in [
                    np.isfinite(pd.to_numeric(group["ra_deg"], errors="coerce")).to_numpy()
                    & np.isfinite(pd.to_numeric(group["dec_deg"], errors="coerce")).to_numpy()
                    for _, group in data[data["ring_type"] == "exterior"].groupby("region_id")
                ]
            )
        else:
            mask = (
                np.isfinite(pd.to_numeric(data["ra_center"], errors="coerce")).to_numpy()
                & np.isfinite(pd.to_numeric(data["dec_limit"], errors="coerce")).to_numpy()
            )
            usable = int(mask.sum()) >= 2
        if not usable:
            label = footprint.get("label", f"Footprint {index + 1}")
            warnings_found.append(f"{label}: footprint has no curve with at least two finite points.")
    return warnings_found


def _warnings_source(config: dict[str, Any]) -> str:
    plots = config.get("plots") or {}
    lines = [
        "qa_warnings = []",
        "if qa_total_rows == 0:",
        '    qa_warnings.append("Catalog is empty.")',
        "for _, qa_missing_row in qa_missing_values.iterrows():",
        '    if qa_total_rows > 0 and int(qa_missing_row["Missing"]) == qa_total_rows:',
        "        qa_warnings.append(f'Column {qa_missing_row[\"Column\"]!r} is entirely null.')",
    ]
    warning_columns = _warning_columns(config)
    if _pzs_runtime_input(config) and warning_columns:
        lines.extend(
            [
                _lazy_data_open_source(config, warning_columns).replace("plot_data", "qa_diagnostic_data"),
                "qa_diagnostic_is_lazy = qa_access_mode == 'lazy'",
            ]
        )
        source_name = "qa_diagnostic_data"
    elif _qa_data_mode(config) == "lazy" and warning_columns:
        lines.extend(
            [_lazy_data_open_source(config, warning_columns).replace("plot_data", "qa_diagnostic_data")]
        )
        source_name = "qa_diagnostic_data"
    else:
        source_name = "df"

    spatial = plots.get("spatial")
    if spatial:
        ra = spatial.get("ra_column", "ra")
        dec = spatial.get("dec_column", "dec")
        if _pzs_runtime_input(config):
            lines.extend(
                [
                    "if qa_diagnostic_is_lazy:",
                    f"    qa_spatial = qa_spatial_diagnostics({source_name}, {ra!r}, {dec!r})",
                    "else:",
                    f"    qa_ra = pd.to_numeric({source_name}[{ra!r}], errors='coerce').to_numpy()",
                    f"    qa_dec = pd.to_numeric({source_name}[{dec!r}], errors='coerce').to_numpy()",
                    "    qa_finite_ra = np.isfinite(qa_ra)",
                    "    qa_finite_dec = np.isfinite(qa_dec)",
                    "    qa_spatial = pd.Series({",
                    '        "nonfinite_ra": (~qa_finite_ra).sum(),',
                    '        "nonfinite_dec": (~qa_finite_dec).sum(),',
                    '        "ra_outside_range": (qa_finite_ra & ((qa_ra < 0) | (qa_ra >= 360))).sum(),',
                    '        "dec_outside_range": (qa_finite_dec & ((qa_dec < -90) | (qa_dec > 90))).sum(),',
                    '        "valid_pairs": (qa_finite_ra & qa_finite_dec).sum(),',
                    "    })",
                ]
            )
        elif _qa_data_mode(config) == "lazy":
            lines.append(f"qa_spatial = qa_spatial_diagnostics({source_name}, {ra!r}, {dec!r})")
        else:
            lines.extend(
                [
                    f"qa_ra = pd.to_numeric(df[{ra!r}], errors='coerce').to_numpy()",
                    f"qa_dec = pd.to_numeric(df[{dec!r}], errors='coerce').to_numpy()",
                    "qa_finite_ra = np.isfinite(qa_ra)",
                    "qa_finite_dec = np.isfinite(qa_dec)",
                    "qa_spatial = pd.Series({",
                    '    "nonfinite_ra": (~qa_finite_ra).sum(),',
                    '    "nonfinite_dec": (~qa_finite_dec).sum(),',
                    '    "ra_outside_range": (qa_finite_ra & ((qa_ra < 0) | (qa_ra >= 360))).sum(),',
                    '    "dec_outside_range": (qa_finite_dec & ((qa_dec < -90) | (qa_dec > 90))).sum(),',
                    '    "valid_pairs": (qa_finite_ra & qa_finite_dec).sum(),',
                    "})",
                ]
            )
        lines.extend(
            [
                'if qa_spatial["nonfinite_ra"]:',
                f'    qa_warnings.append(f"Column {ra}: '
                "{int(qa_spatial['nonfinite_ra'])} non-finite coordinate values.\")",
                'if qa_spatial["nonfinite_dec"]:',
                f'    qa_warnings.append(f"Column {dec}: '
                "{int(qa_spatial['nonfinite_dec'])} non-finite coordinate values.\")",
                'if qa_spatial["ra_outside_range"]:',
                f'    qa_warnings.append(f"Column {ra}: '
                "{int(qa_spatial['ra_outside_range'])} values outside [0, 360).\")",
                'if qa_spatial["dec_outside_range"]:',
                f'    qa_warnings.append(f"Column {dec}: '
                "{int(qa_spatial['dec_outside_range'])} values outside [-90, 90].\")",
                'if qa_spatial["valid_pairs"] == 0:',
                '    qa_warnings.append("Spatial plot has no finite RA/Dec pairs.")',
            ]
        )

    for key, default in (("redshift", "redshift"), ("redshift_error", "redshift error")):
        plot = plots.get(key)
        if not plot:
            continue
        column = plot.get("column", default)
        value_range = plot.get("range")
        variable = f"qa_numeric_{key}"
        if _pzs_runtime_input(config):
            lines.extend(
                [
                    "if qa_diagnostic_is_lazy:",
                    f"    {variable} = qa_numeric_diagnostics({source_name}, {column!r}, {value_range!r})",
                    "else:",
                    f"    qa_values = pd.to_numeric({source_name}[{column!r}], errors='coerce').to_numpy()",
                    "    qa_finite = np.isfinite(qa_values)",
                    f"    {variable} = pd.Series({{'finite': qa_finite.sum(), 'in_range': qa_finite.sum()}})",
                ]
            )
            if value_range is not None:
                lines.append(
                    f"    {variable}['in_range'] = (qa_finite & (qa_values >= {value_range[0]!r}) "
                    f"& (qa_values <= {value_range[1]!r})).sum()"
                )
        elif _qa_data_mode(config) == "lazy":
            lines.append(f"{variable} = qa_numeric_diagnostics({source_name}, {column!r}, {value_range!r})")
        else:
            lines.extend(
                [
                    f"qa_values = pd.to_numeric(df[{column!r}], errors='coerce').to_numpy()",
                    "qa_finite = np.isfinite(qa_values)",
                    f"{variable} = pd.Series({{'finite': qa_finite.sum(), 'in_range': qa_finite.sum()}})",
                ]
            )
            if value_range is not None:
                lines.append(
                    f"{variable}['in_range'] = (qa_finite & (qa_values >= {value_range[0]!r}) "
                    f"& (qa_values <= {value_range[1]!r})).sum()"
                )
        lines.extend(
            [
                f"if {variable}['finite'] == 0:",
                f'    qa_warnings.append("Configured numeric column {column!r} has no finite values.")',
            ]
        )
        if value_range is not None:
            lines.extend(
                [
                    f"if {variable}['finite'] > 0 and {variable}['in_range'] == 0:",
                    f'    qa_warnings.append("Column {column!r} has no values inside '
                    f'histogram range {value_range!r}.")',
                ]
            )

    categorical = ([plots["quality"]] if plots.get("quality") else []) + list(
        plots.get("categorical", []) or []
    )
    for plot in categorical:
        column = plot.get("column", "quality")
        lines.extend(
            [
                "if qa_total_rows == 0 or "
                f"int(qa_missing_count_values.get({column!r}, qa_total_rows)) == qa_total_rows:",
                f'    qa_warnings.append("Configured categorical column {column!r} '
                'has no non-null categories.")',
            ]
        )
    for warning in _unusable_footprint_warnings(config):
        lines.append(f"qa_warnings.append({warning!r})")
    if (_pzs_runtime_input(config) or _qa_data_mode(config) == "lazy") and warning_columns:
        lines.append("del qa_diagnostic_data")
    lines.extend(
        [
            'qa_warning_text = "\\n".join(f"- {warning}" for warning in qa_warnings)',
            'display(Markdown(qa_warning_text if qa_warnings else "_No objective QA warnings._"))',
        ]
    )
    return "\n".join(lines)


def _lazy_helpers_source() -> str:
    return """# Exact partition-wise aggregations used by large-input QA cells.
def _qa_map_partitions(source, function, *args, meta):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="output of the function must be a DataFrame to generate an LSDB.*",
            category=RuntimeWarning,
        )
        return source.map_partitions(function, *args, meta=meta)


def _qa_partition_row_count(partition):
    return pd.Series([len(partition)], name="count", dtype="int64")


def qa_row_count(source):
    counts = _qa_map_partitions(
        source,
        _qa_partition_row_count,
        meta=pd.Series(name="count", dtype="int64"),
    ).compute()
    return int(counts.sum())


def _qa_partition_missing_counts(partition):
    return partition.isna().sum().rename("missing").astype("int64")


def qa_missing_counts(source):
    partials = _qa_map_partitions(
        source,
        _qa_partition_missing_counts,
        meta=pd.Series(name="missing", dtype="int64"),
    ).compute()
    return partials.groupby(level=0).sum()


def _qa_partition_numeric_diagnostics(partition, column, value_range):
    values = pd.to_numeric(partition[column], errors="coerce").to_numpy()
    finite = np.isfinite(values)
    in_range = finite
    if value_range is not None:
        in_range = finite & (values >= value_range[0]) & (values <= value_range[1])
    return pd.Series(
        {"finite": finite.sum(), "in_range": in_range.sum()}, dtype="int64"
    )


def qa_numeric_diagnostics(source, column, value_range=None):
    partials = _qa_map_partitions(
        source,
        _qa_partition_numeric_diagnostics,
        column,
        value_range,
        meta=pd.Series(dtype="int64"),
    ).compute()
    return partials.groupby(level=0).sum()


def _qa_partition_spatial_diagnostics(partition, ra_column, dec_column):
    ra = pd.to_numeric(partition[ra_column], errors="coerce").to_numpy()
    dec = pd.to_numeric(partition[dec_column], errors="coerce").to_numpy()
    finite_ra = np.isfinite(ra)
    finite_dec = np.isfinite(dec)
    valid = finite_ra & finite_dec
    return pd.Series(
        {
            "nonfinite_ra": (~finite_ra).sum(),
            "nonfinite_dec": (~finite_dec).sum(),
            "ra_outside_range": (finite_ra & ((ra < 0) | (ra >= 360))).sum(),
            "dec_outside_range": (finite_dec & ((dec < -90) | (dec > 90))).sum(),
            "valid_pairs": valid.sum(),
        },
        dtype="int64",
    )


def qa_spatial_diagnostics(source, ra_column, dec_column):
    partials = _qa_map_partitions(
        source,
        _qa_partition_spatial_diagnostics,
        ra_column,
        dec_column,
        meta=pd.Series(dtype="int64"),
    ).compute()
    return partials.groupby(level=0).sum()


def _qa_partition_numeric_range(partition, column):
    values = pd.to_numeric(partition[column], errors="coerce")
    return pd.Series({"min": values.min(), "max": values.max()}, dtype="float64")


def qa_numeric_range(source, column):
    stats = _qa_map_partitions(
        source,
        _qa_partition_numeric_range,
        column,
        meta=pd.Series(dtype="float64"),
    ).compute()
    return float(stats.loc["min"].min()), float(stats.loc["max"].max())


def _qa_partition_histogram1d(partition, column, edges):
    values = pd.to_numeric(partition[column], errors="coerce").to_numpy()
    values = values[np.isfinite(values)]
    counts, _ = np.histogram(values, bins=edges)
    return pd.Series(counts, index=np.arange(len(counts)), name="count", dtype="int64")


def qa_histogram1d(source, column, bins=50, value_range=None):
    if value_range is None:
        value_range = qa_numeric_range(source, column)
    if not np.all(np.isfinite(value_range)):
        value_range = (0.0, 1.0)
    elif value_range[0] == value_range[1]:
        value_range = (value_range[0] - 0.5, value_range[1] + 0.5)
    edges = np.linspace(value_range[0], value_range[1], bins + 1)
    partials = _qa_map_partitions(
        source,
        _qa_partition_histogram1d,
        column,
        edges,
        meta=pd.Series(name="count", dtype="int64"),
    ).compute()
    counts = partials.groupby(level=0).sum().reindex(range(bins), fill_value=0)
    return counts.to_numpy(), edges


def _qa_partition_histogram2d(partition, ra_column, dec_column, xedges, yedges):
    ra = pd.to_numeric(partition[ra_column], errors="coerce").to_numpy()
    dec = pd.to_numeric(partition[dec_column], errors="coerce").to_numpy()
    x = -np.deg2rad(((ra + 180) % 360) - 180)
    y = np.deg2rad(dec)
    valid = np.isfinite(x) & np.isfinite(y)
    counts, _, _ = np.histogram2d(x[valid], y[valid], bins=[xedges, yedges])
    return pd.Series(counts.ravel(), name="count", dtype="float64")


def qa_histogram2d(source, ra_column, dec_column, xedges, yedges):
    size = (len(xedges) - 1) * (len(yedges) - 1)
    partials = _qa_map_partitions(
        source,
        _qa_partition_histogram2d,
        ra_column,
        dec_column,
        xedges,
        yedges,
        meta=pd.Series(name="count", dtype="float64"),
    ).compute()
    counts = partials.groupby(level=0).sum().reindex(range(size), fill_value=0)
    return counts.to_numpy().reshape(len(xedges) - 1, len(yedges) - 1)


def _qa_partition_value_counts(partition, column):
    return partition[column].value_counts(dropna=False).rename("count")


def qa_value_counts(source, column):
    partials = _qa_map_partitions(
        source,
        _qa_partition_value_counts,
        column,
        meta=pd.Series(name="count", dtype="int64"),
    ).compute()
    return partials.groupby(level=0, dropna=False).sum().sort_index()
""".rstrip()


def _notebook_path(config: dict[str, Any], path: str | Path) -> str:
    input_file = Path(path)
    if config.get("include_absolute_input_path", True):
        return str(input_file.resolve())
    notebook_dir = _output_notebook(config).parent
    return os.path.relpath(input_file.resolve(), notebook_dir.resolve())


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
                _code_cell(_hist_source(config, redshift, default_label="redshift")),
            ]
        )

    quality = plots.get("quality")
    if quality:
        cells.extend(
            [
                _markdown_cell("### Quality Flags\n\n" + quality.get("description", "")),
                _code_cell(_quality_source(config, quality)),
            ]
        )

    redshift_error = plots.get("redshift_error")
    if redshift_error:
        cells.extend(
            [
                _markdown_cell("### Redshift error distribution\n\nGlobal redshift error distribution."),
                _code_cell(_hist_source(config, redshift_error, default_label="redshift error")),
            ]
        )

    categorical = plots.get("categorical", []) or []
    if categorical:
        cells.append(_markdown_cell("### Categorical distributions"))
        for plot in categorical:
            description = plot.get("description", "")
            if description:
                cells.append(_markdown_cell(description))
            cells.append(_code_cell(_quality_source(config, plot)))

    return cells


def _spatial_plot_source(config: dict[str, Any], spatial: dict[str, Any]) -> str:
    title = spatial.get("title", f"{config.get('title', 'Catalog')} - Spatial distribution")
    ra_column = spatial.get("ra_column", "ra")
    dec_column = spatial.get("dec_column", "dec")
    footprint_code = _footprint_plot_source(config, _configured_footprints(spatial))
    if _pzs_runtime_input(config):
        density_source = (
            _lazy_data_open_source(config, [ra_column, dec_column])
            + "\nxbins = np.linspace(-np.pi, np.pi, 180)\n"
            "ybins = np.linspace(-np.pi / 2, np.pi / 2, 90)\n"
            "if qa_access_mode == 'lazy':\n"
            f"    H = qa_histogram2d(plot_data, {ra_column!r}, {dec_column!r}, xbins, ybins)\n"
            "else:\n"
            f"    x_points = ra_to_mollweide_x(plot_data[{ra_column!r}].values)\n"
            f"    y_points = np.deg2rad(plot_data[{dec_column!r}].values)\n"
            "    valid = np.isfinite(x_points) & np.isfinite(y_points)\n"
            "    x_points = x_points[valid]\n"
            "    y_points = y_points[valid]\n"
            "    H, _, _ = np.histogram2d(x_points, y_points, bins=[xbins, ybins])\n"
            "H = np.ma.masked_where(H == 0, H)\n\n"
        )
        cleanup_source = (
            "\ndel plot_data, H\n"
            "if 'x_points' in locals(): del x_points\n"
            "if 'y_points' in locals(): del y_points\n"
        )
    elif _qa_data_mode(config) == "lazy":
        density_source = (
            _lazy_data_open_source(config, [ra_column, dec_column])
            + "\nxbins = np.linspace(-np.pi, np.pi, 180)\n"
            "ybins = np.linspace(-np.pi / 2, np.pi / 2, 90)\n"
            f"H = qa_histogram2d(plot_data, {ra_column!r}, {dec_column!r}, xbins, ybins)\n"
            "H = np.ma.masked_where(H == 0, H)\n\n"
        )
        cleanup_source = "\ndel plot_data, H\n"
    else:
        density_source = (
            f"x_points = ra_to_mollweide_x(df[{ra_column!r}].values)\n"
            f"y_points = np.deg2rad(df[{dec_column!r}].values)\n"
            "valid = np.isfinite(x_points) & np.isfinite(y_points)\n"
            "x_points = x_points[valid]\n"
            "y_points = y_points[valid]\n\n"
            "xbins = np.linspace(-np.pi, np.pi, 180)\n"
            "ybins = np.linspace(-np.pi / 2, np.pi / 2, 90)\n"
            "H, xedges, yedges = np.histogram2d(x_points, y_points, bins=[xbins, ybins])\n"
            "H = np.ma.masked_where(H == 0, H)\n\n"
        )
        cleanup_source = ""

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
        f"{density_source}"
        "fig = plt.figure(figsize=(16, 8))\n"
        'ax = fig.add_subplot(111, projection="mollweide")\n'
        "if H.count():\n"
        "    mesh = ax.pcolormesh(\n"
        '        xbins, ybins, H.T, norm=LogNorm(), shading="auto", cmap="viridis", zorder=1\n'
        "    )\n"
        "    cbar = fig.colorbar(mesh, ax=ax, pad=0.05)\n"
        '    cbar.set_label("Number of objects")\n'
        "else:\n"
        '    ax.text(0.5, 0.5, "No finite coordinate pairs", transform=ax.transAxes, ha="center")\n'
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
        f"{cleanup_source}"
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


def _lazy_data_open_source(config: dict[str, Any], columns: list[str]) -> str:
    if _pzs_runtime_input(config):
        return f"plot_data = qa_open_data(columns={columns!r})"
    input_file = _notebook_path(config, config["input_file"])
    input_format = str(config.get("input_format", "parquet")).lower()
    if input_format == "parquet":
        return f"plot_data = dd.read_parquet({input_file!r}, columns={columns!r})"
    if input_format == "csv":
        return f"plot_data = dd.read_csv({input_file!r}, usecols={columns!r})"
    if input_format == "hats":
        return f"plot_data = lsdb.open_catalog({input_file!r}, columns={columns!r})"
    raise ValueError("input_format must be one of: parquet, csv, hats.")


def _hist_source(config: dict[str, Any], plot: dict[str, Any], default_label: str) -> str:
    column = plot.get("column", default_label)
    value_range = plot.get("range")
    bins = int(plot.get("bins", 50))
    title = plot.get("title", f"{column} distribution")
    if _pzs_runtime_input(config):
        lazy_hist_source = (
            f"    hist_counts, hist_edges = qa_histogram1d(\n"
            f"        plot_data, {column!r}, bins={bins}, value_range={value_range!r}\n"
            "    )\n"
        )
        runtime_filter = "    runtime_hist_data = plot_data\n"
        runtime_xlim = ""
        cleanup_lines = []
        if value_range:
            min_value, max_value = value_range
            runtime_filter = (
                f"    runtime_hist_data = plot_data[(plot_data[{column!r}] >= {min_value}) "
                f"& (plot_data[{column!r}] <= {max_value})]\n"
            )
            runtime_xlim = f"plt.xlim({min_value}, {max_value})\n"
        cleanup_lines = [
            "del plot_data",
            "if 'hist_counts' in locals(): del hist_counts, hist_edges, hist_centers",
            "if 'runtime_hist_data' in locals(): del runtime_hist_data",
        ]
        return (
            _lazy_data_open_source(config, [column])
            + "\nif qa_access_mode == 'lazy':\n"
            + lazy_hist_source
            + "    hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2\n"
            + "    plt.figure(figsize=(8, 6))\n"
            + "    sns.histplot(x=hist_centers, weights=hist_counts, bins=hist_edges.tolist(), kde=False)\n"
            + "else:\n"
            + "    plt.figure(figsize=(8, 6))\n"
            + runtime_filter
            + f"    sns.histplot(data=runtime_hist_data, x={column!r}, kde=True, bins={bins})\n"
            + f"plt.xlabel({column!r})\n"
            + 'plt.ylabel("Count")\n'
            + f"plt.title({title!r})\n"
            + runtime_xlim
            + "plt.tight_layout()\n"
            + "plt.show()\n"
            + "\n".join(cleanup_lines)
        )
    if _qa_data_mode(config) == "lazy":
        lazy_xlim = f"plt.xlim({value_range[0]}, {value_range[1]})\n" if value_range else ""
        return (
            _lazy_data_open_source(config, [column]) + "\n" + f"hist_counts, hist_edges = qa_histogram1d(\n"
            f"    plot_data, {column!r}, bins={bins}, value_range={value_range!r}\n"
            ")\n"
            "hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2\n"
            "plt.figure(figsize=(8, 6))\n"
            "sns.histplot(\n"
            "    x=hist_centers,\n"
            "    weights=hist_counts,\n"
            "    bins=hist_edges.tolist(),\n"
            "    kde=False,\n"
            ")\n"
            f"plt.xlabel({column!r})\n"
            'plt.ylabel("Count")\n'
            f"plt.title({title!r})\n"
            f"{lazy_xlim}"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "del plot_data, hist_counts, hist_edges, hist_centers"
        )
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
        f"plt.title({title!r})\n"
        f"{xlim}"
        "plt.tight_layout()\n"
        "plt.show()"
    )


def _quality_source(config: dict[str, Any], quality: dict[str, Any]) -> str:
    column = quality.get("column", "quality")
    title = quality.get("title", f"{column} distribution")
    label_rotation = quality.get("label_rotation", 0)
    if _pzs_runtime_input(config):
        return (
            _lazy_data_open_source(config, [column])
            + "\nif qa_access_mode == 'lazy':\n"
            + f"    quality_counts = qa_value_counts(plot_data, {column!r})\n"
            + "    quality_counts = quality_counts[quality_counts.index.notna()]\n"
            + "    plt.figure(figsize=(8, 6))\n"
            + "    if quality_counts.empty:\n"
            + '        plt.text(0.5, 0.5, "No non-null categories", ha="center")\n'
            + "    else:\n"
            + "        sns.barplot(x=quality_counts.index.astype(str), y=quality_counts.to_numpy())\n"
            + "else:\n"
            + f"    quality_order = sorted(plot_data[{column!r}].dropna().unique(), key=str)\n"
            + "    plt.figure(figsize=(8, 6))\n"
            + "    if quality_order:\n"
            + f"        sns.countplot(data=plot_data, x={column!r}, order=quality_order)\n"
            + "    else:\n"
            + '        plt.text(0.5, 0.5, "No non-null categories", ha="center")\n'
            + f"plt.xlabel({column!r})\n"
            + 'plt.ylabel("Count")\n'
            + f"plt.title({title!r})\n"
            + f"plt.xticks(rotation={label_rotation!r})\n"
            + "plt.tight_layout()\n"
            + "plt.show()\n"
            + "del plot_data\n"
            + "if 'quality_counts' in locals(): del quality_counts\n"
            + "if 'quality_order' in locals(): del quality_order"
        )
    if _qa_data_mode(config) == "lazy":
        return (
            _lazy_data_open_source(config, [column])
            + "\n"
            + f"quality_counts = qa_value_counts(plot_data, {column!r})\n"
            "quality_counts = quality_counts[quality_counts.index.notna()]\n"
            "plt.figure(figsize=(8, 6))\n"
            "if quality_counts.empty:\n"
            '    plt.text(0.5, 0.5, "No non-null categories", ha="center")\n'
            "else:\n"
            "    sns.barplot(\n"
            "        x=quality_counts.index.astype(str),\n"
            "        y=quality_counts.to_numpy(),\n"
            "    )\n"
            f"plt.xlabel({column!r})\n"
            'plt.ylabel("Count")\n'
            f"plt.title({title!r})\n"
            f"plt.xticks(rotation={label_rotation!r})\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "del plot_data, quality_counts"
        )
    return (
        "plt.figure(figsize=(8, 6))\n"
        f"quality_order = sorted(df[{column!r}].dropna().unique(), key=str)\n"
        "if quality_order:\n"
        "    sns.countplot(\n"
        "        data=df,\n"
        f"        x={column!r},\n"
        "        order=quality_order,\n"
        "    )\n"
        "else:\n"
        '    plt.text(0.5, 0.5, "No non-null categories", ha="center")\n\n'
        f"plt.xlabel({column!r})\n"
        'plt.ylabel("Count")\n'
        f"plt.title({title!r})\n"
        f"plt.xticks(rotation={label_rotation!r})\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "del quality_order"
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
