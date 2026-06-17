import warnings
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_HATS_MARGIN_THRESHOLD_ARCSEC = 5.0
HATS_COORDINATE_ERROR = (
    "HATS output requires RA/Dec already in standard degree ranges. "
    "Use prepare to Parquet first, then curate with coordinate transformations, then write HATS."
)
DEFAULT_HATS_OUTPUT_WITH_MARGIN = True


def is_hats_input(path: Path) -> bool:
    """Return whether a path looks like a HATS catalog or collection."""
    if not path.is_dir():
        return False
    return (
        (path / "hats.properties").exists()
        or (path / "collection.properties").exists()
        or (path / "catalog" / "hats.properties").exists()
    )


def hats_config(config: dict[str, Any], error_cls: type[Exception] = ValueError) -> dict[str, Any]:
    """Return the optional HATS config block as a mapping."""
    hats_block = config.get("hats", {})
    if hats_block is None:
        hats_block = {}
    if not isinstance(hats_block, dict):
        raise error_cls("hats must be a mapping.")
    return hats_block


def _hats_value(
    config: dict[str, Any],
    key: str,
    default: Any = None,
    error_cls: type[Exception] = ValueError,
) -> Any:
    hats_block = hats_config(config, error_cls)
    return hats_block.get(key, config.get(key, default))


def hats_output_with_margin(config: dict[str, Any], error_cls: type[Exception] = ValueError) -> bool:
    """Return whether HATS output should include a margin cache."""
    value = _hats_value(
        config,
        "hats_output_with_margin",
        DEFAULT_HATS_OUTPUT_WITH_MARGIN,
        error_cls,
    )
    if not isinstance(value, bool):
        raise error_cls("hats.hats_output_with_margin must be true or false.")
    return value


def hats_required_column(config: dict[str, Any], key: str, error_cls: type[Exception] = ValueError) -> str:
    """Return a required HATS column setting."""
    hats_block = hats_config(config, error_cls)
    value = hats_block.get(key, config.get(key))
    if not isinstance(value, str) or not value.strip():
        raise error_cls(f"hats.{key} must be a non-empty string when output_format: hats.")
    return value


def hats_ra_dec_columns_required(
    config: dict[str, Any], error_cls: type[Exception] = ValueError
) -> tuple[str, str]:
    """Return required HATS RA/Dec column names."""
    return hats_required_column(config, "ra_column", error_cls), hats_required_column(
        config, "dec_column", error_cls
    )


def hats_ra_dec_columns_with_defaults(
    config: dict[str, Any],
    default_ra_column: str,
    default_dec_column: str,
    error_cls: type[Exception] = ValueError,
) -> tuple[str, str]:
    """Return HATS RA/Dec column names, defaulting to validated curate columns."""
    hats_block = hats_config(config, error_cls)
    ra_column = hats_block.get("ra_column", config.get("ra_column", default_ra_column))
    dec_column = hats_block.get("dec_column", config.get("dec_column", default_dec_column))
    if not isinstance(ra_column, str) or not ra_column.strip():
        raise error_cls("hats.ra_column must be a non-empty string when output_format: hats.")
    if not isinstance(dec_column, str) or not dec_column.strip():
        raise error_cls("hats.dec_column must be a non-empty string when output_format: hats.")
    return ra_column, dec_column


def hats_catalog_name(
    output_dir: Path, config: dict[str, Any], error_cls: type[Exception] = ValueError
) -> str:
    """Return the HATS catalog artifact name."""
    hats_block = hats_config(config, error_cls)
    value = hats_block.get("catalog_name", config.get("catalog_name", output_dir.name))
    if not isinstance(value, str) or not value.strip():
        raise error_cls("hats.catalog_name must be a non-empty string when provided.")
    return value


def hats_margin_threshold(config: dict[str, Any], error_cls: type[Exception] = ValueError) -> float:
    """Return the HATS margin cache threshold in arcseconds."""
    hats_block = hats_config(config, error_cls)
    value = hats_block.get("margin_threshold", DEFAULT_HATS_MARGIN_THRESHOLD_ARCSEC)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise error_cls("hats.margin_threshold must be a positive number.")
    if hats_output_with_margin(config, error_cls) and value == 0:
        raise error_cls(
            "hats.margin_threshold cannot be 0 when hats_output_with_margin is true. "
            "Set a positive value or omit it to use the default."
        )
    if hats_output_with_margin(config, error_cls) and value < 0:
        raise error_cls("hats.margin_threshold must be a positive number.")
    if not hats_output_with_margin(config, error_cls) and value < 0:
        raise error_cls("hats.margin_threshold must be a non-negative number.")
    return float(value)


def hats_margin_threshold_was_configured(config: dict[str, Any]) -> bool:
    """Return whether the user explicitly configured a HATS margin threshold."""
    hats_block = hats_config(config)
    return "margin_threshold" in hats_block or "margin_threshold" in config


def hats_sort_columns(config: dict[str, Any], error_cls: type[Exception] = ValueError) -> str | None:
    """Return the optional HATS import sort column."""
    hats_block = hats_config(config, error_cls)
    value = hats_block.get("sort_columns")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise error_cls("hats.sort_columns must be a non-empty string when provided.")
    return value


def hats_create_thumbnail(config: dict[str, Any], error_cls: type[Exception] = ValueError) -> bool:
    """Return whether LSDB should create a HATS thumbnail."""
    return bool(hats_config(config, error_cls).get("create_thumbnail", False))


def read_hats_properties(path: Path) -> dict[str, str]:
    """Read a HATS properties file into a simple string mapping."""
    properties: dict[str, str] = {}
    if not path.exists():
        return properties
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        properties[key.strip()] = value.strip()
    return properties


def hats_primary_catalog_path(path: Path) -> Path:
    """Return the primary object catalog path for a HATS catalog or collection."""
    if (path / "hats.properties").exists():
        return path
    collection_properties = read_hats_properties(path / "collection.properties")
    primary = collection_properties.get("hats_primary_table_url")
    if primary:
        primary_path = Path(primary)
        return primary_path if primary_path.is_absolute() else path / primary_path
    if (path / "catalog" / "hats.properties").exists():
        return path / "catalog"
    return path


def _hats_margin_threshold_from_properties(properties: dict[str, str]) -> float | None:
    value = properties.get("hats_margin_threshold")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def hats_margin_catalogs(path: Path) -> list[tuple[Path, float]]:
    """Return margin catalog directories and thresholds inside a HATS collection."""
    if not (path / "collection.properties").exists():
        return []
    margins: list[tuple[Path, float]] = []
    for child in sorted(candidate for candidate in path.iterdir() if candidate.is_dir()):
        properties = read_hats_properties(child / "hats.properties")
        if properties.get("dataproduct_type") != "margin":
            continue
        threshold = _hats_margin_threshold_from_properties(properties)
        if threshold is not None:
            margins.append((child, threshold))
    return margins


def select_hats_margin_catalog(path: Path, threshold: float | None = None) -> Path | None:
    """Select a margin catalog from a HATS collection."""
    margins = hats_margin_catalogs(path)
    if not margins:
        return None
    if threshold is not None:
        for margin_path, margin_threshold in margins:
            if abs(margin_threshold - threshold) <= 1e-9:
                return margin_path
        return None
    return min(margins, key=lambda item: item[1])[0]


def open_hats_catalog_kwargs(path: Path, config: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
    """Return path and kwargs for opening a HATS catalog with the desired margin."""
    if not hats_output_with_margin(config):
        return hats_primary_catalog_path(path), {}

    threshold = hats_margin_threshold(config)
    threshold_was_configured = hats_margin_threshold_was_configured(config)
    if threshold_was_configured:
        margin_path = select_hats_margin_catalog(path, threshold=threshold)
        if margin_path is not None:
            return path, {"margin_cache": margin_path}
        return hats_primary_catalog_path(path), {}

    collection_properties = read_hats_properties(path / "collection.properties")
    if collection_properties.get("default_margin"):
        return path, {}

    margin_path = select_hats_margin_catalog(path)
    return path, {"margin_cache": margin_path} if margin_path is not None else {}


def warn_missing_default_hats_margin(path: Path) -> None:
    """Warn that a HATS collection did not open with a default margin."""
    warnings.warn(
        f"HATS input '{path}' did not open with a default margin. The input may have no margin "
        "cache, or an existing margin may not be marked as default. The pipeline will try to use "
        "an existing margin catalog from the collection or generate the requested output margin.",
        UserWarning,
        stacklevel=2,
    )


def hats_import_dask_runtime(config: dict[str, Any], cluster_config: dict[str, Any]) -> tuple[int, int]:
    """Return Dask worker/thread settings passed through to hats_import."""
    args = dict(cluster_config.get("args", {}) or {})
    return int(args.get("n_workers", 1) or 1), int(args.get("threads_per_worker", 1) or 1)


def write_hats_from_dataframe(
    df: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
    *,
    ra_column: str,
    dec_column: str,
    partition_bytes: int,
    error_cls: type[Exception] = ValueError,
) -> int:
    """Write a pandas dataframe as a HATS collection through LSDB."""
    import lsdb

    catalog = lsdb.from_dataframe(
        df,
        ra_column=ra_column,
        dec_column=dec_column,
        margin_threshold=(
            hats_margin_threshold(config, error_cls) if hats_output_with_margin(config, error_cls) else None
        ),
        partition_bytes=partition_bytes,
    )
    catalog.write_catalog(
        output_dir,
        catalog_name=hats_catalog_name(output_dir, config, error_cls),
        as_collection=True,
        overwrite=True,
        progress_bar=bool(config.get("progress_bar", False)),
        create_thumbnail=hats_create_thumbnail(config, error_cls),
    )
    return 1


def run_hats_import_from_parquet(
    parquet_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    client: Any,
    *,
    ra_column: str,
    dec_column: str,
    cluster_config: dict[str, Any],
    error_cls: type[Exception] = ValueError,
) -> None:
    """Import a Parquet dataset into a HATS collection with hats_import."""
    from hats_import import CollectionArguments
    from hats_import.pipeline import pipeline_with_client

    n_workers, threads_per_worker = hats_import_dask_runtime(config, cluster_config)
    collection_args = CollectionArguments(
        output_artifact_name=hats_catalog_name(output_dir, config, error_cls),
        output_path=output_dir,
        dask_n_workers=n_workers,
        dask_threads_per_worker=threads_per_worker,
        progress_bar=bool(config.get("progress_bar", False)),
    ).catalog(
        input_path=parquet_dir,
        file_reader="parquet",
        ra_column=ra_column,
        dec_column=dec_column,
        sort_columns=hats_sort_columns(config, error_cls),
    )
    if hats_output_with_margin(config, error_cls):
        collection_args = collection_args.add_margin(
            margin_threshold=hats_margin_threshold(config, error_cls), is_default=True
        )
    pipeline_with_client(collection_args, client)


def add_margin_to_hats_collection(
    collection_dir: Path,
    catalog_name: str,
    config: dict[str, Any],
    client: Any,
    *,
    cluster_config: dict[str, Any],
    error_cls: type[Exception] = ValueError,
) -> None:
    """Add a default margin cache to an existing HATS collection."""
    from hats_import import CollectionArguments
    from hats_import.pipeline import pipeline_with_client

    if not hats_output_with_margin(config, error_cls):
        return
    n_workers, threads_per_worker = hats_import_dask_runtime(config, cluster_config)
    collection_args = (
        CollectionArguments(
            output_artifact_name=collection_dir.name,
            output_path=collection_dir.parent,
            dask_n_workers=n_workers,
            dask_threads_per_worker=threads_per_worker,
            progress_bar=bool(config.get("progress_bar", False)),
        )
        .catalog(catalog_path=collection_dir / catalog_name)
        .add_margin(margin_threshold=hats_margin_threshold(config, error_cls), is_default=True)
    )
    pipeline_with_client(collection_args, client)
