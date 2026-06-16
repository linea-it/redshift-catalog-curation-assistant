from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_HATS_MARGIN_THRESHOLD_ARCSEC = 5.0
HATS_COORDINATE_ERROR = (
    "HATS output requires RA/Dec already in standard degree ranges. "
    "Use prepare to Parquet first, then curate with coordinate transformations, then write HATS."
)


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
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise error_cls("hats.margin_threshold must be a non-negative number.")
    return float(value)


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
        margin_threshold=hats_margin_threshold(config, error_cls),
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
    collection_args = (
        CollectionArguments(
            output_artifact_name=hats_catalog_name(output_dir, config, error_cls),
            output_path=output_dir,
            dask_n_workers=n_workers,
            dask_threads_per_worker=threads_per_worker,
            progress_bar=bool(config.get("progress_bar", False)),
        )
        .catalog(
            input_path=parquet_dir,
            file_reader="parquet",
            ra_column=ra_column,
            dec_column=dec_column,
            sort_columns=hats_sort_columns(config, error_cls),
        )
        .add_margin(margin_threshold=hats_margin_threshold(config, error_cls), is_default=True)
    )
    pipeline_with_client(collection_args, client)
