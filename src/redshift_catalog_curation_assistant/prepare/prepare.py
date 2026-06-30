import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from dask import delayed

from ..executor import dask_client_context, dask_cluster_config
from ..hats import (
    HATS_COORDINATE_ERROR,
    hats_catalog_name,
    hats_margin_threshold,
    hats_output_with_margin,
    hats_ra_dec_columns_required,
    hats_sort_columns,
    is_hats_input,
    run_hats_import_from_parquet,
    write_hats_from_dataframe,
)
from ..io import DEFAULT_DASK_THRESHOLD_BYTES, read_table

LOGGER = logging.getLogger(__name__)

COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".zip"}
HEADERLESS_SUFFIXES = {".dat", ".idz"}
PARQUET_SUFFIXES = {".parquet", ".pq"}
FITS_SUFFIXES = {".fits", ".fit", ".fts"}
TEXT_SUFFIXES = {".csv", ".txt", ".dat", ".idz"}
DEFAULT_CHUNK_SIZE_ROWS = 200_000
DEFAULT_TARGET_PARTITION_SIZE_MB = 100
OUTPUT_MODES = {"auto", "single", "partitioned"}
OUTPUT_FORMATS = {"parquet", "hats"}
SCHEMA_POLICIES = {"strict", "union"}


class PrepareError(ValueError):
    """Raised when catalog preparation cannot continue safely."""


def load_prepare_config(path: Path) -> dict[str, Any]:
    """Load a YAML prepare configuration file."""
    with open(path, "r") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def _bytes_from_mb(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(float(value) * 1024 * 1024)


def _large_file_threshold_bytes(config: dict[str, Any]) -> int:
    return _bytes_from_mb(
        config.get("large_file_threshold_mb", config.get("dask_threshold_mb")),
        DEFAULT_DASK_THRESHOLD_BYTES,
    )


def _target_partition_size(config: dict[str, Any]) -> str:
    value = config.get("target_partition_size_mb", DEFAULT_TARGET_PARTITION_SIZE_MB)
    return f"{int(float(value))}MB"


def _target_partition_size_bytes(config: dict[str, Any]) -> int:
    return _bytes_from_mb(
        config.get("target_partition_size_mb"), DEFAULT_TARGET_PARTITION_SIZE_MB * 1024 * 1024
    )


def _output_mode(config: dict[str, Any]) -> str:
    output_mode = str(config.get("output_mode", "auto")).lower()
    if output_mode not in OUTPUT_MODES:
        raise PrepareError("output_mode must be one of: auto, single, partitioned.")
    return output_mode


def _output_format(config: dict[str, Any]) -> str:
    output_format = str(config.get("output_format", "parquet")).lower()
    if output_format not in OUTPUT_FORMATS:
        raise PrepareError("output_format must be one of: parquet, hats.")
    return output_format


def _schema_policy(config: dict[str, Any]) -> str:
    policy = str(config.get("schema_policy", "strict")).lower()
    if policy not in SCHEMA_POLICIES:
        raise PrepareError("schema_policy must be one of: strict, union.")
    return policy


def _union_schema(schemas: list[list[str]]) -> list[str]:
    return list(dict.fromkeys(column for schema in schemas for column in schema))


def _hats_ra_dec_columns(config: dict[str, Any]) -> tuple[str, str]:
    return hats_ra_dec_columns_required(config, PrepareError)


def _hats_catalog_name(output_dir: Path, config: dict[str, Any]) -> str:
    return hats_catalog_name(output_dir, config, PrepareError)


def _hats_margin_threshold(config: dict[str, Any]) -> float:
    return hats_margin_threshold(config, PrepareError)


def _hats_output_with_margin(config: dict[str, Any]) -> bool:
    return hats_output_with_margin(config, PrepareError)


def _hats_sort_columns(config: dict[str, Any]) -> str | None:
    return hats_sort_columns(config, PrepareError)


def _hats_coordinate_range(values: Any) -> tuple[int, float, float]:
    count = values.count()
    min_value = values.min()
    max_value = values.max()
    if hasattr(min_value, "compute"):
        import dask

        count, min_value, max_value = dask.compute(count, min_value, max_value)
    return int(count), float(min_value), float(max_value)


def _validate_hats_coordinate_columns(df: Any, config: dict[str, Any]) -> None:
    ra_column, dec_column = _hats_ra_dec_columns(config)
    missing = [column for column in [ra_column, dec_column] if column not in df.columns]
    if missing:
        raise PrepareError(f"{HATS_COORDINATE_ERROR} Missing coordinate columns: {', '.join(missing)}.")

    for column, label in [(ra_column, "RA"), (dec_column, "Dec")]:
        if not pd.api.types.is_numeric_dtype(df[column].dtype):
            raise PrepareError(f"{HATS_COORDINATE_ERROR} {label} column '{column}' must be numeric.")

    ra_count, ra_min, ra_max = _hats_coordinate_range(df[ra_column])
    dec_count, dec_min, dec_max = _hats_coordinate_range(df[dec_column])
    if ra_count <= 0 or dec_count <= 0:
        raise PrepareError(f"{HATS_COORDINATE_ERROR} Coordinate columns must contain non-null values.")
    if ra_min < 0.0 or ra_max >= 360.0:
        raise PrepareError(
            f"{HATS_COORDINATE_ERROR} RA column '{ra_column}' observed range: [{ra_min}, {ra_max}]."
        )
    if dec_min <= -90.0 or dec_max >= 90.0:
        raise PrepareError(
            f"{HATS_COORDINATE_ERROR} Dec column '{dec_column}' observed range: [{dec_min}, {dec_max}]."
        )


def _data_suffix(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    data_suffixes = [suffix for suffix in suffixes if suffix not in COMPRESSED_SUFFIXES]
    return data_suffixes[-1] if data_suffixes else path.suffix.lower()


def _is_compressed(path: Path) -> bool:
    return any(suffix.lower() in COMPRESSED_SUFFIXES for suffix in path.suffixes)


def _decompress_command(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return f"gzip -dk {path}"
    if suffix == ".bz2":
        return f"bzip2 -dk {path}"
    if suffix == ".xz":
        return f"xz -dk {path}"
    if suffix == ".zip":
        return f"unzip {path}"
    return f"decompress {path}"


def _check_large_compressed(path: Path, threshold_bytes: int) -> None:
    if not path.is_file() or not _is_compressed(path) or path.stat().st_size < threshold_bytes:
        return
    msg = (
        f"Compressed input is larger than the configured threshold: {path}\n\n"
        "Large compressed files are not processed directly because they cannot be partitioned "
        "efficiently. Decompress the file first, then run prepare again.\n\n"
        f"Suggested command:\n  {_decompress_command(path)}"
    )
    raise PrepareError(msg)


def _total_input_size(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if path.is_dir():
            total += sum(part.stat().st_size for part in path.rglob("*") if part.is_file())
        else:
            total += path.stat().st_size
    return total


def _as_paths(paths: list[str | Path]) -> list[Path]:
    return [Path(path) for path in paths]


def _input_paths(config: dict[str, Any]) -> list[Path]:
    if "input_files" in config:
        paths = config["input_files"]
        if not isinstance(paths, list | tuple) or not paths:
            raise PrepareError("input_files must be a non-empty list of paths.")
        if not all(isinstance(path, str | Path) and str(path).strip() for path in paths):
            raise PrepareError("input_files must contain only non-empty path strings.")
        return _as_paths(list(paths))
    if "input_file" not in config:
        raise PrepareError("prepare config requires input_file or input_files.")
    input_file = config["input_file"]
    if not isinstance(input_file, str | Path) or not str(input_file).strip():
        raise PrepareError("input_file must be a non-empty path string.")
    return [Path(input_file)]


def _validate_prepare_bool(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if value is not None and not isinstance(value, bool):
        raise PrepareError(f"{key} must be true or false.")


def _validate_prepare_positive_int(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PrepareError(f"{key} must be a positive integer.")


def _validate_prepare_non_negative_number(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise PrepareError(f"{key} must be a non-negative number.")


def _validate_prepare_positive_number(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise PrepareError(f"{key} must be a positive number.")


def _validate_prepare_column_names(config: dict[str, Any]) -> None:
    column_names = config.get("column_names")
    if column_names is None:
        return
    if not isinstance(column_names, list | tuple) or not column_names:
        raise PrepareError("column_names must be a non-empty list of strings.")
    if not all(isinstance(column, str) and column.strip() for column in column_names):
        raise PrepareError("column_names must contain only non-empty strings.")


def _parquet_schema_for_path(path: Path) -> list[str]:
    import pyarrow.dataset as ds

    return [str(name) for name in ds.dataset(path, format="parquet").schema.names]


def _dry_run_schema_for_path(path: Path, suffix: str, config: dict[str, Any]) -> list[str]:
    if path.is_dir() and suffix in PARQUET_SUFFIXES:
        return _parquet_schema_for_path(path)
    return _schema_for_path(path, suffix, config)


def dry_run_prepare_config(config: dict[str, Any]) -> Path:
    """Validate a prepare configuration and input schemas without writing output."""
    if not isinstance(config, dict):
        raise PrepareError("prepare config must be a YAML mapping.")
    if not config:
        raise PrepareError("prepare config is empty.")

    input_paths = _input_paths(config)
    output_dir_value = config.get("output_dir")
    if not isinstance(output_dir_value, str | Path) or not str(output_dir_value).strip():
        raise PrepareError("prepare config requires a non-empty output_dir.")
    output_dir = Path(output_dir_value)
    for key in ["overwrite", "allow_large_single_output", "progress_bar"]:
        _validate_prepare_bool(config, key)
    _validate_prepare_column_names(config)
    _validate_prepare_positive_int(config, "fits_hdu")
    _validate_prepare_positive_int(config, "chunk_size_rows")
    _validate_prepare_non_negative_number(config, "large_file_threshold_mb")
    _validate_prepare_non_negative_number(config, "dask_threshold_mb")
    _validate_prepare_positive_number(config, "target_partition_size_mb")
    if "part_prefix" in config and (
        not isinstance(config["part_prefix"], str) or not config["part_prefix"].strip()
    ):
        raise PrepareError("part_prefix must be a non-empty string.")

    output_format = _output_format(config)
    _output_mode(config)
    schema_policy = _schema_policy(config)
    _target_partition_size_bytes(config)
    cluster_config = dask_cluster_config(config)
    _dask_logs_dir(cluster_config, output_dir)

    for path in input_paths:
        if not path.exists():
            raise PrepareError(f"input path does not exist: {path}")
        if is_hats_input(path):
            raise PrepareError(
                "Input is already a HATS catalog or collection. HATS is supported directly by "
                "inspect and curate, so prepare is not required for this input."
            )
        _check_large_compressed(path, _large_file_threshold_bytes(config))

    if len(input_paths) == 1 and input_paths[0].is_dir():
        if not any(input_paths[0].glob("*.parquet")) and not (input_paths[0] / "_metadata").exists():
            raise PrepareError(f"Unsupported directory input: {input_paths[0]}. Expected a Parquet dataset.")
        schema = _parquet_schema_for_path(input_paths[0])
        if output_format == "hats":
            ra_column, dec_column = _hats_ra_dec_columns(config)
            missing = [column for column in [ra_column, dec_column] if column not in schema]
            if missing:
                raise PrepareError(
                    f"{HATS_COORDINATE_ERROR} Missing coordinate columns: {', '.join(missing)}."
                )
            _hats_margin_threshold(config)
            _hats_catalog_name(output_dir, config)
            _hats_sort_columns(config)
        return output_dir

    suffixes = {_data_suffix(path) for path in input_paths}
    if len(suffixes) != 1:
        raise PrepareError("All input files must have the same format before prepare can partition them.")
    suffix = suffixes.pop()
    schemas = {path: _dry_run_schema_for_path(path, suffix, config) for path in input_paths}
    first_schema = schemas[input_paths[0]]
    mismatches = {path: schema for path, schema in schemas.items() if schema != first_schema}
    if mismatches and schema_policy == "strict":
        lines = ["Multi-file inputs must describe one logical catalog with the same schema."]
        lines.append(f"{input_paths[0]}: {', '.join(first_schema)}")
        for path, schema in mismatches.items():
            lines.append(f"{path}: {', '.join(schema)}")
        raise PrepareError("\n".join(lines))

    is_small_input = _total_input_size(input_paths) < _large_file_threshold_bytes(config)
    _resolve_output_mode(
        config,
        is_single_small_file=len(input_paths) == 1 and is_small_input,
        is_multi_file=len(input_paths) > 1,
    )
    effective_schema = _union_schema(list(schemas.values()))
    if output_format == "hats":
        ra_column, dec_column = _hats_ra_dec_columns(config)
        missing = [column for column in [ra_column, dec_column] if column not in effective_schema]
        if missing:
            raise PrepareError(f"{HATS_COORDINATE_ERROR} Missing coordinate columns: {', '.join(missing)}.")
        _hats_margin_threshold(config)
        _hats_catalog_name(output_dir, config)
        _hats_sort_columns(config)
    return output_dir


def _write_manifest(
    output_dir: Path,
    input_paths: list[Path],
    input_format: str,
    config: dict[str, Any],
    n_partitions: int,
    output_mode: str,
) -> None:
    output_format = _output_format(config)
    manifest = {
        "source_paths": [str(path) for path in input_paths],
        "source_format": input_format,
        "partition_format": output_format,
        "requested_output_mode": _output_mode(config),
        "output_mode": output_mode,
        "n_partitions": n_partitions,
        "large_file_threshold_mb": float(
            config.get("large_file_threshold_mb", config.get("dask_threshold_mb", 100))
        ),
        "target_partition_size_mb": float(
            config.get("target_partition_size_mb", DEFAULT_TARGET_PARTITION_SIZE_MB)
        ),
    }
    if "fits_hdu" in config:
        manifest["fits_hdu"] = int(config["fits_hdu"])
    if output_format == "hats":
        manifest["hats"] = {
            "catalog_name": _hats_catalog_name(output_dir, config),
            "ra_column": _hats_ra_dec_columns(config)[0],
            "dec_column": _hats_ra_dec_columns(config)[1],
            "hats_output_with_margin": _hats_output_with_margin(config),
            "margin_threshold": _hats_margin_threshold(config) if _hats_output_with_margin(config) else None,
            "sort_columns": _hats_sort_columns(config),
        }
    (output_dir / "_redshift_curator_manifest.json").write_text(json.dumps(manifest, indent=2))


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            msg = f"Output directory already exists: {output_dir}. Pass --overwrite to replace it."
            raise PrepareError(msg)
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _read_fits_chunk(path: str, fits_hdu: int, start: int, stop: int) -> pd.DataFrame:
    import fitsio

    with fitsio.FITS(path) as fits_file:
        data_chunk = fits_file[fits_hdu].read(rows=range(start, stop))

    df_dict = {}
    for name in data_chunk.dtype.names or ():
        column = data_chunk[name]
        if column.ndim == 1:
            values = np.asarray(column)
            if values.dtype.byteorder not in ("=", "|"):
                values = values.astype(values.dtype.newbyteorder("="), copy=False)
            df_dict[name] = values
        else:
            df_dict[name] = pd.Series([np.asarray(value).tolist() for value in column], dtype="object")
    return pd.DataFrame(df_dict)


def _schema_for_path(path: Path, suffix: str, config: dict[str, Any]) -> list[str]:
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    if suffix in HEADERLESS_SUFFIXES:
        detected_columns = pd.read_csv(path, sep=r"\s+", comment="#", header=None, nrows=1).shape[1]
        column_names = config.get("column_names")
        if not column_names:
            msg = (
                "Headerless whitespace inputs require column_names in prepare config "
                "or --column-name/--column-names in the CLI."
            )
            raise PrepareError(msg)
        if len(column_names) != detected_columns:
            msg = f"column_names has {len(column_names)} entries, but {path} has {detected_columns} columns."
            raise PrepareError(msg)
        return list(column_names)
    if suffix in PARQUET_SUFFIXES:
        import pyarrow.parquet as pq

        return [str(name) for name in pq.read_schema(path).names]
    if suffix in FITS_SUFFIXES:
        import fitsio

        with fitsio.FITS(path) as fits_file:
            return list(fits_file[int(config.get("fits_hdu", 1))].get_colnames())
    raise PrepareError(f"Unsupported prepare input format: {suffix or path}")


def _parquet_arrow_schema(path: Path) -> Any:
    import pyarrow.parquet as pq

    return pq.read_schema(path)


def _validate_matching_schemas(paths: list[Path], suffix: str, config: dict[str, Any]) -> list[str]:
    schemas = {path: _schema_for_path(path, suffix, config) for path in paths}
    first_path = paths[0]
    first_schema = schemas[first_path]
    mismatches = {path: schema for path, schema in schemas.items() if schema != first_schema}
    if not mismatches:
        return first_schema

    if _schema_policy(config) == "union":
        return _union_schema(list(schemas.values()))

    lines = ["Multi-file inputs must describe one logical catalog with the same schema."]
    lines.append(f"{first_path}: {', '.join(first_schema)}")
    for path, schema in mismatches.items():
        lines.append(f"{path}: {', '.join(schema)}")
    raise PrepareError("\n".join(lines))


def _fits_effective_chunk_size(configured_chunk_size: int, row_size_bytes: int, target_bytes: int) -> int:
    if row_size_bytes <= 0 or target_bytes <= 0:
        return configured_chunk_size
    target_rows = max(int(target_bytes // row_size_bytes), 1)
    return max(min(configured_chunk_size, target_rows), 1)


def _fits_to_dask_dataframe(
    path: Path, fits_hdu: int, chunk_size: int, target_partition_bytes: int
) -> tuple[Any, int, int, list[str]]:
    import dask.dataframe as dd
    import fitsio

    with fitsio.FITS(path) as fits_file:
        hdu = fits_file[fits_hdu]
        n_rows = int(hdu.get_nrows())
        colnames = list(hdu.get_colnames())
        row_size_bytes = int(hdu.get_rec_dtype()[0].itemsize)

    if n_rows == 0:
        meta = pd.DataFrame({name: pd.Series(dtype="object") for name in colnames})
        return dd.from_pandas(meta, npartitions=1), 1, n_rows, colnames

    chunk_size = _fits_effective_chunk_size(chunk_size, row_size_bytes, target_partition_bytes)
    first_stop = min(chunk_size, n_rows)
    meta = _read_fits_chunk(str(path), fits_hdu, 0, first_stop).iloc[:0]
    chunks = [
        delayed(_read_fits_chunk)(str(path), fits_hdu, start, min(start + chunk_size, n_rows))
        for start in range(0, n_rows, chunk_size)
    ]
    dask_chunks = [dd.from_delayed(chunk, meta=meta) for chunk in chunks]
    return dd.concat(dask_chunks), len(dask_chunks), n_rows, colnames


def _fits_paths_to_dask_dataframe(
    paths: list[Path], fits_hdu: int, chunk_size: int, target_partition_bytes: int
) -> tuple[Any, int]:
    import dask.dataframe as dd

    dataframes = []
    n_partitions = 0
    for path in paths:
        df, path_partitions, _n_rows, _colnames = _fits_to_dask_dataframe(
            path, fits_hdu, chunk_size, target_partition_bytes
        )
        dataframes.append(df)
        n_partitions += path_partitions
    if len(dataframes) == 1:
        return dataframes[0], n_partitions
    return dd.concat(dataframes, join="outer"), n_partitions


def _tabular_to_dask_dataframe(paths: list[Path], config: dict[str, Any]) -> tuple[Any, str]:
    import dask.dataframe as dd

    suffix = _data_suffix(paths[0])
    path_strings = [str(path) for path in paths]
    if suffix in {".csv", ".txt"}:
        if len(paths) > 1 and _schema_policy(config) == "union":
            frames = [dd.read_csv(str(path), blocksize=_target_partition_size(config)) for path in paths]
            return dd.concat(frames, join="outer"), suffix.lstrip(".")
        return dd.read_csv(path_strings, blocksize=_target_partition_size(config)), suffix.lstrip(".")
    if suffix in HEADERLESS_SUFFIXES:
        column_names = config.get("column_names")
        if not column_names:
            msg = (
                "Headerless whitespace inputs require column_names in prepare config "
                "or --column-name/--column-names in the CLI."
            )
            raise PrepareError(msg)
        return (
            dd.read_csv(
                path_strings,
                sep=r"\s+",
                comment="#",
                header=None,
                names=list(column_names),
                blocksize=_target_partition_size(config),
            ),
            suffix.lstrip("."),
        )
    if suffix in PARQUET_SUFFIXES:
        if len(paths) > 1 and _schema_policy(config) == "union":
            frames = [
                dd.read_parquet(
                    str(path),
                    split_row_groups=True,
                    blocksize=_target_partition_size(config),
                    aggregate_files=False,
                )
                for path in paths
            ]
            return dd.concat(frames, join="outer"), "parquet"
        return (
            dd.read_parquet(
                path_strings,
                split_row_groups=True,
                blocksize=_target_partition_size(config),
                aggregate_files=False,
            ),
            "parquet",
        )
    raise PrepareError(f"Unsupported prepare input format: {suffix or paths[0]}")


def _parquet_coordinate_dask_dataframe(path: Path, config: dict[str, Any]) -> Any:
    import dask.dataframe as dd

    ra_column, dec_column = _hats_ra_dec_columns(config)
    return dd.read_parquet(
        str(path),
        columns=[ra_column, dec_column],
        split_row_groups=True,
        blocksize=_target_partition_size(config),
        aggregate_files=False,
    )


def _write_dask_parquet(
    df: Any,
    output_dir: Path,
    prefix: str,
    overwrite: bool,
    output_mode: str,
    schema: Any | None = None,
) -> int:
    if output_mode == "single":
        df = df.repartition(npartitions=1)
    n_parts = int(df.npartitions)
    n_digits = max(len(str(n_parts)), 1)
    parquet_kwargs = {"schema": schema} if schema is not None else {}
    df.to_parquet(
        output_dir,
        engine="pyarrow",
        write_index=False,
        overwrite=overwrite,
        name_function=lambda index: f"{prefix}-part{index:0{n_digits}d}.parquet",
        **parquet_kwargs,
    )
    return n_parts


def _write_small_input(path: Path, output_dir: Path, config: dict[str, Any]) -> int:
    df = read_table(
        path,
        fits_hdu=int(config.get("fits_hdu", 1)),
        column_names=config.get("column_names"),
        dask_threshold_bytes=None,
    )
    output_path = output_dir / f"{path.stem}-part0.parquet"
    df.to_parquet(output_path, index=False)
    return 1


def _write_small_inputs(paths: list[Path], suffix: str, output_dir: Path, config: dict[str, Any]) -> int:
    df = _small_inputs_dataframe(paths, suffix, config)
    prefix = str(config.get("part_prefix") or paths[0].stem)
    df.to_parquet(output_dir / f"{prefix}-part0.parquet", index=False)
    return 1


def _small_input_dataframe(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    return read_table(
        path,
        fits_hdu=int(config.get("fits_hdu", 1)),
        column_names=config.get("column_names"),
        dask_threshold_bytes=None,
    )


def _small_inputs_dataframe(paths: list[Path], suffix: str, config: dict[str, Any]) -> pd.DataFrame:
    if len(paths) > 1:
        _validate_matching_schemas(paths, suffix, config)
    frames = [_small_input_dataframe(path, config) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else frames[0]


def _write_small_hats_inputs(paths: list[Path], suffix: str, output_dir: Path, config: dict[str, Any]) -> int:
    ra_column, dec_column = _hats_ra_dec_columns(config)
    df = _small_inputs_dataframe(paths, suffix, config)
    _validate_hats_coordinate_columns(df, config)

    return write_hats_from_dataframe(
        df,
        output_dir,
        config,
        ra_column=ra_column,
        dec_column=dec_column,
        partition_bytes=_target_partition_size_bytes(config),
        error_cls=PrepareError,
    )


def _run_hats_import_from_parquet(
    parquet_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    client: Any,
    cluster_config: dict[str, Any],
) -> None:
    ra_column, dec_column = _hats_ra_dec_columns(config)
    run_hats_import_from_parquet(
        parquet_dir,
        output_dir,
        config,
        client,
        ra_column=ra_column,
        dec_column=dec_column,
        cluster_config=cluster_config,
        error_cls=PrepareError,
    )


def _resolve_output_mode(config: dict[str, Any], is_single_small_file: bool, is_multi_file: bool) -> str:
    output_mode = _output_mode(config)
    if output_mode == "auto":
        return "single" if is_single_small_file and not is_multi_file else "partitioned"
    if (
        output_mode == "single"
        and not is_single_small_file
        and not bool(config.get("allow_large_single_output", False))
    ):
        msg = (
            "output_mode='single' can concentrate a large or multi-file catalog into one partition. "
            "Set allow_large_single_output: true if this is intentional."
        )
        raise PrepareError(msg)
    return output_mode


def _dask_logs_dir(cluster_config: dict[str, Any], output_dir: Path) -> Path | None:
    logs_dir = cluster_config.get("logs_dir")
    if logs_dir:
        return Path(logs_dir)
    if cluster_config.get("name") == "slurm":
        return output_dir / "logs"
    return None


def prepare_catalog(config: dict[str, Any]) -> Path:
    """Prepare catalog input as a partitioned Parquet or HATS dataset."""
    started_at = time.monotonic()
    input_paths = _as_paths(config.get("input_files") or [config["input_file"]])
    output_dir = Path(config["output_dir"])
    overwrite = bool(config.get("overwrite", False))
    threshold_bytes = _large_file_threshold_bytes(config)
    chunk_size = int(config.get("chunk_size_rows", DEFAULT_CHUNK_SIZE_ROWS))
    prefix = str(config.get("part_prefix") or input_paths[0].stem)
    output_format = _output_format(config)
    _schema_policy(config)
    input_size = _total_input_size(input_paths)
    LOGGER.info(
        "Starting prepare: inputs=%d, size=%.2f GB, output_format=%s, output=%s",
        len(input_paths),
        input_size / (1024**3),
        output_format,
        output_dir,
    )

    if output_format == "hats":
        _hats_ra_dec_columns(config)
        _hats_margin_threshold(config)
        _hats_catalog_name(output_dir, config)
        _hats_sort_columns(config)

    for path in input_paths:
        if is_hats_input(path):
            raise PrepareError(
                "Input is already a HATS catalog or collection. HATS is supported directly by "
                "inspect and curate, so prepare is not required for this input."
            )

    for path in input_paths:
        _check_large_compressed(path, threshold_bytes)

    if len(input_paths) == 1 and input_paths[0].is_dir():
        if not any(input_paths[0].glob("*.parquet")) and not (input_paths[0] / "_metadata").exists():
            raise PrepareError(f"Unsupported directory input: {input_paths[0]}. Expected a Parquet dataset.")
        if output_format == "parquet":
            LOGGER.info("Input is already a Parquet dataset; no preparation is required")
            LOGGER.info("Prepare completed in %.1f seconds", time.monotonic() - started_at)
            return input_paths[0]
        input_format = "parquet"
        _prepare_output_dir(output_dir, overwrite=overwrite)
        cluster_config = dask_cluster_config(config)
        LOGGER.info("Setting up Dask cluster for Parquet-to-HATS conversion")
        with dask_client_context(
            cluster_config, logs_dir=_dask_logs_dir(cluster_config, output_dir)
        ) as client:
            LOGGER.info("Validating HATS coordinate columns")
            _validate_hats_coordinate_columns(
                _parquet_coordinate_dask_dataframe(input_paths[0], config), config
            )
            LOGGER.info("Converting Parquet dataset to HATS")
            _run_hats_import_from_parquet(input_paths[0], output_dir, config, client, cluster_config)
        LOGGER.info("Writing preparation manifest")
        _write_manifest(
            output_dir,
            input_paths,
            input_format,
            config,
            n_partitions=0,
            output_mode="hats",
        )
        LOGGER.info("Prepare completed in %.1f seconds", time.monotonic() - started_at)
        return output_dir

    suffixes = {_data_suffix(path) for path in input_paths}
    if len(suffixes) != 1:
        raise PrepareError("All input files must have the same format before prepare can partition them.")
    suffix = suffixes.pop()
    input_format = suffix.lstrip(".")
    if len(input_paths) > 1:
        _validate_matching_schemas(input_paths, suffix, config)

    is_small_input = _total_input_size(input_paths) < threshold_bytes
    is_single_small_file = len(input_paths) == 1 and is_small_input
    output_mode = _resolve_output_mode(
        config,
        is_single_small_file=is_single_small_file,
        is_multi_file=len(input_paths) > 1,
    )
    LOGGER.info("Detected input format=%s; resolved output mode=%s", input_format, output_mode)

    _prepare_output_dir(output_dir, overwrite=overwrite)

    if output_format == "hats" and is_small_input:
        cluster_config = dask_cluster_config(config)
        LOGGER.info("Setting up Dask cluster for in-memory HATS output")
        with dask_client_context(cluster_config, logs_dir=_dask_logs_dir(cluster_config, output_dir)):
            LOGGER.info("Reading input and writing HATS catalog")
            n_partitions = _write_small_hats_inputs(input_paths, suffix, output_dir, config)
        LOGGER.info("Writing preparation manifest")
        _write_manifest(
            output_dir,
            input_paths,
            input_format,
            config,
            n_partitions=n_partitions,
            output_mode="hats",
        )
        LOGGER.info("Prepare completed in %.1f seconds", time.monotonic() - started_at)
        return output_dir

    if is_small_input and output_mode == "single" and output_format == "parquet":
        LOGGER.info("Reading input and writing single Parquet file")
        if is_single_small_file:
            n_partitions = _write_small_input(input_paths[0], output_dir, config)
        else:
            n_partitions = _write_small_inputs(input_paths, suffix, output_dir, config)
        LOGGER.info("Writing preparation manifest")
        _write_manifest(
            output_dir,
            input_paths,
            input_format,
            config,
            n_partitions=n_partitions,
            output_mode=output_mode,
        )
        LOGGER.info("Prepare completed in %.1f seconds", time.monotonic() - started_at)
        return output_dir

    parquet_output_dir = output_dir
    if output_format == "hats":
        parquet_output_dir = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}-parquet-", dir=str(output_dir.parent))
        )

    if suffix in FITS_SUFFIXES:
        _validate_matching_schemas(input_paths, suffix, config)
        LOGGER.info("Building partitioned Dask dataframe from FITS input")
        df, n_partitions = _fits_paths_to_dask_dataframe(
            input_paths,
            int(config.get("fits_hdu", 1)),
            chunk_size,
            _target_partition_size_bytes(config),
        )
        write_schema = None
    else:
        LOGGER.info("Opening tabular input as a Dask dataframe")
        df, input_format = _tabular_to_dask_dataframe(input_paths, config)
        n_partitions = int(df.npartitions)
        write_schema = _parquet_arrow_schema(input_paths[0]) if suffix in PARQUET_SUFFIXES else None

    cluster_config = dask_cluster_config(config)
    logs_dir = _dask_logs_dir(cluster_config, output_dir)
    try:
        LOGGER.info("Setting up Dask cluster for distributed preparation")
        with dask_client_context(cluster_config, logs_dir=logs_dir) as client:
            if output_format == "hats":
                LOGGER.info("Validating HATS coordinate columns")
                _validate_hats_coordinate_columns(df, config)
            LOGGER.info("Writing partitioned Parquet dataset")
            n_written = _write_dask_parquet(
                df,
                parquet_output_dir,
                prefix=prefix,
                overwrite=True,
                output_mode=output_mode,
                schema=write_schema,
            )
            if output_format == "hats":
                LOGGER.info("Converting partitioned Parquet dataset to HATS")
                _run_hats_import_from_parquet(parquet_output_dir, output_dir, config, client, cluster_config)
    finally:
        if output_format == "hats" and parquet_output_dir.exists():
            LOGGER.debug("Removing temporary Parquet dataset: %s", parquet_output_dir)
            shutil.rmtree(parquet_output_dir)

    LOGGER.info("Writing preparation manifest")
    _write_manifest(
        output_dir,
        input_paths,
        input_format,
        config,
        n_partitions=n_written or n_partitions,
        output_mode="hats" if output_format == "hats" else output_mode,
    )
    LOGGER.info("Prepare completed in %.1f seconds", time.monotonic() - started_at)
    return output_dir
