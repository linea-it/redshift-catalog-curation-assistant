import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from dask import delayed

from ..executor import dask_client_context, dask_cluster_config
from ..io import DEFAULT_DASK_THRESHOLD_BYTES, read_table

COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".zip"}
HEADERLESS_SUFFIXES = {".dat", ".idz"}
PARQUET_SUFFIXES = {".parquet", ".pq"}
FITS_SUFFIXES = {".fits", ".fit", ".fts"}
TEXT_SUFFIXES = {".csv", ".txt", ".dat", ".idz"}
DEFAULT_CHUNK_SIZE_ROWS = 200_000
DEFAULT_TARGET_PARTITION_SIZE_MB = 100
OUTPUT_MODES = {"auto", "single", "partitioned"}


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


def _output_mode(config: dict[str, Any]) -> str:
    output_mode = str(config.get("output_mode", "auto")).lower()
    if output_mode not in OUTPUT_MODES:
        raise PrepareError("output_mode must be one of: auto, single, partitioned.")
    return output_mode


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


def _as_paths(paths: list[str | Path]) -> list[Path]:
    return [Path(path) for path in paths]


def _write_manifest(
    output_dir: Path,
    input_paths: list[Path],
    input_format: str,
    config: dict[str, Any],
    n_partitions: int,
    output_mode: str,
) -> None:
    manifest = {
        "source_paths": [str(path) for path in input_paths],
        "source_format": input_format,
        "partition_format": "parquet",
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


def _validate_matching_schemas(paths: list[Path], suffix: str, config: dict[str, Any]) -> list[str]:
    schemas = {path: _schema_for_path(path, suffix, config) for path in paths}
    first_path = paths[0]
    first_schema = schemas[first_path]
    mismatches = {path: schema for path, schema in schemas.items() if schema != first_schema}
    if not mismatches:
        return first_schema

    lines = ["Multi-file inputs must describe one logical catalog with the same schema."]
    lines.append(f"{first_path}: {', '.join(first_schema)}")
    for path, schema in mismatches.items():
        lines.append(f"{path}: {', '.join(schema)}")
    raise PrepareError("\n".join(lines))


def _fits_to_dask_dataframe(path: Path, fits_hdu: int, chunk_size: int) -> tuple[Any, int, int, list[str]]:
    import dask.dataframe as dd
    import fitsio

    with fitsio.FITS(path) as fits_file:
        hdu = fits_file[fits_hdu]
        n_rows = int(hdu.get_nrows())
        colnames = list(hdu.get_colnames())

    if n_rows == 0:
        meta = pd.DataFrame({name: pd.Series(dtype="object") for name in colnames})
        return dd.from_pandas(meta, npartitions=1), 1, n_rows, colnames

    first_stop = min(chunk_size, n_rows)
    meta = _read_fits_chunk(str(path), fits_hdu, 0, first_stop).iloc[:0]
    chunks = [
        delayed(_read_fits_chunk)(str(path), fits_hdu, start, min(start + chunk_size, n_rows))
        for start in range(0, n_rows, chunk_size)
    ]
    dask_chunks = [dd.from_delayed(chunk, meta=meta) for chunk in chunks]
    return dd.concat(dask_chunks), len(dask_chunks), n_rows, colnames


def _fits_paths_to_dask_dataframe(paths: list[Path], fits_hdu: int, chunk_size: int) -> tuple[Any, int]:
    import dask.dataframe as dd

    dataframes = []
    n_partitions = 0
    for path in paths:
        df, path_partitions, _n_rows, _colnames = _fits_to_dask_dataframe(path, fits_hdu, chunk_size)
        dataframes.append(df)
        n_partitions += path_partitions
    if len(dataframes) == 1:
        return dataframes[0], n_partitions
    return dd.concat(dataframes), n_partitions


def _tabular_to_dask_dataframe(paths: list[Path], config: dict[str, Any]) -> tuple[Any, str]:
    import dask.dataframe as dd

    suffix = _data_suffix(paths[0])
    path_strings = [str(path) for path in paths]
    if suffix in {".csv", ".txt"}:
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
        return dd.read_parquet(path_strings), "parquet"
    raise PrepareError(f"Unsupported prepare input format: {suffix or paths[0]}")


def _write_dask_parquet(df: Any, output_dir: Path, prefix: str, overwrite: bool, output_mode: str) -> int:
    if output_mode == "single":
        df = df.repartition(npartitions=1)
    n_parts = int(df.npartitions)
    n_digits = max(len(str(n_parts)), 1)
    df.to_parquet(
        output_dir,
        engine="pyarrow",
        write_index=False,
        overwrite=overwrite,
        name_function=lambda index: f"{prefix}-part{index:0{n_digits}d}.parquet",
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


def prepare_catalog(config: dict[str, Any]) -> Path:
    """Prepare catalog input as a partitioned Parquet dataset."""
    input_paths = _as_paths(config.get("input_files") or [config["input_file"]])
    output_dir = Path(config["output_dir"])
    overwrite = bool(config.get("overwrite", False))
    threshold_bytes = _large_file_threshold_bytes(config)
    chunk_size = int(config.get("chunk_size_rows", DEFAULT_CHUNK_SIZE_ROWS))
    prefix = str(config.get("part_prefix") or input_paths[0].stem)

    for path in input_paths:
        _check_large_compressed(path, threshold_bytes)

    if len(input_paths) == 1 and input_paths[0].is_dir():
        if not any(input_paths[0].glob("*.parquet")) and not (input_paths[0] / "_metadata").exists():
            raise PrepareError(f"Unsupported directory input: {input_paths[0]}. Expected a Parquet dataset.")
        return input_paths[0]

    suffixes = {_data_suffix(path) for path in input_paths}
    if len(suffixes) != 1:
        raise PrepareError("All input files must have the same format before prepare can partition them.")
    suffix = suffixes.pop()
    input_format = suffix.lstrip(".")
    if len(input_paths) > 1:
        _validate_matching_schemas(input_paths, suffix, config)

    is_single_small_file = len(input_paths) == 1 and input_paths[0].stat().st_size < threshold_bytes
    output_mode = _resolve_output_mode(
        config,
        is_single_small_file=is_single_small_file,
        is_multi_file=len(input_paths) > 1,
    )

    _prepare_output_dir(output_dir, overwrite=overwrite)

    if is_single_small_file and output_mode == "single":
        n_partitions = _write_small_input(input_paths[0], output_dir, config)
        _write_manifest(
            output_dir,
            input_paths,
            input_format,
            config,
            n_partitions=n_partitions,
            output_mode=output_mode,
        )
        return output_dir

    if suffix in FITS_SUFFIXES:
        df, n_partitions = _fits_paths_to_dask_dataframe(
            input_paths, int(config.get("fits_hdu", 1)), chunk_size
        )
    else:
        df, input_format = _tabular_to_dask_dataframe(input_paths, config)
        if output_mode != "single" and (
            suffix in PARQUET_SUFFIXES or any(path.stat().st_size >= threshold_bytes for path in input_paths)
        ):
            df = df.repartition(partition_size=_target_partition_size(config))
        n_partitions = int(df.npartitions)

    cluster_config = dask_cluster_config(config)
    logs_dir = Path(cluster_config["logs_dir"]) if cluster_config.get("logs_dir") else None
    with dask_client_context(cluster_config, logs_dir=logs_dir):
        n_written = _write_dask_parquet(
            df,
            output_dir,
            prefix=prefix,
            overwrite=True,
            output_mode=output_mode,
        )

    _write_manifest(
        output_dir,
        input_paths,
        input_format,
        config,
        n_partitions=n_written or n_partitions,
        output_mode=output_mode,
    )
    return output_dir
