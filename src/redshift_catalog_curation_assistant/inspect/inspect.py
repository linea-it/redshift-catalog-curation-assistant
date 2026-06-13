import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..io import DEFAULT_DASK_THRESHOLD_BYTES

PARQUET_SUFFIXES = {".parquet", ".pq"}
FITS_SUFFIXES = {".fits", ".fit", ".fts"}
COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".zip"}
SAMPLE_MAX_COLUMNS = 100
PARQUET_STATS_BATCH_SIZE = 50
FITS_STATS_BATCH_SIZE = 8
StatsValue = float | int | None
STATS_MODES = {"candidates", "all", "none"}

PATTERNS = {
    "ra": [
        r"^ra$",
        r"(^|_)ra_deg($|_)",
        r"(^|_)target_ra($|_)",
        r"(^|_)obsra($|_)",
        r"(^|_)ra_j2000($|_)",
        r"^alpha",
        r"(^|_)right_ascension($|_)",
    ],
    "dec": [
        r"^dec$",
        r"dec_deg",
        r"target_dec",
        r"obsdec",
        r"dec_j2000",
        r"^delta",
        r"(^|_)declination($|_)",
    ],
    "redshift": [
        r"^z$",
        r"redshift",
        r"zspec",
        r"z_spec",
        r"zphot",
        r"z_phot",
        r"best_z",
        r"z_ml",
        r"z_helio",
    ],
    "quality": [
        r"quality",
        r"qual",
        r"qop",
        r"(^|_)flag($|_)",
        r"zflag",
        r"zwarn",
        r"zwarning",
        r"z_flag",
        r"vi_quality",
        r"confidence",
    ],
    "redshift_error": [
        r"(^|_)z_err($|_)",
        r"^zerr$",
        r"(^|_)err_z($|_)",
        r"(^|_)z_error($|_)",
        r"(^|_)sigma_z($|_)",
    ],
    "id": [
        r"^id$",
        r"(^|_)objectid($|_)",
        r"(^|_)object_id($|_)",
        r"(^|_)targetid($|_)",
        r"(^|_)specid($|_)",
        r"(^|_)catalogid($|_)",
        r"^tileid$",
    ],
    "object_type": [
        r"(^|_)class($|_)",
        r"(^|_)subclass($|_)",
        r"(^|_)objtype($|_)",
        r"(^|_)spectype($|_)",
        r"(^|_)subtype($|_)",
        r"(^|_)eta_type($|_)",
        r"^type$",
    ],
}


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def candidate_columns(columns: list[str], patterns: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return columns matching each semantic category."""
    matches: dict[str, list[str]] = {k: [] for k in patterns}
    for col in columns:
        for cat, pats in patterns.items():
            for p in pats:
                if re.search(p, col, flags=re.IGNORECASE):
                    matches[cat].append(col)
                    break
    return matches


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _flatten_candidates(candidates: dict[str, list[str]]) -> list[str]:
    return _ordered_unique(col for cols in candidates.values() for col in cols)


def _sample_columns(columns: list[str], candidate_cols: list[str], config: dict[str, Any]) -> list[str]:
    max_columns = int(config.get("sample_max_columns", SAMPLE_MAX_COLUMNS))
    if max_columns <= 0:
        return []
    return _ordered_unique([*candidate_cols, *columns])[:max_columns]


def _column_selection(config: dict[str, Any]) -> list[str] | None:
    selection = config.get("column_selection")
    if selection is None:
        return None
    if not isinstance(selection, list | tuple) or not all(isinstance(value, str) for value in selection):
        msg = (
            "column_selection must be a list of column names. Set 'column_selection' in YAML, "
            "or pass --column-selection/--column-selection-list in the CLI."
        )
        raise ValueError(msg)
    return _ordered_unique(value.strip() for value in selection if value.strip())


def _selected_report_columns(columns: list[str], config: dict[str, Any]) -> tuple[list[str], list[str]]:
    selection = _column_selection(config)
    if selection is None:
        return columns, []

    available = set(columns)
    missing = [column for column in selection if column not in available]
    if missing:
        msg = (
            "column_selection contains columns not present in input: "
            + ", ".join(missing)
            + ". Update 'column_selection' in YAML, or --column-selection/--column-selection-list in the CLI."
        )
        raise ValueError(msg)

    return selection, [
        f"Report was limited to {len(selection)} selected columns out of {len(columns)} available columns."
    ]


def _stats_mode(config: dict[str, Any]) -> str:
    stats_mode = str(config.get("stats_mode", "candidates")).lower()
    if stats_mode not in STATS_MODES:
        msg = (
            "stats_mode must be one of: candidates, all, none. Set 'stats_mode' in YAML, "
            "or pass --stats-mode in the CLI."
        )
        raise ValueError(msg)
    return stats_mode


def _selected_stats_columns(
    columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    candidate_cols: list[str],
    config: dict[str, Any],
    input_kind: str,
) -> tuple[list[str], list[str], list[str]]:
    stats_mode = _stats_mode(config)

    if stats_mode == "none":
        return [], [], [f"{input_kind} statistics were skipped because stats_mode='none'."]

    if stats_mode == "all":
        return numeric_columns, categorical_columns, []

    selected = set(candidate_cols)
    selected_numeric = [col for col in numeric_columns if col in selected]
    selected_categorical = [col for col in categorical_columns if col in selected]
    warnings = []
    if not selected_numeric and not selected_categorical:
        warnings.append("No candidate columns were eligible for statistics.")
    return selected_numeric, selected_categorical, warnings


def _is_dask_dataframe(df: Any) -> bool:
    return df.__class__.__module__.startswith("dask.dataframe")


def _compute_if_needed(value: Any) -> Any:
    if hasattr(value, "compute"):
        return value.compute()
    return value


def _head(df: pd.DataFrame | Any, n_rows: int) -> pd.DataFrame:
    if _is_dask_dataframe(df):
        return df.head(n_rows, npartitions=-1)
    return df.head(n_rows)


def _native_float_values(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype=np.float64, na_value=np.nan)


def gather_stats(df: pd.DataFrame | Any) -> dict[str, dict[str, StatsValue]]:
    """Collect simple numeric statistics for a DataFrame."""
    stats: dict[str, dict[str, StatsValue]] = {}
    numeric = df.select_dtypes(include="number")
    if _is_dask_dataframe(df):
        if len(numeric.columns) == 0:
            return stats
        for col in numeric.columns:
            series = numeric[col]
            count = int(series.count().compute())
            total = int(series.size.compute())
            stats[col] = {
                "count": count,
                "null_count": total - count,
                "mean": float(series.mean().compute()) if count else None,
                "std": float(series.std().compute()) if count > 1 else None,
                "min": float(series.min().compute()) if count else None,
                "max": float(series.max().compute()) if count else None,
            }
        return stats

    for col in numeric.columns:
        values = _native_float_values(numeric[col])
        count = int(np.count_nonzero(~np.isnan(values)))
        total = int(len(values))
        stats[col] = {
            "count": count,
            "null_count": total - count,
            "mean": float(np.nanmean(values)) if count else None,
            "std": float(np.nanstd(values, ddof=1)) if count > 1 else None,
            "min": float(np.nanmin(values)) if count else None,
            "max": float(np.nanmax(values)) if count else None,
        }
    return stats


def gather_categorical_uniques(df: pd.DataFrame | Any, limit: int = 10) -> dict[str, list[str]]:
    """Collect bounded unique values for non-numeric columns."""
    uniques = {}
    categorical = df.select_dtypes(exclude="number")
    for col in categorical.columns:
        series = categorical[col].dropna()
        if not _is_dask_dataframe(df):
            series = series.apply(_fits_scalar_value)
            series = series[series.astype(str) != ""]
        values = _compute_if_needed(series.astype(str).unique())
        uniques[col] = values[:limit].tolist()
    return uniques


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


def _is_parquet_input(path: Path) -> bool:
    if path.is_dir():
        return (path / "_metadata").exists() or any(path.glob("*.parquet")) or any(path.glob("*.pq"))
    return _data_suffix(path) in PARQUET_SUFFIXES


def _is_fits_input(path: Path) -> bool:
    return path.is_file() and _data_suffix(path) in FITS_SUFFIXES


def _parquet_dataset(path: Path) -> Any:
    import pyarrow.dataset as ds

    return ds.dataset(path, format="parquet")


def _parquet_columns(schema: Any) -> list[str]:
    return list(schema.names)


def _parquet_dtypes(schema: Any) -> dict[str, str]:
    return {field.name: str(field.type) for field in schema}


def _parquet_numeric_columns(schema: Any) -> list[str]:
    import pyarrow.types as pat

    return [
        field.name
        for field in schema
        if pat.is_integer(field.type) or pat.is_floating(field.type) or pat.is_decimal(field.type)
    ]


def _parquet_categorical_columns(schema: Any) -> list[str]:
    import pyarrow.types as pat

    return [
        field.name
        for field in schema
        if pat.is_string(field.type)
        or pat.is_large_string(field.type)
        or pat.is_dictionary(field.type)
        or pat.is_boolean(field.type)
    ]


def _parquet_count_rows(dataset: Any) -> int:
    return int(dataset.count_rows())


def _parquet_to_pandas(dataset: Any, columns: list[str], limit: int | None = None) -> pd.DataFrame:
    table = dataset.to_table(columns=columns)
    if limit is not None:
        table = table.slice(0, limit)
    return table.to_pandas()


def _parquet_stats_columns(
    schema: Any, columns: list[str], candidate_cols: list[str], config: dict[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    return _selected_stats_columns(
        columns,
        numeric_columns=_parquet_numeric_columns(schema),
        categorical_columns=_parquet_categorical_columns(schema),
        candidate_cols=candidate_cols,
        config=config,
        input_kind="Parquet",
    )


def _parquet_numeric_stats(
    dataset: Any, columns: list[str], batch_size: int
) -> dict[str, dict[str, StatsValue]]:
    stats: dict[str, dict[str, StatsValue]] = {}
    if not columns:
        return stats

    for start in range(0, len(columns), batch_size):
        batch_columns = columns[start : start + batch_size]
        stats.update(gather_stats(_parquet_to_pandas(dataset, batch_columns)))
    return stats


def _parquet_categorical_uniques(
    dataset: Any, columns: list[str], batch_size: int, limit: int
) -> dict[str, list[str]]:
    uniques: dict[str, list[str]] = {}
    if not columns:
        return uniques

    for start in range(0, len(columns), batch_size):
        batch_columns = columns[start : start + batch_size]
        uniques.update(gather_categorical_uniques(_parquet_to_pandas(dataset, batch_columns), limit=limit))
    return uniques


def _build_parquet_report(input_path: Path, survey: str, config: dict[str, Any]) -> dict[str, Any]:
    dataset = _parquet_dataset(input_path)
    schema = dataset.schema
    all_columns = _parquet_columns(schema)
    columns, warnings = _selected_report_columns(all_columns, config)
    patterns = build_patterns(config)
    candidates = candidate_columns(columns, patterns)
    candidate_cols = _flatten_candidates(candidates)

    sample_columns = _sample_columns(columns, candidate_cols, config)
    if len(sample_columns) < len(columns):
        warnings.append(
            f"Parquet sample was limited to {len(sample_columns)} of {len(columns)} columns. "
            "Increase 'sample_max_columns' in YAML, or pass --sample-max-columns in the CLI."
        )
    sample = _parquet_to_pandas(dataset, sample_columns, limit=5) if sample_columns else pd.DataFrame()

    batch_size = int(config.get("parquet_stats_batch_size", PARQUET_STATS_BATCH_SIZE))
    if batch_size <= 0:
        msg = "parquet_stats_batch_size must be greater than zero. This option is available in YAML config."
        raise ValueError(msg)

    numeric_cols, categorical_cols, stats_warnings = _parquet_stats_columns(
        schema, columns, candidate_cols, config
    )
    numeric_cols = [column for column in numeric_cols if column in columns]
    categorical_cols = [column for column in categorical_cols if column in columns]
    warnings.extend(stats_warnings)
    dtypes = _parquet_dtypes(schema)

    return {
        "survey": survey,
        "input_file": str(input_path),
        "n_rows": _parquet_count_rows(dataset),
        "n_columns": len(all_columns),
        "n_columns_selected": len(columns),
        "columns": columns,
        "dtypes": {column: dtypes[column] for column in columns},
        "candidates": candidates,
        "numeric_stats": _parquet_numeric_stats(dataset, numeric_cols, batch_size),
        "categorical_uniques": _parquet_categorical_uniques(
            dataset,
            categorical_cols,
            batch_size=batch_size,
            limit=int(config.get("unique_limit", 10)),
        ),
        "sample": sample.to_dict(orient="records"),
        "warnings": warnings,
    }


def _fits_table_column_names(header: Any) -> list[str]:
    n_columns = int(header.get("TFIELDS", 0) or 0)
    return [str(header.get(f"TTYPE{index}", f"COL{index}")) for index in range(1, n_columns + 1)]


def _fits_table_formats(header: Any, columns: list[str]) -> dict[str, str]:
    return {column: str(header.get(f"TFORM{index}", "")) for index, column in enumerate(columns, start=1)}


def _fits_format_code(format_value: str) -> str:
    for char in reversed(format_value.strip().upper()):
        if char.isalpha():
            return char
    return ""


def _fits_numeric_columns(formats: dict[str, str]) -> list[str]:
    numeric_codes = {"B", "I", "J", "K", "E", "D", "C", "M"}
    return [column for column, fmt in formats.items() if _fits_format_code(fmt) in numeric_codes]


def _fits_categorical_columns(formats: dict[str, str]) -> list[str]:
    categorical_codes = {"A", "L"}
    return [column for column, fmt in formats.items() if _fits_format_code(fmt) in categorical_codes]


def _fits_scalar_columns(fits_hdu: Any, columns: list[str]) -> tuple[list[str], list[str]]:
    scalar_columns = []
    skipped_columns = []
    dtype = fits_hdu.get_rec_dtype()[0]
    for column in columns:
        field_dtype = dtype.fields[column][0]
        if field_dtype.shape == ():
            scalar_columns.append(column)
        else:
            skipped_columns.append(column)
    return scalar_columns, skipped_columns


def _fits_scalar_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _fits_sample(fits_hdu: Any, columns: list[str], n_rows: int = 5) -> list[dict[str, Any]]:
    if not columns:
        return []
    data = fits_hdu.read(columns=columns, rows=range(min(n_rows, fits_hdu.get_nrows())))
    sample = []
    for row_data in data:
        row = {}
        for column in columns:
            row[column] = _fits_scalar_value(row_data[column])
        sample.append(row)
    return sample


def _fits_numeric_stats(
    fits_hdu: Any, columns: list[str], batch_size: int
) -> dict[str, dict[str, StatsValue]]:
    stats: dict[str, dict[str, StatsValue]] = {}
    for start in range(0, len(columns), batch_size):
        batch_columns = columns[start : start + batch_size]
        data = fits_hdu.read(columns=batch_columns)
        batch_df = pd.DataFrame({column: np.asarray(data[column]) for column in batch_columns})
        stats.update(gather_stats(batch_df))
    return stats


def _fits_categorical_uniques(
    fits_hdu: Any, columns: list[str], batch_size: int, limit: int
) -> dict[str, list[str]]:
    uniques: dict[str, list[str]] = {}
    for start in range(0, len(columns), batch_size):
        batch_columns = columns[start : start + batch_size]
        data = fits_hdu.read(columns=batch_columns)
        batch_df = pd.DataFrame({column: np.asarray(data[column]) for column in batch_columns})
        uniques.update(gather_categorical_uniques(batch_df, limit=limit))
    return uniques


def _build_fits_report(input_path: Path, survey: str, config: dict[str, Any]) -> dict[str, Any]:
    import fitsio

    from ..fits.fits import _check_compressed_fits_size

    _check_compressed_fits_size(input_path)
    fits_hdu = int(config.get("fits_hdu", 1))

    with fitsio.FITS(input_path) as fits_file:
        hdu = fits_file[fits_hdu]
        header = hdu.read_header()
        all_columns = _fits_table_column_names(header)
        all_formats = _fits_table_formats(header, all_columns)
        columns, warnings = _selected_report_columns(all_columns, config)
        formats = {column: all_formats[column] for column in columns}
        patterns = build_patterns(config)
        candidates = candidate_columns(columns, patterns)
        candidate_cols = _flatten_candidates(candidates)

        sample_columns = _sample_columns(columns, candidate_cols, config)
        if len(sample_columns) < len(columns):
            warnings.append(
                f"FITS sample was limited to {len(sample_columns)} of {len(columns)} columns. "
                "Increase 'sample_max_columns' in YAML, or pass --sample-max-columns in the CLI."
            )

        numeric_cols, categorical_cols, stats_warnings = _selected_stats_columns(
            columns,
            numeric_columns=_fits_numeric_columns(formats),
            categorical_columns=_fits_categorical_columns(formats),
            candidate_cols=candidate_cols,
            config=config,
            input_kind="FITS",
        )
        warnings.extend(stats_warnings)

        sample = _fits_sample(hdu, sample_columns)

        scalar_numeric_cols, skipped_numeric_cols = (
            _fits_scalar_columns(hdu, numeric_cols) if numeric_cols else ([], [])
        )
        scalar_categorical_cols, skipped_categorical_cols = (
            _fits_scalar_columns(hdu, categorical_cols) if categorical_cols else ([], [])
        )
        skipped_cols = [*skipped_numeric_cols, *skipped_categorical_cols]
        if skipped_cols:
            warnings.append("FITS vector columns were skipped for statistics: " + ", ".join(skipped_cols))

        batch_size = int(config.get("fits_stats_batch_size", FITS_STATS_BATCH_SIZE))
        if batch_size <= 0:
            msg = "fits_stats_batch_size must be greater than zero. This option is available in YAML config."
            raise ValueError(msg)

        return {
            "survey": survey,
            "input_file": str(input_path),
            "n_rows": int(header.get("NAXIS2", 0) or 0),
            "n_columns": len(all_columns),
            "n_columns_selected": len(columns),
            "columns": columns,
            "dtypes": formats,
            "candidates": candidates,
            "numeric_stats": _fits_numeric_stats(hdu, scalar_numeric_cols, batch_size),
            "categorical_uniques": _fits_categorical_uniques(
                hdu,
                scalar_categorical_cols,
                batch_size=batch_size,
                limit=int(config.get("unique_limit", 10)),
            ),
            "sample": sample,
            "warnings": warnings,
        }


def build_patterns(config: dict[str, Any]) -> dict[str, list[str]]:
    """Merge default column patterns with optional config patterns."""
    patterns = {category: values.copy() for category, values in PATTERNS.items()}
    configured = config.get("column_patterns", {})
    for category, values in configured.items():
        patterns[category] = list(values)
    return patterns


def json_default(value: Any) -> Any:
    """Convert common catalog scalar values to JSON-compatible values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _dask_threshold_bytes(config: dict[str, Any]) -> int | None:
    threshold_mb = config.get("dask_threshold_mb", DEFAULT_DASK_THRESHOLD_BYTES / (1024 * 1024))
    if threshold_mb is None:
        return None
    threshold_mb = float(threshold_mb)
    if threshold_mb <= 0:
        return None
    return int(threshold_mb * 1024 * 1024)


def _prepare_command_hint(input_path: Path) -> str:
    output_dir = Path("reports") / "prepared" / f"{input_path.stem}.parquet"
    return f"redshift-curator prepare --path {input_path} --output-dir {output_dir} --overwrite"


def _check_raw_input_policy(input_path: Path, config: dict[str, Any]) -> None:
    if not input_path.is_file() or _is_parquet_input(input_path):
        return

    threshold_bytes = _dask_threshold_bytes(config) or DEFAULT_DASK_THRESHOLD_BYTES
    if input_path.stat().st_size < threshold_bytes:
        return
    if bool(config.get("allow_large_raw_inspect", False)):
        return

    if _is_compressed(input_path):
        msg = (
            f"Input file is larger than the configured inspect threshold and is compressed: {input_path}\n\n"
            "Inspect does not process large compressed raw catalogs directly. Decompress the file first, "
            "then run prepare to materialize a Parquet dataset before inspection.\n\n"
            f"Suggested decompression command:\n  {_decompress_command(input_path)}\n\n"
            "After decompression, run:\n"
            f"  {_prepare_command_hint(Path(str(input_path).removesuffix(input_path.suffix)))}\n"
            "Then inspect the prepared Parquet directory."
        )
        raise ValueError(msg)

    input_format = "FITS" if _is_fits_input(input_path) else "raw"
    msg = (
        f"{input_format} input is larger than the configured inspect threshold: {input_path}\n\n"
        "Run prepare first so the catalog is normalized to a Parquet dataset without loading the whole "
        "raw file into memory during inspection.\n\n"
        "Suggested command:\n"
        f"  {_prepare_command_hint(input_path)}\n\n"
        "Then inspect the prepared Parquet directory. If direct raw inspection is intentional, set "
        "allow_large_raw_inspect: true in YAML."
    )
    raise ValueError(msg)


def _dask_logs_dir(cluster_config: dict[str, Any], outdir: Path) -> Path | None:
    logs_dir = cluster_config.get("logs_dir")
    if logs_dir:
        return Path(logs_dir)
    if cluster_config.get("name") == "slurm":
        return outdir / "logs"
    return None


def _build_report(
    input_path: Path, survey: str, df: pd.DataFrame | Any, config: dict[str, Any]
) -> dict[str, Any]:
    patterns = build_patterns(config)
    all_columns = list(df.columns)
    columns, warnings = _selected_report_columns(all_columns, config)
    report_df = df[columns] if columns else df[[]]
    candidates = candidate_columns(columns, patterns)
    candidate_cols = _flatten_candidates(candidates)

    sample_columns = _sample_columns(columns, candidate_cols, config)
    if len(sample_columns) < len(columns):
        warnings.append(
            f"Sample was limited to {len(sample_columns)} of {len(columns)} columns. "
            "Increase 'sample_max_columns' in YAML, or pass --sample-max-columns in the CLI."
        )
    sample_input = report_df[sample_columns] if sample_columns else report_df[[]]
    sample = _head(sample_input, 5)
    sample = _compute_if_needed(sample)

    numeric_columns = list(report_df.select_dtypes(include="number").columns)
    categorical_columns = list(report_df.select_dtypes(exclude="number").columns)
    numeric_cols, categorical_cols, stats_warnings = _selected_stats_columns(
        columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        candidate_cols=candidate_cols,
        config=config,
        input_kind="Tabular",
    )
    warnings.extend(stats_warnings)

    stats_df = report_df[numeric_cols] if numeric_cols else report_df[[]]
    categorical_df = report_df[categorical_cols] if categorical_cols else report_df[[]]

    return {
        "survey": survey,
        "input_file": str(input_path),
        "n_rows": int(len(df)),
        "n_columns": int(len(all_columns)),
        "n_columns_selected": int(len(columns)),
        "columns": columns,
        "dtypes": {c: str(t) for c, t in report_df.dtypes.items()},
        "candidates": candidates,
        "numeric_stats": gather_stats(stats_df),
        "categorical_uniques": gather_categorical_uniques(
            categorical_df, limit=int(config.get("unique_limit", 10))
        ),
        "sample": sample.to_dict(orient="records"),
        "warnings": warnings,
    }


def _write_report(outdir: Path, survey: str, input_path: Path, report: dict[str, Any]) -> None:
    (outdir / "inspect_report.json").write_text(json.dumps(report, indent=2, default=json_default))
    md = [f"# Inspect report: {survey}\n"]
    md.append(f"Input file: {input_path}\n")
    md.append(
        f"Rows: {report['n_rows']}  Columns: {report['n_columns']}  "
        f"Selected columns: {report['n_columns_selected']}\n"
    )
    md.append("## Candidate columns by category\n")
    for k, v in report["candidates"].items():
        md.append(f"- **{k}**: {', '.join(v) if v else '—'}\n")
    (outdir / "inspect_report.md").write_text("\n".join(md))


def run_inspect_config(config: dict[str, Any]) -> Path:
    """Run catalog inspection from a loaded config and write reports."""
    cfg = config
    input_path = Path(cfg["input_file"])
    survey = cfg.get("survey_name", input_path.stem)
    outdir = Path("reports") / survey

    if _is_parquet_input(input_path):
        report = _build_parquet_report(input_path, survey, cfg)
        outdir.mkdir(parents=True, exist_ok=True)
        _write_report(outdir, survey, input_path, report)
        return outdir

    _check_raw_input_policy(input_path, cfg)

    if _is_fits_input(input_path):
        report = _build_fits_report(input_path, survey, cfg)
        outdir.mkdir(parents=True, exist_ok=True)
        _write_report(outdir, survey, input_path, report)
        return outdir

    from ..io import read_table

    df = read_table(
        input_path,
        fits_hdu=cfg.get("fits_hdu", 1),
        column_names=cfg.get("column_names"),
        dask_threshold_bytes=_dask_threshold_bytes(cfg),
    )

    if _is_dask_dataframe(df):
        from ..executor import dask_client_context, dask_cluster_config

        cluster_config = dask_cluster_config(cfg)
        with dask_client_context(cluster_config, logs_dir=_dask_logs_dir(cluster_config, outdir)):
            report = _build_report(input_path, survey, df, cfg)
    else:
        report = _build_report(input_path, survey, df, cfg)

    outdir.mkdir(parents=True, exist_ok=True)
    _write_report(outdir, survey, input_path, report)
    return outdir


def run_inspect(config_path: Path) -> Path:
    """Run catalog inspection from a YAML config file and write reports."""
    return run_inspect_config(load_config(config_path))
