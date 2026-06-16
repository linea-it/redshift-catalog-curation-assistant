import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..hats import is_hats_input
from ..io import DEFAULT_DASK_THRESHOLD_BYTES

PARQUET_SUFFIXES = {".parquet", ".pq"}
FITS_SUFFIXES = {".fits", ".fit", ".fts"}
COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".zip"}
SAMPLE_MAX_COLUMNS = 100
PARQUET_STATS_BATCH_SIZE = 50
FITS_STATS_BATCH_SIZE = 8
REDSHIFT_NULL_WARNING_FRACTION = 0.5
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
        return yaml.safe_load(f) or {}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _validate_positive_int(config: dict[str, Any], key: str) -> None:
    if key not in config:
        return
    if not _is_int(config[key]) or config[key] <= 0:
        raise ValueError(f"{key} must be a positive integer.")


def _validate_non_negative_int(config: dict[str, Any], key: str) -> None:
    if key not in config:
        return
    if not _is_int(config[key]) or config[key] < 0:
        raise ValueError(f"{key} must be an integer >= 0.")


def _validate_string_list(config: dict[str, Any], key: str, message: str) -> None:
    if key not in config:
        return
    values = config[key]
    if not isinstance(values, list | tuple) or not all(isinstance(value, str) for value in values):
        raise ValueError(message)


def _validate_inspect_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Inspect config must be a YAML mapping.")
    if not config:
        raise ValueError("Inspect config is empty. Set 'input_file' in YAML or pass --path in the CLI.")

    input_file = config.get("input_file")
    if input_file is None:
        raise ValueError("Inspect config requires 'input_file'. Set it in YAML or pass --path in the CLI.")
    if not isinstance(input_file, str | Path) or not str(input_file).strip():
        raise ValueError("input_file must be a non-empty path string.")

    survey_name = config.get("survey_name")
    if survey_name is not None and (not isinstance(survey_name, str) or not survey_name.strip()):
        raise ValueError("survey_name must be a non-empty string when provided.")
    output_dir = config.get("output_dir")
    if output_dir is not None and (not isinstance(output_dir, str | Path) or not str(output_dir).strip()):
        raise ValueError("output_dir must be a non-empty path string when provided.")

    _validate_positive_int(config, "fits_hdu")
    _validate_non_negative_int(config, "unique_limit")
    _validate_non_negative_int(config, "sample_max_columns")
    _validate_positive_int(config, "parquet_stats_batch_size")
    _validate_positive_int(config, "fits_stats_batch_size")
    _validate_positive_int(config, "fits_stats_chunk_rows")
    _validate_positive_int(config, "chunk_size_rows")
    _validate_positive_int(config, "sample_seed")
    _stats_mode(config)

    _validate_string_list(
        config,
        "column_names",
        "column_names must be a list of strings. Set 'column_names' in YAML, "
        "or pass --column-name/--column-names in the CLI.",
    )
    _validate_string_list(
        config,
        "column_selection",
        "column_selection must be a list of column names. Set 'column_selection' in YAML, "
        "or pass --column-selection/--column-selection-list in the CLI.",
    )

    column_patterns = config.get("column_patterns", {})
    if not isinstance(column_patterns, dict):
        raise ValueError("column_patterns must be a mapping from category names to regex lists.")
    for category, values in column_patterns.items():
        if (
            not isinstance(category, str)
            or not isinstance(values, list | tuple)
            or not all(isinstance(value, str) for value in values)
        ):
            raise ValueError(f"column_patterns.{category} must be a list of regex strings.")

    dask_threshold_mb = config.get("dask_threshold_mb")
    if dask_threshold_mb is not None and (not _is_number(dask_threshold_mb) or dask_threshold_mb < 0):
        raise ValueError("dask_threshold_mb must be a number >= 0 or null.")

    if "dask_cluster" in config and not isinstance(config["dask_cluster"], dict):
        raise ValueError("dask_cluster must be a mapping. Use YAML for cluster configuration.")
    if "allow_large_raw_inspect" in config and not isinstance(config["allow_large_raw_inspect"], bool):
        raise ValueError("allow_large_raw_inspect must be a boolean.")
    if "parallel_stats" in config and not isinstance(config["parallel_stats"], bool):
        raise ValueError("parallel_stats must be a boolean.")

    return config


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


def _semantic_warnings(
    candidates: dict[str, list[str]],
    numeric_columns: list[str],
    numeric_stats: dict[str, dict[str, StatsValue]],
) -> list[str]:
    warnings = []
    numeric_column_set = set(numeric_columns)

    if not candidates.get("ra"):
        warnings.append("No RA candidate columns were found.")
    if not candidates.get("dec"):
        warnings.append("No DEC candidate columns were found.")
    if not candidates.get("redshift"):
        warnings.append("No redshift candidate columns were found.")
    elif len(candidates["redshift"]) > 1:
        warnings.append(
            "Multiple redshift candidate columns were found: " + ", ".join(candidates["redshift"]) + "."
        )

    non_numeric_ra = [column for column in candidates.get("ra", []) if column not in numeric_column_set]
    non_numeric_dec = [column for column in candidates.get("dec", []) if column not in numeric_column_set]
    if non_numeric_ra:
        warnings.append("RA candidate columns are not numeric: " + ", ".join(non_numeric_ra) + ".")
    if non_numeric_dec:
        warnings.append("DEC candidate columns are not numeric: " + ", ".join(non_numeric_dec) + ".")

    null_warnings = []
    for column in candidates.get("redshift", []):
        stats = numeric_stats.get(column)
        if not stats:
            continue
        count = stats.get("count")
        null_count = stats.get("null_count")
        if not isinstance(count, int | float) or not isinstance(null_count, int | float):
            continue
        total = count + null_count
        if total <= 0:
            continue
        null_fraction = null_count / total
        if null_fraction >= REDSHIFT_NULL_WARNING_FRACTION and null_count > 0:
            null_warnings.append(f"{column} ({null_fraction:.1%} null)")
    if null_warnings:
        warnings.append("Redshift candidate columns have many null values: " + ", ".join(null_warnings) + ".")

    return warnings


def _is_dask_dataframe(df: Any) -> bool:
    return df.__class__.__module__.startswith(("dask.dataframe", "dask_expr."))


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


def _parallel_stats_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("parallel_stats", False))


def _stats_partitions(
    partial_stats: list[dict[str, dict[str, StatsValue]]],
) -> dict[str, dict[str, StatsValue]]:
    aggregates: dict[str, dict[str, float]] = {}
    for stats in partial_stats:
        for column, values in stats.items():
            count = values.get("count") or 0
            null_count = values.get("null_count") or 0
            mean = values.get("mean")
            std = values.get("std")
            min_value = values.get("min")
            max_value = values.get("max")
            if not isinstance(count, int | float) or count <= 0:
                aggregate = aggregates.setdefault(
                    column,
                    {
                        "count": 0.0,
                        "null_count": 0.0,
                        "sum": 0.0,
                        "sum_sq_delta": 0.0,
                        "min": math.inf,
                        "max": -math.inf,
                    },
                )
                aggregate["null_count"] += float(null_count) if isinstance(null_count, int | float) else 0.0
                continue

            aggregate = aggregates.setdefault(
                column,
                {
                    "count": 0.0,
                    "null_count": 0.0,
                    "sum": 0.0,
                    "sum_sq_delta": 0.0,
                    "min": math.inf,
                    "max": -math.inf,
                },
            )
            count_float = float(count)
            mean_float = float(mean) if isinstance(mean, int | float) else 0.0
            std_float = float(std) if isinstance(std, int | float) else 0.0
            aggregate["count"] += count_float
            aggregate["null_count"] += float(null_count) if isinstance(null_count, int | float) else 0.0
            aggregate["sum"] += mean_float * count_float
            if count_float > 1:
                aggregate["sum_sq_delta"] += std_float**2 * (count_float - 1.0)
            if isinstance(min_value, int | float):
                aggregate["min"] = min(aggregate["min"], float(min_value))
            if isinstance(max_value, int | float):
                aggregate["max"] = max(aggregate["max"], float(max_value))

    merged: dict[str, dict[str, StatsValue]] = {}
    for column, aggregate in aggregates.items():
        count = int(aggregate["count"])
        null_count = int(aggregate["null_count"])
        if count <= 0:
            merged[column] = {
                "count": 0,
                "null_count": null_count,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
            continue
        mean = aggregate["sum"] / count
        total_sum_sq_delta = aggregate["sum_sq_delta"]
        for stats in partial_stats:
            column_stats = stats.get(column)
            if not column_stats:
                continue
            partial_count = column_stats.get("count") or 0
            partial_mean = column_stats.get("mean")
            if not isinstance(partial_count, int | float) or partial_count <= 0:
                continue
            if not isinstance(partial_mean, int | float):
                continue
            total_sum_sq_delta += float(partial_count) * (float(partial_mean) - mean) ** 2
        merged[column] = {
            "count": count,
            "null_count": null_count,
            "mean": mean,
            "std": math.sqrt(total_sum_sq_delta / (count - 1)) if count > 1 else None,
            "min": aggregate["min"] if aggregate["min"] != math.inf else None,
            "max": aggregate["max"] if aggregate["max"] != -math.inf else None,
        }
    return merged


def _merge_unique_partitions(partials: list[dict[str, list[str]]], limit: int) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for partial in partials:
        for column, values in partial.items():
            current = merged.setdefault(column, [])
            for value in values:
                if value not in current:
                    current.append(value)
                if len(current) >= limit:
                    break
    return merged


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


def _parquet_fragment_stats(fragment: Any, columns: list[str]) -> dict[str, dict[str, StatsValue]]:
    table = fragment.to_table(columns=columns)
    return gather_stats(table.to_pandas())


def _parquet_numeric_stats_parallel(
    dataset: Any, columns: list[str], batch_size: int
) -> dict[str, dict[str, StatsValue]]:
    if not columns:
        return {}

    from dask import delayed

    tasks = []
    fragments = list(dataset.get_fragments())
    for fragment in fragments:
        for start in range(0, len(columns), batch_size):
            batch_columns = columns[start : start + batch_size]
            tasks.append(delayed(_parquet_fragment_stats)(fragment, batch_columns))
    if not tasks:
        return {}
    partials = list(delayed(list)(tasks).compute())
    return _stats_partitions(partials)


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


def _parquet_fragment_uniques(fragment: Any, columns: list[str], limit: int) -> dict[str, list[str]]:
    table = fragment.to_table(columns=columns)
    return gather_categorical_uniques(table.to_pandas(), limit=limit)


def _parquet_categorical_uniques_parallel(
    dataset: Any, columns: list[str], batch_size: int, limit: int
) -> dict[str, list[str]]:
    if not columns:
        return {}

    from dask import delayed

    tasks = []
    fragments = list(dataset.get_fragments())
    for fragment in fragments:
        for start in range(0, len(columns), batch_size):
            batch_columns = columns[start : start + batch_size]
            tasks.append(delayed(_parquet_fragment_uniques)(fragment, batch_columns, limit))
    if not tasks:
        return {}
    partials = list(delayed(list)(tasks).compute())
    return _merge_unique_partitions(partials, limit=limit)


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
    unique_limit = int(config.get("unique_limit", 10))
    if _parallel_stats_enabled(config):
        from ..executor import dask_client_context, dask_cluster_config

        cluster_config = dask_cluster_config(config)
        logs_dir = _dask_logs_dir(cluster_config, Path("reports") / survey)
        with dask_client_context(cluster_config, logs_dir=logs_dir):
            numeric_stats = _parquet_numeric_stats_parallel(dataset, numeric_cols, batch_size)
            categorical_uniques = _parquet_categorical_uniques_parallel(
                dataset,
                categorical_cols,
                batch_size=batch_size,
                limit=unique_limit,
            )
    else:
        numeric_stats = _parquet_numeric_stats(dataset, numeric_cols, batch_size)
        categorical_uniques = _parquet_categorical_uniques(
            dataset,
            categorical_cols,
            batch_size=batch_size,
            limit=unique_limit,
        )
    warnings.extend(_semantic_warnings(candidates, _parquet_numeric_columns(schema), numeric_stats))

    return {
        "survey": survey,
        "input_file": str(input_path),
        "n_rows": _parquet_count_rows(dataset),
        "n_columns": len(all_columns),
        "n_columns_selected": len(columns),
        "columns": columns,
        "dtypes": {column: dtypes[column] for column in columns},
        "candidates": candidates,
        "numeric_stats": numeric_stats,
        "categorical_uniques": categorical_uniques,
        "sample": sample.to_dict(orient="records"),
        "warnings": warnings,
    }


def _hats_select_columns(catalog: Any, columns: list[str]) -> Any:
    drop_columns = [column for column in catalog.columns if column not in set(columns)]
    if not drop_columns:
        return catalog
    return catalog.drop(columns=drop_columns)


def _hats_sample(catalog: Any, columns: list[str], n_rows: int, seed: int) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame()
    sampled = _hats_select_columns(catalog, columns).random_sample(n=n_rows, seed=seed)
    return pd.DataFrame(sampled)


def _hats_numeric_columns(dtypes: pd.Series) -> list[str]:
    return [column for column, dtype in dtypes.items() if pd.api.types.is_numeric_dtype(dtype)]


def _hats_categorical_columns(dtypes: pd.Series) -> list[str]:
    return [column for column, dtype in dtypes.items() if not pd.api.types.is_numeric_dtype(dtype)]


def _hats_numeric_stats(catalog: Any, columns: list[str]) -> dict[str, dict[str, StatsValue]]:
    if not columns:
        return {}
    stats_frame = catalog.aggregate_column_statistics(include_columns=columns)
    stats: dict[str, dict[str, StatsValue]] = {}
    for column in columns:
        if column not in stats_frame.index:
            continue
        row = stats_frame.loc[column]
        row_count = int(row.get("row_count", 0) or 0)
        null_count = int(row.get("null_count", 0) or 0)
        count = max(row_count - null_count, 0)
        stats[column] = {
            "count": count,
            "null_count": null_count,
            "mean": None,
            "std": None,
            "min": float(row["min_value"]) if count else None,
            "max": float(row["max_value"]) if count else None,
        }
    return stats


def _hats_categorical_uniques(
    catalog: Any, columns: list[str], limit: int, seed: int
) -> dict[str, list[str]]:
    if not columns:
        return {}
    sample = _hats_sample(catalog, columns, n_rows=max(limit * 5, limit), seed=seed)
    return gather_categorical_uniques(sample, limit=limit)


def _build_hats_report(input_path: Path, survey: str, config: dict[str, Any]) -> dict[str, Any]:
    try:
        import lsdb
    except ImportError as exc:
        raise RuntimeError(
            "lsdb is required to inspect HATS catalogs. Install the project with LSDB."
        ) from exc

    catalog = lsdb.open_catalog(input_path)
    all_columns = list(catalog.columns)
    columns, warnings = _selected_report_columns(all_columns, config)
    patterns = build_patterns(config)
    candidates = candidate_columns(columns, patterns)
    candidate_cols = _flatten_candidates(candidates)
    dtypes = catalog.dtypes

    sample_columns = _sample_columns(columns, candidate_cols, config)
    if len(sample_columns) < len(columns):
        warnings.append(
            f"HATS sample was limited to {len(sample_columns)} of {len(columns)} columns. "
            "Increase 'sample_max_columns' in YAML, or pass --sample-max-columns in the CLI."
        )
    seed = int(config.get("sample_seed", 42))
    sample = _hats_sample(catalog, sample_columns, n_rows=5, seed=seed)

    numeric_columns = [column for column in _hats_numeric_columns(dtypes) if column in columns]
    categorical_columns = [column for column in _hats_categorical_columns(dtypes) if column in columns]
    numeric_cols, categorical_cols, stats_warnings = _selected_stats_columns(
        columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        candidate_cols=candidate_cols,
        config=config,
        input_kind="HATS",
    )
    warnings.extend(stats_warnings)
    numeric_stats = _hats_numeric_stats(catalog, numeric_cols)
    categorical_uniques = _hats_categorical_uniques(
        catalog,
        categorical_cols,
        limit=int(config.get("unique_limit", 10)),
        seed=seed,
    )
    if categorical_cols:
        warnings.append(
            "HATS categorical unique values were estimated from an LSDB random_sample, not a full scan."
        )
    warnings.extend(_semantic_warnings(candidates, numeric_columns, numeric_stats))

    return {
        "survey": survey,
        "input_file": str(input_path),
        "input_format": "hats",
        "n_rows": int(len(catalog)),
        "n_columns": len(all_columns),
        "n_columns_selected": len(columns),
        "columns": columns,
        "dtypes": {column: str(dtypes[column]) for column in columns},
        "candidates": candidates,
        "numeric_stats": numeric_stats,
        "categorical_uniques": categorical_uniques,
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


def _fits_chunk_dataframe(
    input_path: str, fits_hdu: int, columns: list[str], start: int, stop: int
) -> pd.DataFrame:
    import fitsio

    with fitsio.FITS(input_path) as fits_file:
        data = fits_file[fits_hdu].read(columns=columns, rows=range(start, stop))
    return pd.DataFrame({column: np.asarray(data[column]) for column in columns})


def _fits_chunk_stats(
    input_path: str, fits_hdu: int, columns: list[str], start: int, stop: int
) -> dict[str, dict[str, StatsValue]]:
    return gather_stats(_fits_chunk_dataframe(input_path, fits_hdu, columns, start, stop))


def _fits_numeric_stats_parallel(
    input_path: Path,
    fits_hdu: int,
    n_rows: int,
    columns: list[str],
    batch_size: int,
    chunk_rows: int,
) -> dict[str, dict[str, StatsValue]]:
    if not columns:
        return {}

    from dask import delayed

    tasks = []
    for row_start in range(0, n_rows, chunk_rows):
        row_stop = min(row_start + chunk_rows, n_rows)
        for column_start in range(0, len(columns), batch_size):
            batch_columns = columns[column_start : column_start + batch_size]
            tasks.append(
                delayed(_fits_chunk_stats)(str(input_path), fits_hdu, batch_columns, row_start, row_stop)
            )
    if not tasks:
        return {}
    partials = list(delayed(list)(tasks).compute())
    return _stats_partitions(partials)


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


def _fits_chunk_uniques(
    input_path: str, fits_hdu: int, columns: list[str], start: int, stop: int, limit: int
) -> dict[str, list[str]]:
    frame = _fits_chunk_dataframe(input_path, fits_hdu, columns, start, stop)
    return gather_categorical_uniques(frame, limit)


def _fits_categorical_uniques_parallel(
    input_path: Path,
    fits_hdu: int,
    n_rows: int,
    columns: list[str],
    batch_size: int,
    chunk_rows: int,
    limit: int,
) -> dict[str, list[str]]:
    if not columns:
        return {}

    from dask import delayed

    tasks = []
    for row_start in range(0, n_rows, chunk_rows):
        row_stop = min(row_start + chunk_rows, n_rows)
        for column_start in range(0, len(columns), batch_size):
            batch_columns = columns[column_start : column_start + batch_size]
            tasks.append(
                delayed(_fits_chunk_uniques)(
                    str(input_path), fits_hdu, batch_columns, row_start, row_stop, limit
                )
            )
    if not tasks:
        return {}
    partials = list(delayed(list)(tasks).compute())
    return _merge_unique_partitions(partials, limit=limit)


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
        n_rows = int(header.get("NAXIS2", 0) or 0)
        unique_limit = int(config.get("unique_limit", 10))
        if _parallel_stats_enabled(config):
            from ..executor import dask_client_context, dask_cluster_config

            cluster_config = dask_cluster_config(config)
            chunk_rows = int(config.get("fits_stats_chunk_rows", config.get("chunk_size_rows", 200_000)))
            with dask_client_context(
                cluster_config, logs_dir=_dask_logs_dir(cluster_config, Path("reports") / survey)
            ):
                numeric_stats = _fits_numeric_stats_parallel(
                    input_path,
                    fits_hdu,
                    n_rows,
                    scalar_numeric_cols,
                    batch_size,
                    chunk_rows,
                )
                categorical_uniques = _fits_categorical_uniques_parallel(
                    input_path,
                    fits_hdu,
                    n_rows,
                    scalar_categorical_cols,
                    batch_size,
                    chunk_rows,
                    unique_limit,
                )
        else:
            numeric_stats = _fits_numeric_stats(hdu, scalar_numeric_cols, batch_size)
            categorical_uniques = _fits_categorical_uniques(
                hdu,
                scalar_categorical_cols,
                batch_size=batch_size,
                limit=unique_limit,
            )
        warnings.extend(_semantic_warnings(candidates, _fits_numeric_columns(formats), numeric_stats))

        return {
            "survey": survey,
            "input_file": str(input_path),
            "n_rows": n_rows,
            "n_columns": len(all_columns),
            "n_columns_selected": len(columns),
            "columns": columns,
            "dtypes": formats,
            "candidates": candidates,
            "numeric_stats": numeric_stats,
            "categorical_uniques": categorical_uniques,
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
    numeric_stats = gather_stats(stats_df)
    categorical_uniques = gather_categorical_uniques(
        categorical_df, limit=int(config.get("unique_limit", 10))
    )
    warnings.extend(_semantic_warnings(candidates, numeric_columns, numeric_stats))

    return {
        "survey": survey,
        "input_file": str(input_path),
        "n_rows": int(len(df)),
        "n_columns": int(len(all_columns)),
        "n_columns_selected": int(len(columns)),
        "columns": columns,
        "dtypes": {c: str(t) for c, t in report_df.dtypes.items()},
        "candidates": candidates,
        "numeric_stats": numeric_stats,
        "categorical_uniques": categorical_uniques,
        "sample": sample.to_dict(orient="records"),
        "warnings": warnings,
    }


def _markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_markdown_value(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _append_columns_markdown(md: list[str], report: dict[str, Any]) -> None:
    md.append("## Columns\n")
    rows = [
        [index, column, report["dtypes"].get(column, "")]
        for index, column in enumerate(report["columns"], start=1)
    ]
    md.append(_markdown_table(["#", "column", "dtype"], rows) if rows else "No columns selected.")
    md.append("")


def _append_numeric_stats_markdown(md: list[str], report: dict[str, Any]) -> None:
    md.append("## Numeric statistics\n")
    stats = report["numeric_stats"]
    if not stats:
        md.append("No numeric statistics were collected.\n")
        return

    stat_keys = ["count", "null_count", "mean", "std", "min", "max"]
    rows = [[column, *(values.get(key) for key in stat_keys)] for column, values in stats.items()]
    md.append(_markdown_table(["column", *stat_keys], rows))
    md.append("")


def _append_categorical_uniques_markdown(md: list[str], report: dict[str, Any]) -> None:
    md.append("## Categorical unique values\n")
    uniques = report["categorical_uniques"]
    if not uniques:
        md.append("No categorical unique values were collected.\n")
        return

    rows = [
        [column, json.dumps(values, default=json_default, ensure_ascii=False)]
        for column, values in uniques.items()
    ]
    md.append(_markdown_table(["column", "values"], rows))
    md.append("")


def _append_sample_markdown(md: list[str], report: dict[str, Any]) -> None:
    md.append("## Sample rows\n")
    sample = report["sample"]
    if not sample:
        md.append("No sample rows were collected.\n")
        return

    md.append("```json")
    md.append(json.dumps(sample, indent=2, default=json_default, ensure_ascii=False))
    md.append("```\n")


def _append_warnings_markdown(md: list[str], report: dict[str, Any]) -> None:
    md.append("## Warnings\n")
    warnings = report["warnings"]
    if not warnings:
        md.append("No warnings.\n")
        return

    for warning in warnings:
        md.append(f"- {warning}")
    md.append("")


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
        md.append(f"- **{k}**: {', '.join(v) if v else 'None'}\n")
    _append_columns_markdown(md, report)
    _append_numeric_stats_markdown(md, report)
    _append_categorical_uniques_markdown(md, report)
    _append_sample_markdown(md, report)
    _append_warnings_markdown(md, report)
    (outdir / "inspect_report.md").write_text("\n".join(md))


def run_inspect_config(config: dict[str, Any]) -> Path:
    """Run catalog inspection from a loaded config and write reports."""
    cfg = _validate_inspect_config(config)
    input_path = Path(cfg["input_file"])
    survey = cfg.get("survey_name", input_path.stem)
    outdir = Path(cfg["output_dir"]) if cfg.get("output_dir") is not None else Path("reports") / survey

    if is_hats_input(input_path):
        from ..executor import dask_client_context, dask_cluster_config

        cluster_config = dask_cluster_config(cfg)
        with dask_client_context(cluster_config, logs_dir=_dask_logs_dir(cluster_config, outdir)):
            report = _build_hats_report(input_path, survey, cfg)
        outdir.mkdir(parents=True, exist_ok=True)
        _write_report(outdir, survey, input_path, report)
        return outdir

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
