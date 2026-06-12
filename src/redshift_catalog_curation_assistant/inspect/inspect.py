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
PARQUET_WIDE_COLUMN_THRESHOLD = 200
PARQUET_SAMPLE_MAX_COLUMNS = 100
PARQUET_STATS_BATCH_SIZE = 50
StatsValue = float | int | None

PATTERNS = {
    "ra": [
        r"^ra$",
        r"(^|_)ra_deg($|_)",
        r"(^|_)target_ra($|_)",
        r"(^|_)obsra($|_)",
        r"(^|_)ra_j2000($|_)",
        r"^alpha$",
        r"(^|_)right_ascension($|_)",
    ],
    "dec": [
        r"^dec$",
        r"dec_deg",
        r"target_dec",
        r"obsdec",
        r"dec_j2000",
        r"declination",
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
        values = _compute_if_needed(categorical[col].dropna().astype(str).unique())
        uniques[col] = values[:limit].tolist()
    return uniques


def _is_parquet_input(path: Path) -> bool:
    if path.is_dir():
        return (path / "_metadata").exists() or any(path.glob("*.parquet")) or any(path.glob("*.pq"))
    return path.suffix.lower() in PARQUET_SUFFIXES


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


def _parquet_sample_columns(
    columns: list[str], candidate_cols: list[str], config: dict[str, Any]
) -> list[str]:
    max_columns = int(config.get("parquet_sample_max_columns", PARQUET_SAMPLE_MAX_COLUMNS))
    if max_columns <= 0:
        return []
    return _ordered_unique([*candidate_cols, *columns])[:max_columns]


def _parquet_stats_columns(
    schema: Any, columns: list[str], candidate_cols: list[str], config: dict[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    numeric = _parquet_numeric_columns(schema)
    categorical = _parquet_categorical_columns(schema)
    stats_mode = str(config.get("parquet_stats_mode", "auto")).lower()
    wide_threshold = int(config.get("parquet_wide_column_threshold", PARQUET_WIDE_COLUMN_THRESHOLD))
    is_wide = len(columns) > wide_threshold

    if stats_mode not in {"auto", "candidates", "all", "none"}:
        msg = "parquet_stats_mode must be one of: auto, candidates, all, none."
        raise ValueError(msg)
    if stats_mode == "none":
        return [], [], ["Parquet statistics were skipped because parquet_stats_mode='none'."]

    warnings: list[str] = []
    if stats_mode == "all" or (stats_mode == "auto" and not is_wide):
        return numeric, categorical, warnings

    selected = set(candidate_cols)
    selected_numeric = [col for col in numeric if col in selected]
    selected_categorical = [col for col in categorical if col in selected]
    if is_wide and stats_mode == "auto":
        warnings.append(
            "Parquet input is wide; numeric_stats and categorical_uniques were limited to "
            "candidate columns. Set parquet_stats_mode: all to inspect every column."
        )
    if not selected_numeric and not selected_categorical:
        warnings.append("No candidate columns were eligible for Parquet statistics.")
    return selected_numeric, selected_categorical, warnings


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
    columns = _parquet_columns(schema)
    patterns = build_patterns(config)
    candidates = candidate_columns(columns, patterns)
    candidate_cols = _flatten_candidates(candidates)
    warnings = []

    sample_columns = _parquet_sample_columns(columns, candidate_cols, config)
    if len(sample_columns) < len(columns):
        warnings.append(
            f"Parquet sample was limited to {len(sample_columns)} of {len(columns)} columns. "
            "Increase parquet_sample_max_columns to include more columns."
        )
    sample = _parquet_to_pandas(dataset, sample_columns, limit=5) if sample_columns else pd.DataFrame()

    batch_size = int(config.get("parquet_stats_batch_size", PARQUET_STATS_BATCH_SIZE))
    if batch_size <= 0:
        msg = "parquet_stats_batch_size must be greater than zero."
        raise ValueError(msg)

    numeric_cols, categorical_cols, stats_warnings = _parquet_stats_columns(
        schema, columns, candidate_cols, config
    )
    warnings.extend(stats_warnings)

    return {
        "survey": survey,
        "input_file": str(input_path),
        "n_rows": _parquet_count_rows(dataset),
        "n_columns": len(columns),
        "columns": columns,
        "dtypes": _parquet_dtypes(schema),
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


def _dask_logs_dir(cluster_config: dict[str, Any], outdir: Path) -> Path | None:
    logs_dir = cluster_config.get("logs_dir")
    if logs_dir:
        return Path(logs_dir)
    if cluster_config.get("name") == "slurm":
        return outdir / "dask-logs"
    return None


def _build_report(
    input_path: Path, survey: str, df: pd.DataFrame | Any, config: dict[str, Any]
) -> dict[str, Any]:
    patterns = build_patterns(config)
    sample = _head(df, 5)
    sample = _compute_if_needed(sample)

    return {
        "survey": survey,
        "input_file": str(input_path),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "candidates": candidate_columns(list(df.columns), patterns),
        "numeric_stats": gather_stats(df),
        "categorical_uniques": gather_categorical_uniques(df, limit=int(config.get("unique_limit", 10))),
        "sample": sample.to_dict(orient="records"),
        "warnings": [],
    }


def run_inspect_config(config: dict[str, Any]) -> Path:
    """Run catalog inspection from a loaded config and write reports."""
    cfg = config
    input_path = Path(cfg["input_file"])
    survey = cfg.get("survey_name", input_path.stem)
    outdir = Path("reports") / survey
    outdir.mkdir(parents=True, exist_ok=True)

    if _is_parquet_input(input_path):
        report = _build_parquet_report(input_path, survey, cfg)
        (outdir / "inspect_report.json").write_text(json.dumps(report, indent=2, default=json_default))
        md = [f"# Inspect report: {survey}\n"]
        md.append(f"Input file: {input_path}\n")
        md.append(f"Rows: {report['n_rows']}  Columns: {report['n_columns']}\n")
        md.append("## Candidate columns by category\n")
        for k, v in report["candidates"].items():
            md.append(f"- **{k}**: {', '.join(v) if v else '—'}\n")
        (outdir / "inspect_report.md").write_text("\n".join(md))
        return outdir

    from ..io import read_table

    df = read_table(
        input_path,
        fits_hdu=cfg.get("fits_hdu", 1),
        column_names=cfg.get("column_names"),
        dask_threshold_bytes=_dask_threshold_bytes(cfg),
        load_big_fits=bool(cfg.get("load_big_fits", False)),
    )

    if _is_dask_dataframe(df):
        from ..executor import dask_client_context, dask_cluster_config

        cluster_config = dask_cluster_config(cfg)
        with dask_client_context(cluster_config, logs_dir=_dask_logs_dir(cluster_config, outdir)):
            report = _build_report(input_path, survey, df, cfg)
    else:
        report = _build_report(input_path, survey, df, cfg)

    (outdir / "inspect_report.json").write_text(json.dumps(report, indent=2, default=json_default))
    md = [f"# Inspect report: {survey}\n"]
    md.append(f"Input file: {input_path}\n")
    md.append(f"Rows: {report['n_rows']}  Columns: {report['n_columns']}\n")
    md.append("## Candidate columns by category\n")
    for k, v in report["candidates"].items():
        md.append(f"- **{k}**: {', '.join(v) if v else '—'}\n")
    (outdir / "inspect_report.md").write_text("\n".join(md))
    return outdir


def run_inspect(config_path: Path) -> Path:
    """Run catalog inspection from a YAML config file and write reports."""
    return run_inspect_config(load_config(config_path))
