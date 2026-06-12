import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..io import DEFAULT_DASK_THRESHOLD_BYTES

PATTERNS = {
    "ra": [r"^ra$", r"ra_deg", r"target_ra", r"obsra", r"ra_j2000", r"alpha", r"right_ascension"],
    "dec": [r"^dec$", r"dec_deg", r"target_dec", r"obsdec", r"dec_j2000", r"declination"],
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
    "quality": [r"quality", r"qual", r"qop", r"flag", r"zflag", r"z_flag", r"vi_quality", r"confidence"],
    "redshift_error": [r"z_err", r"zerr", r"err_z", r"z_error", r"sigma_z"],
    "id": [r"id", r"objectid", r"object_id", r"targetid", r"specid", r"catalogid"],
    "object_type": [r"class", r"subclass", r"objtype", r"spectype", r"subtype", r"eta_type", r"type"],
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


def gather_stats(df: pd.DataFrame | Any) -> dict[str, dict[str, float | int]]:
    """Collect simple numeric statistics for a DataFrame."""
    stats: dict[str, dict[str, float | int]] = {}
    numeric = df.select_dtypes(include="number")
    if _is_dask_dataframe(df):
        if len(numeric.columns) == 0:
            return stats
        for col in numeric.columns:
            series = numeric[col]
            stats[col] = {
                "count": int(series.count().compute()),
                "mean": float(series.mean().compute()),
                "std": float(series.std().compute()),
                "min": float(series.min().compute()),
                "max": float(series.max().compute()),
            }
        return stats

    for col in numeric.columns:
        stats[col] = {
            "count": int(numeric[col].count()),
            "mean": float(numeric[col].mean()),
            "std": float(numeric[col].std()),
            "min": float(numeric[col].min()),
            "max": float(numeric[col].max()),
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
