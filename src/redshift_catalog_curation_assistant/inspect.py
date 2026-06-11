import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

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


def gather_stats(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Collect simple numeric statistics for a DataFrame."""
    stats: dict[str, dict[str, float | int]] = {}
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        stats[col] = {
            "count": int(numeric[col].count()),
            "mean": float(numeric[col].mean()),
            "std": float(numeric[col].std()),
            "min": float(numeric[col].min()),
            "max": float(numeric[col].max()),
        }
    return stats


def gather_categorical_uniques(df: pd.DataFrame, limit: int = 10) -> dict[str, list[str]]:
    """Collect bounded unique values for non-numeric columns."""
    uniques = {}
    categorical = df.select_dtypes(exclude="number")
    for col in categorical.columns:
        values = categorical[col].dropna().astype(str).unique()
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


def run_inspect_config(config: dict[str, Any]) -> Path:
    """Run catalog inspection from a loaded config and write reports."""
    cfg = config
    input_path = Path(cfg["input_file"])
    survey = cfg.get("survey_name", input_path.stem)
    outdir = Path("reports") / survey
    outdir.mkdir(parents=True, exist_ok=True)

    from .io import read_table

    df = read_table(input_path, fits_hdu=cfg.get("fits_hdu", 1), column_names=cfg.get("column_names"))
    patterns = build_patterns(cfg)

    report = {
        "survey": survey,
        "input_file": str(input_path),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "candidates": candidate_columns(list(df.columns), patterns),
        "numeric_stats": gather_stats(df),
        "categorical_uniques": gather_categorical_uniques(df, limit=int(cfg.get("unique_limit", 10))),
        "sample": df.head(5).to_dict(orient="records"),
        "warnings": [],
    }

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
