import json
from pathlib import Path
import re
from typing import Dict, List

import pandas as pd
import yaml

PATTERNS = {
    "redshift": [r"^z$", r"redshift", r"zspec", r"z_spec", r"zphot", r"z_phot"],
    "quality": [r"quality", r"qop", r"flag", r"zflag", r"confidence"],
    "redshift_error": [r"z_err", r"zerr", r"err_z", r"z_error"],
    "id": [r"id", r"objectid", r"object_id", r"targetid"],
    "object_type": [r"class", r"subclass", r"objtype", r"spectype", r"type"],
}


def load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def candidate_columns(columns: List[str], patterns: Dict[str, List[str]]) -> Dict[str, List[str]]:
    matches = {k: [] for k in patterns}
    for col in columns:
        for cat, pats in patterns.items():
            for p in pats:
                if re.search(p, col, flags=re.IGNORECASE):
                    matches[cat].append(col)
                    break
    return matches


def gather_stats(df: pd.DataFrame) -> Dict:
    stats = {}
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


def run_inspect(config_path: Path):
    cfg = load_config(config_path)
    input_path = Path(cfg.get("input_file"))
    survey = cfg.get("survey_name", input_path.stem)
    outdir = Path("reports") / survey
    outdir.mkdir(parents=True, exist_ok=True)

    from .io import read_table

    df = read_table(input_path)

    report = {
        "survey": survey,
        "input_file": str(input_path),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "candidates": candidate_columns(list(df.columns), PATTERNS),
        "numeric_stats": gather_stats(df),
        "sample": df.head(5).to_dict(orient="records"),
    }

    (outdir / "inspect_report.json").write_text(json.dumps(report, indent=2))
    md = [f"# Inspect report: {survey}\n"]
    md.append(f"Input file: {input_path}\n")
    md.append(f"Rows: {report['n_rows']}  Columns: {report['n_columns']}\n")
    md.append("## Candidate columns by category\n")
    for k, v in report["candidates"].items():
        md.append(f"- **{k}**: {', '.join(v) if v else '—'}\n")
    (outdir / "inspect_report.md").write_text("\n".join(md))
