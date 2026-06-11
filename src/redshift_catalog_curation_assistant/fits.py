from pathlib import Path
from typing import Any


def describe_fits(path: Path, max_columns: int = 12) -> list[dict[str, Any]]:
    """Return a compact description of every HDU in a FITS file."""
    from astropy.io import fits

    summaries = []
    with fits.open(path, memmap=True) as hdul:
        for index, hdu in enumerate(hdul):
            summary: dict[str, Any] = {
                "index": index,
                "name": hdu.name,
                "type": type(hdu).__name__,
                "shape": hdu.data.shape if hdu.data is not None else None,
            }
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                column_names = list(hdu.columns.names)
                summary.update(
                    {
                        "n_rows": int(len(hdu.data)) if hdu.data is not None else 0,
                        "n_columns": len(column_names),
                        "columns": column_names[:max_columns],
                        "truncated_columns": max(len(column_names) - max_columns, 0),
                    }
                )
            summaries.append(summary)
    return summaries


def format_fits_description(summaries: list[dict[str, Any]]) -> str:
    """Format FITS HDU summaries for CLI output."""
    lines = []
    for summary in summaries:
        hdu_line = (
            f"HDU {summary['index']}: {summary['name']} "
            f"({summary['type']}), shape={summary['shape']}"
        )
        lines.append(hdu_line)
        if "n_rows" in summary:
            lines.append(f"  rows={summary['n_rows']} columns={summary['n_columns']}")
            columns = ", ".join(summary["columns"]) if summary["columns"] else "none"
            if summary["truncated_columns"]:
                columns = f"{columns}, ... (+{summary['truncated_columns']} more)"
            lines.append(f"  column preview: {columns}")
    return "\n".join(lines)
