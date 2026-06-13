from pathlib import Path
from typing import Any

LARGE_COMPRESSED_FITS_THRESHOLD_BYTES = 2 * 1024 * 1024 * 1024
LARGE_COMPRESSED_FITS_COMPRESSED_THRESHOLD_BYTES = 100 * 1024 * 1024


class LargeCompressedFitsError(ValueError):
    """Raised when a compressed FITS file is too large for safe inspection."""


def _is_gzip_path(path: Path) -> bool:
    return path.suffix.lower() == ".gz"


def _gzip_uncompressed_size(path: Path) -> int:
    with open(path, "rb") as handle:
        handle.seek(-4, 2)
        return int.from_bytes(handle.read(4), byteorder="little")


def _fits_without_gzip_suffix(path: Path) -> Path:
    return path.with_suffix("")


def _check_compressed_fits_size(path: Path) -> None:
    if not _is_gzip_path(path):
        return

    compressed_size = path.stat().st_size
    gzip_footer_size = _gzip_uncompressed_size(path)
    if (
        compressed_size < LARGE_COMPRESSED_FITS_COMPRESSED_THRESHOLD_BYTES
        and gzip_footer_size < LARGE_COMPRESSED_FITS_THRESHOLD_BYTES
    ):
        return

    uncompressed_path = _fits_without_gzip_suffix(path)
    compressed_size_gb = compressed_size / (1024**3)
    footer_size_gb = gzip_footer_size / (1024**3)
    compressed_threshold_gb = LARGE_COMPRESSED_FITS_COMPRESSED_THRESHOLD_BYTES / (1024**3)
    threshold_gb = LARGE_COMPRESSED_FITS_THRESHOLD_BYTES / (1024**3)
    msg = (
        f"Compressed FITS file is too large for safe HDU inspection: {path}\n\n"
        f"Compressed size: {compressed_size_gb:.2f} GiB "
        f"(threshold: {compressed_threshold_gb:.2f} GiB).\n"
        f"Gzip footer uncompressed-size field: {footer_size_gb:.2f} GiB "
        f"(threshold: {threshold_gb:.2f} GiB; values above 4 GiB may wrap).\n\n"
        "gzip-compressed FITS files cannot be memory-mapped efficiently. "
        "Decompress the file first, then inspect the uncompressed FITS file:\n\n"
        f"  gzip -dk {path}\n"
        f"  redshift-curator inspect-fits {uncompressed_path}"
    )
    raise LargeCompressedFitsError(msg)


def _image_shape_from_header(header: Any) -> tuple[int, ...] | None:
    naxis = int(header.get("NAXIS", 0) or 0)
    if naxis == 0:
        return None
    return tuple(int(header.get(f"NAXIS{axis}", 0) or 0) for axis in range(naxis, 0, -1))


def _table_column_names_from_header(header: Any) -> list[str]:
    n_columns = int(header.get("TFIELDS", 0) or 0)
    return [str(header.get(f"TTYPE{index}", f"COL{index}")) for index in range(1, n_columns + 1)]


def _is_table_header(header: Any) -> bool:
    return str(header.get("XTENSION", "")).strip().upper() in {"BINTABLE", "TABLE"}


def describe_fits(path: Path, max_columns: int = 12) -> list[dict[str, Any]]:
    """Return a compact description of every HDU in a FITS file."""
    from astropy.io import fits

    _check_compressed_fits_size(path)

    summaries = []
    with fits.open(path, memmap=True) as hdul:
        for index, hdu in enumerate(hdul):
            is_table = _is_table_header(hdu.header)
            summary: dict[str, Any] = {
                "index": index,
                "name": hdu.name,
                "type": type(hdu).__name__,
                "shape": (int(hdu.header.get("NAXIS2", 0) or 0),)
                if is_table
                else _image_shape_from_header(hdu.header),
            }
            if is_table:
                column_names = _table_column_names_from_header(hdu.header)
                summary.update(
                    {
                        "n_rows": int(hdu.header.get("NAXIS2", 0) or 0),
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
        hdu_line = f"HDU {summary['index']}: {summary['name']} ({summary['type']}), shape={summary['shape']}"
        lines.append(hdu_line)
        if "n_rows" in summary:
            lines.append(f"  rows={summary['n_rows']} columns={summary['n_columns']}")
            columns = ", ".join(summary["columns"]) if summary["columns"] else "none"
            if summary["truncated_columns"]:
                columns = f"{columns}, ... (+{summary['truncated_columns']} more)"
            lines.append(f"  column preview: {columns}")
    return "\n".join(lines)
