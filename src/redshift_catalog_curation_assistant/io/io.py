from pathlib import Path
from typing import Any

import pandas as pd

HEADERLESS_SUFFIXES = {".dat", ".idz"}
DEFAULT_DASK_THRESHOLD_BYTES = 100 * 1024 * 1024


class LargeFitsError(ValueError):
    """Raised when reading a large FITS file without explicit opt-in."""


def _should_use_dask(path: Path, threshold_bytes: int | None) -> bool:
    if threshold_bytes is None:
        return False
    return path.stat().st_size >= threshold_bytes


def _import_dask_dataframe() -> Any:
    try:
        import dask.dataframe as dd
    except ImportError as exc:
        msg = "Dask is required to read large tabular files. Install the package with the dask extra."
        raise RuntimeError(msg) from exc
    return dd


def _is_parquet_dataset_dir(path: Path) -> bool:
    return (path / "_metadata").exists() or any(path.glob("*.parquet"))


def _table_to_dataframe(table) -> pd.DataFrame:
    data = {}
    for name in table.colnames:
        column = table[name]
        if len(column.shape) > 1:
            data[name] = column.tolist()
        else:
            data[name] = column
    return pd.DataFrame(data)


def _read_headerless_table(path: Path, column_names: list[str] | None, use_dask: bool) -> pd.DataFrame | Any:
    detected_columns = pd.read_csv(path, sep=r"\s+", comment="#", header=None, nrows=1).shape[1]
    if not column_names:
        msg = (
            f"{path} does not contain column names. Provide a 'column_names' list in the YAML "
            f"with exactly {detected_columns} entries to read this file."
        )
        raise ValueError(msg)
    if len(column_names) != detected_columns:
        msg = (
            f"Configured column_names has {len(column_names)} entries, but {path} has "
            f"{detected_columns} columns."
        )
        raise ValueError(msg)

    if use_dask:
        dd = _import_dask_dataframe()
        df = dd.read_csv(str(path), sep=r"\s+", comment="#", header=None, names=column_names)
        return df

    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=column_names)
    return df


def read_table(
    path: Path,
    fits_hdu: int | None = 1,
    column_names: list[str] | None = None,
    dask_threshold_bytes: int | None = DEFAULT_DASK_THRESHOLD_BYTES,
    load_big_fits: bool = False,
) -> pd.DataFrame | Any:
    """Read a supported catalog file into a DataFrame."""
    path = Path(path)
    if path.is_dir():
        if _is_parquet_dataset_dir(path):
            dd = _import_dask_dataframe()
            return dd.read_parquet(str(path))
        raise ValueError(
            f"Unsupported directory input: {path}. " "Only partitioned Parquet datasets are supported."
        )

    suffixes = [suffix.lower() for suffix in path.suffixes]
    data_suffixes = [suffix for suffix in suffixes if suffix != ".gz"]
    suffix = data_suffixes[-1] if data_suffixes else path.suffix.lower()
    use_dask = suffix != ".fits" and _should_use_dask(path, dask_threshold_bytes)

    if suffix in HEADERLESS_SUFFIXES:
        return _read_headerless_table(path, column_names=column_names, use_dask=use_dask)
    if suffix in [".csv", ".txt"]:
        if use_dask:
            dd = _import_dask_dataframe()
            return dd.read_csv(str(path))
        return pd.read_csv(path)
    if suffix in [".parquet", ".pq"]:
        if use_dask:
            dd = _import_dask_dataframe()
            return dd.read_parquet(str(path))
        return pd.read_parquet(path)
    if suffix in [".fits"]:
        if not load_big_fits and _should_use_dask(path, dask_threshold_bytes):
            threshold_mb = dask_threshold_bytes / (1024 * 1024) if dask_threshold_bytes is not None else 0
            msg = (
                f"FITS file is larger than the configured threshold ({threshold_mb:g} MB): {path}\n\n"
                "Reading large FITS files currently loads the selected HDU into memory with astropy "
                "and pandas. To continue anyway, pass --load-big-fits or set load_big_fits: true "
                "in the YAML config.\n\n"
                "Recommended workflow:\n"
                f"  1. redshift-curator inspect-fits {path}\n"
                f"  2. redshift-curator convert-fits {path} output.parquet --fits-hdu {fits_hdu}\n"
                "  3. redshift-curator inspect --path output.parquet"
            )
            raise LargeFitsError(msg)

        from astropy.table import Table

        table = Table.read(str(path), hdu=fits_hdu)
        return _table_to_dataframe(table)
    raise ValueError(f"Unsupported file type: {''.join(suffixes) or path}")
