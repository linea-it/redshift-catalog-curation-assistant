from pathlib import Path

import pandas as pd

HEADERLESS_SUFFIXES = {".dat", ".idz"}


def _table_to_dataframe(table) -> pd.DataFrame:
    data = {}
    for name in table.colnames:
        column = table[name]
        if len(column.shape) > 1:
            data[name] = column.tolist()
        else:
            data[name] = column
    return pd.DataFrame(data)


def _read_headerless_table(path: Path, column_names: list[str] | None) -> pd.DataFrame:
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

    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=column_names)
    return df


def read_table(path: Path, fits_hdu: int | None = 1, column_names: list[str] | None = None) -> pd.DataFrame:
    """Read a supported catalog file into a pandas DataFrame."""
    path = Path(path)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    data_suffixes = [suffix for suffix in suffixes if suffix != ".gz"]
    suffix = data_suffixes[-1] if data_suffixes else path.suffix.lower()

    if suffix in HEADERLESS_SUFFIXES:
        return _read_headerless_table(path, column_names=column_names)
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if suffix in [".fits"]:
        from astropy.table import Table

        table = Table.read(str(path), hdu=fits_hdu)
        return _table_to_dataframe(table)
    raise ValueError(f"Unsupported file type: {''.join(suffixes) or path}")
