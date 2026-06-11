from pathlib import Path
from typing import Optional

import pandas as pd


def read_table(path: Path, fits_hdu: Optional[int] = 1) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if suffix in [".fits"]:
        try:
            from astropy.table import Table

            t = Table.read(str(path), hdu=fits_hdu)
            return t.to_pandas()
        except Exception as exc:  # pragma: no cover - best-effort
            raise
    raise ValueError(f"Unsupported file type: {suffix}")
