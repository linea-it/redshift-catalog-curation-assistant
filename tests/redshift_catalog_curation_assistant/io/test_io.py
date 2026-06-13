import pytest

from redshift_catalog_curation_assistant.io import _should_use_dask, read_table


def assert_dask_dataframe(df):
    """Ensure a DataFrame came from Dask without relying on its internal class module."""
    assert hasattr(df, "compute")
    assert hasattr(df, "npartitions")


def test_headerless_file_requires_column_names(tmp_path):
    """Ensure headerless files ask for explicit user-provided column names."""
    path = tmp_path / "sample.idz"
    path.write_text("1 2 TGS436Z001 0.2981 4\n")

    with pytest.raises(ValueError, match="exactly 5 column names.*--column-name/--column-names"):
        read_table(path)


def test_headerless_file_validates_column_name_count(tmp_path):
    """Ensure user-provided column names match the file width."""
    path = tmp_path / "sample.dat"
    path.write_text("10.0 -1.0 0.2\n")

    with pytest.raises(ValueError, match="2 entries.*3 columns.*--column-name/--column-names"):
        read_table(path, column_names=["ra", "dec"])


def test_headerless_file_uses_column_names(tmp_path):
    """Ensure headerless whitespace tables can be read with explicit names."""
    path = tmp_path / "sample.dat"
    path.write_text("# comment\n10.0 -1.0 0.2\n11.0 -1.1 0.3\n")

    df = read_table(path, column_names=["ra", "dec", "z"])

    assert list(df.columns) == ["ra", "dec", "z"]
    assert len(df) == 2


def test_column_names_are_ignored_for_headered_csv(tmp_path):
    """Ensure explicit names do not override files that already carry headers."""
    path = tmp_path / "sample.csv"
    path.write_text("object_id,z\n1,0.1\n")

    df = read_table(path, column_names=["wrong_a", "wrong_b"])

    assert list(df.columns) == ["object_id", "z"]


def test_dask_threshold_uses_file_size(tmp_path):
    """Ensure Dask selection is controlled by file size threshold."""
    path = tmp_path / "sample.csv"
    path.write_text("object_id,z\n1,0.1\n")

    assert _should_use_dask(path, threshold_bytes=path.stat().st_size)
    assert not _should_use_dask(path, threshold_bytes=path.stat().st_size + 1)
    assert not _should_use_dask(path, threshold_bytes=None)


def test_read_table_can_disable_dask_threshold(tmp_path):
    """Ensure callers can force pandas reading regardless of file size."""
    path = tmp_path / "sample.csv"
    path.write_text("object_id,z\n1,0.1\n")

    df = read_table(path, dask_threshold_bytes=None)

    assert df.__class__.__module__.startswith("pandas")
    assert list(df.columns) == ["object_id", "z"]


def test_read_table_uses_dask_for_csv_at_threshold(tmp_path):
    """Ensure CSV files at the configured threshold use Dask."""
    path = tmp_path / "sample.csv"
    path.write_text("object_id,z\n1,0.1\n2,0.2\n")

    df = read_table(path, dask_threshold_bytes=0)

    assert_dask_dataframe(df)
    assert list(df.columns) == ["object_id", "z"]
    assert len(df) == 2


def test_read_table_uses_dask_for_headerless_table_at_threshold(tmp_path):
    """Ensure headerless whitespace tables at the configured threshold use Dask."""
    path = tmp_path / "sample.idz"
    path.write_text("1 10.0 -1.0 0.2\n2 11.0 -1.1 0.3\n")

    df = read_table(path, column_names=["object_id", "ra", "dec", "z"], dask_threshold_bytes=0)

    assert_dask_dataframe(df)
    assert list(df.columns) == ["object_id", "ra", "dec", "z"]
    assert len(df) == 2


def test_read_table_uses_dask_for_parquet_at_threshold(tmp_path):
    """Ensure Parquet files at the configured threshold use Dask."""
    pd = pytest.importorskip("pandas")
    path = tmp_path / "sample.parquet"
    pd.DataFrame({"object_id": [1, 2], "z": [0.1, 0.2]}).to_parquet(path)

    df = read_table(path, dask_threshold_bytes=0)

    assert_dask_dataframe(df)
    assert list(df.columns) == ["object_id", "z"]
    assert len(df) == 2


def test_read_table_uses_dask_for_partitioned_parquet_directory(tmp_path):
    """Ensure partitioned Parquet directories are accepted as Dask inputs."""
    pd = pytest.importorskip("pandas")
    path = tmp_path / "sample.parquet"
    path.mkdir()
    pd.DataFrame({"object_id": [1], "z": [0.1]}).to_parquet(path / "part000.parquet")
    pd.DataFrame({"object_id": [2], "z": [0.2]}).to_parquet(path / "part001.parquet")

    df = read_table(path)

    assert_dask_dataframe(df)
    assert list(df.columns) == ["object_id", "z"]
    assert len(df) == 2


def test_read_table_rejects_non_parquet_directory(tmp_path):
    """Ensure arbitrary directories are not treated as catalog inputs."""
    path = tmp_path / "not_a_catalog"
    path.mkdir()
    (path / "notes.txt").write_text("not parquet\n")

    with pytest.raises(ValueError, match="Only partitioned Parquet datasets"):
        read_table(path)


def test_read_table_can_load_fits_regardless_of_dask_threshold(tmp_path):
    """Ensure FITS reading remains available for callers that need a DataFrame."""
    from astropy.io import fits
    from astropy.table import Table

    path = tmp_path / "sample.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(Table({"z": [0.1]}), name="CATALOG")]).writeto(path)

    df = read_table(path, dask_threshold_bytes=0)

    assert list(df.columns) == ["z"]
    assert len(df) == 1


def test_fits_numeric_columns_are_converted_to_native_byte_order(tmp_path):
    """Ensure FITS big-endian numeric columns are safe for pandas/numpy stats."""
    from astropy.io import fits
    from astropy.table import Table

    path = tmp_path / "sample.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(Table({"object_id": [1, 2], "z": [0.1, 0.2]}), name="CATALOG"),
        ]
    ).writeto(path)

    df = read_table(path, dask_threshold_bytes=None)

    assert df["object_id"].dtype.isnative
    assert df["z"].dtype.isnative
    assert df["z"].mean() == pytest.approx(0.15)
