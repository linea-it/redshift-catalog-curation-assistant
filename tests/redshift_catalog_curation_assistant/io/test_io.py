import pytest

from redshift_catalog_curation_assistant.io import read_table


def test_headerless_file_requires_column_names(tmp_path):
    """Ensure headerless files ask for explicit user-provided column names."""
    path = tmp_path / "sample.idz"
    path.write_text("1 2 TGS436Z001 0.2981 4\n")

    with pytest.raises(ValueError, match="column_names.*exactly 5 entries"):
        read_table(path)


def test_headerless_file_validates_column_name_count(tmp_path):
    """Ensure user-provided column names match the file width."""
    path = tmp_path / "sample.dat"
    path.write_text("10.0 -1.0 0.2\n")

    with pytest.raises(ValueError, match="2 entries.*3 columns"):
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
