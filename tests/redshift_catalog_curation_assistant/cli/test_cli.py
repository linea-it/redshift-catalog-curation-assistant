from click.testing import CliRunner

from redshift_catalog_curation_assistant.cli import cli


def test_cli_help():
    """Ensure the CLI exposes the phase-0 command surface."""
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "inspect" in result.output
    assert "inspect-fits" in result.output
    assert "curate" in result.output
    assert "qa" in result.output
    assert "validate-flags" in result.output
    assert "run" in result.output


def test_version_command():
    """Ensure the version command works before editable installation."""
    result = CliRunner().invoke(cli, ["version"])

    assert result.exit_code == 0
    assert result.output.strip()


def test_inspect_path_command(tmp_path, monkeypatch):
    """Ensure inspect can run with default settings from --path."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")

    result = CliRunner().invoke(cli, ["inspect", "--path", str(csv), "--survey-name", "PATH_SAMPLE"])

    assert result.exit_code == 0
    assert "reports/PATH_SAMPLE" in result.output
    assert (tmp_path / "reports" / "PATH_SAMPLE" / "inspect_report.json").exists()


def test_inspect_rejects_config_and_path(tmp_path):
    """Ensure inspect accepts either YAML config or --path, not both."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"input_file: {csv}\n")

    result = CliRunner().invoke(cli, ["inspect", str(cfg), "--path", str(csv)])

    assert result.exit_code != 0
    assert "Provide exactly one of CONFIG or --path" in result.output


def test_inspect_path_accepts_column_names_for_headerless_file(tmp_path, monkeypatch):
    """Ensure inspect --path can read headerless files with repeated --column-name."""
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "sample.dat"
    data.write_text("10.0 -1.0 0.2\n")

    result = CliRunner().invoke(
        cli,
        [
            "inspect",
            "--path",
            str(data),
            "--survey-name",
            "HEADERLESS",
            "--column-name",
            "ra",
            "--column-name",
            "dec",
            "--column-name",
            "z",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "reports" / "HEADERLESS" / "inspect_report.json").exists()


def test_inspect_path_accepts_column_names_list_for_headerless_file(tmp_path, monkeypatch):
    """Ensure inspect --path can read headerless files with --column-names."""
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "sample.dat"
    data.write_text("10.0 -1.0 0.2\n")

    result = CliRunner().invoke(
        cli,
        [
            "inspect",
            "--path",
            str(data),
            "--survey-name",
            "HEADERLESS_LIST",
            "--column-names",
            '["ra", "dec", "z"]',
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "reports" / "HEADERLESS_LIST" / "inspect_report.json").exists()


def test_inspect_path_accepts_comma_separated_column_names(tmp_path, monkeypatch):
    """Ensure inspect --path accepts comma-separated --column-names."""
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "sample.dat"
    data.write_text("10.0 -1.0 0.2\n")

    result = CliRunner().invoke(
        cli,
        ["inspect", "--path", str(data), "--survey-name", "HEADERLESS_CSV", "--column-names", "ra,dec,z"],
    )

    assert result.exit_code == 0
    assert (tmp_path / "reports" / "HEADERLESS_CSV" / "inspect_report.json").exists()


def test_inspect_rejects_mixed_column_name_options(tmp_path):
    """Ensure the two column-name option styles cannot be mixed."""
    data = tmp_path / "sample.dat"
    data.write_text("10.0 -1.0 0.2\n")

    result = CliRunner().invoke(
        cli,
        ["inspect", "--path", str(data), "--column-name", "ra", "--column-names", '["ra", "dec", "z"]'],
    )

    assert result.exit_code != 0
    assert "Use either --column-name repeatedly or --column-names once" in result.output


def test_planned_command(tmp_path):
    """Ensure planned commands are present and return a stable placeholder."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("input_file: example.csv\n")

    result = CliRunner().invoke(cli, ["curate", str(cfg)])

    assert result.exit_code == 0
    assert "planned for a later phase" in result.output


def test_inspect_fits_command(tmp_path):
    """Ensure the inspect-fits command summarizes table HDUs."""
    from astropy.io import fits
    from astropy.table import Table

    path = tmp_path / "sample.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(Table({"z": [0.1]}), name="CATALOG")]).writeto(path)

    result = CliRunner().invoke(cli, ["inspect-fits", str(path)])

    assert result.exit_code == 0
    assert "HDU 1: CATALOG" in result.output
    assert "rows=1 columns=1" in result.output
