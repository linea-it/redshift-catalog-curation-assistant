import json
from contextlib import contextmanager

import pytest
from click.testing import CliRunner

from redshift_catalog_curation_assistant.cli import cli


def test_cli_help():
    """Ensure the CLI exposes the phase-0 command surface."""
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "inspect" in result.output
    assert "inspect-fits" in result.output
    assert "prepare" in result.output
    assert "curate" in result.output
    assert "qa" in result.output
    assert "validate-flags" in result.output
    assert "run" in result.output
    assert "--log-level" in result.output


def test_inspect_help():
    """Ensure the inspect command exposes inspection-specific options."""
    result = CliRunner().invoke(cli, ["inspect", "--help"])

    assert result.exit_code == 0
    assert "--fits-hdu" in result.output
    assert "--stats-mode" in result.output
    assert "--sample-max-columns" in result.output
    assert "--output-dir" in result.output
    assert "--allow-large-raw-inspect" in result.output


def test_prepare_help():
    """Ensure the prepare command exposes output layout options."""
    result = CliRunner().invoke(cli, ["prepare", "--help"])

    assert result.exit_code == 0
    assert "--output-format" in result.output
    assert "--output-mode" in result.output
    assert "--allow-large-single-output" in result.output
    assert "--hats-ra-column" in result.output
    assert "--dry-run" in result.output


def test_inspect_fits_reports_large_compressed_file_error(tmp_path, monkeypatch):
    """Ensure inspect-fits reports large compressed FITS files without traceback."""
    import redshift_catalog_curation_assistant.fits.fits as fits_module

    path = tmp_path / "large.fits.gz"
    path.write_bytes(b"not a real gzip")
    monkeypatch.setattr(fits_module, "_gzip_uncompressed_size", lambda path: 3 * 1024 * 1024 * 1024)

    result = CliRunner().invoke(cli, ["inspect-fits", str(path)])

    assert result.exit_code != 0
    assert "Compressed FITS file is too large" in result.output
    assert "gzip -dk" in result.output
    assert "Traceback" not in result.output


def test_version_command():
    """Ensure the version command works before editable installation."""
    result = CliRunner().invoke(cli, ["version"])

    assert result.exit_code == 0
    assert result.output.strip()


def test_prepare_path_command(tmp_path):
    """Ensure prepare can materialize a small file as Parquet."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")
    output_dir = tmp_path / "prepared"

    result = CliRunner().invoke(cli, ["prepare", "--path", str(csv), "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert "Prepared catalog written to" in result.output
    assert "Starting prepare" in result.output
    assert "Reading input and writing single Parquet file" in result.output
    assert "Writing preparation manifest" in result.output
    assert "Prepare completed" in result.output
    assert sorted(output_dir.glob("*.parquet"))
    assert (output_dir / "_redshift_curator_manifest.json").exists()


def test_prepare_path_dry_run_does_not_write_output(tmp_path):
    """Ensure prepare dry-run validates schema without materializing output."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")
    output_dir = tmp_path / "prepared"

    result = CliRunner().invoke(
        cli,
        ["prepare", "--path", str(csv), "--output-dir", str(output_dir), "--dry-run"],
    )

    assert result.exit_code == 0
    assert "Prepare dry-run OK" in result.output
    assert str(output_dir) in result.output
    assert not output_dir.exists()


def test_prepare_path_cli_can_validate_hats_flags_in_dry_run(tmp_path):
    """Ensure direct prepare CLI exposes enough HATS options for schema validation."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z\n1,10.0,-1.0,0.1\n")
    output_dir = tmp_path / "prepared_hats"

    result = CliRunner().invoke(
        cli,
        [
            "prepare",
            "--path",
            str(csv),
            "--output-dir",
            str(output_dir),
            "--output-format",
            "hats",
            "--hats-catalog-name",
            "toy",
            "--hats-ra-column",
            "ra",
            "--hats-dec-column",
            "dec",
            "--hats-margin-threshold",
            "5",
            "--no-hats-output-with-margin",
            "--hats-sort-columns",
            "object_id",
            "--hats-create-thumbnail",
            "--progress-bar",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Prepare dry-run OK" in result.output
    assert not output_dir.exists()


def test_prepare_rejects_missing_output_dir(tmp_path):
    """Ensure prepare requires an output directory when no config is provided."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")

    result = CliRunner().invoke(cli, ["prepare", "--path", str(csv)])

    assert result.exit_code != 0
    assert "at least one --path plus --output-dir" in result.output


def test_inspect_path_command(tmp_path, monkeypatch):
    """Ensure inspect can run with default settings from --path."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")

    result = CliRunner().invoke(cli, ["inspect", "--path", str(csv), "--survey-name", "PATH_SAMPLE"])

    assert result.exit_code == 0
    assert "reports/PATH_SAMPLE" in result.output
    assert "Starting inspect" in result.output
    assert "Reading tabular input" in result.output
    assert "Computing in-memory catalog inspection statistics" in result.output
    assert "Writing inspection reports" in result.output
    assert "Inspect completed" in result.output
    assert (tmp_path / "reports" / "PATH_SAMPLE" / "inspect_report.json").exists()


def test_inspect_path_dry_run_does_not_write_reports(tmp_path, monkeypatch):
    """Ensure inspect dry-run validates schema without writing reports."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")

    result = CliRunner().invoke(cli, ["inspect", "--path", str(csv), "--survey-name", "DRY"])

    assert result.exit_code == 0
    assert (tmp_path / "reports" / "DRY" / "inspect_report.json").exists()

    dry_reports = tmp_path / "reports" / "DRY_ONLY"
    result = CliRunner().invoke(
        cli,
        ["inspect", "--path", str(csv), "--survey-name", "DRY_ONLY", "--dry-run"],
    )

    assert result.exit_code == 0
    assert "Inspect dry-run OK" in result.output
    assert "reports/DRY_ONLY" in result.output
    assert not dry_reports.exists()


def test_inspect_path_command_accepts_output_dir(tmp_path, monkeypatch):
    """Ensure inspect --path can write reports to an explicit directory."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,0.1\n")
    output_dir = tmp_path / "inspect-output"

    result = CliRunner().invoke(
        cli,
        [
            "inspect",
            "--path",
            str(csv),
            "--survey-name",
            "PATH_OUTPUT",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert f"Wrote inspection reports to {output_dir}" in result.output
    assert (output_dir / "inspect_report.json").exists()
    assert (output_dir / "inspect_report.md").exists()
    assert not (tmp_path / "reports" / "PATH_OUTPUT").exists()


def test_inspect_path_accepts_partitioned_parquet_directory(tmp_path, monkeypatch):
    """Ensure inspect --path accepts partitioned Parquet directories."""
    import pandas as pd

    import redshift_catalog_curation_assistant.executor as dex

    monkeypatch.chdir(tmp_path)
    parquet_dir = tmp_path / "sample.parquet"
    parquet_dir.mkdir()
    pd.DataFrame({"object_id": [1], "z": [0.1]}).to_parquet(parquet_dir / "part000.parquet")
    pd.DataFrame({"object_id": [2], "z": [0.2]}).to_parquet(parquet_dir / "part001.parquet")

    @contextmanager
    def fake_dask_client_context(cluster_config, logs_dir=None):
        yield

    monkeypatch.setattr(dex, "dask_client_context", fake_dask_client_context)

    result = CliRunner().invoke(cli, ["inspect", "--path", str(parquet_dir), "--survey-name", "PARQUET_DIR"])

    assert result.exit_code == 0
    assert (tmp_path / "reports" / "PARQUET_DIR" / "inspect_report.json").exists()


def test_inspect_parquet_path_does_not_start_dask_cluster(tmp_path, monkeypatch):
    """Ensure Parquet inspection uses the PyArrow path instead of Dask."""
    import pandas as pd

    import redshift_catalog_curation_assistant.executor as dex

    monkeypatch.chdir(tmp_path)
    parquet_dir = tmp_path / "sample.parquet"
    parquet_dir.mkdir()
    pd.DataFrame({"object_id": [1], "z": [0.1]}).to_parquet(parquet_dir / "part000.parquet")

    client_calls = []

    @contextmanager
    def fake_dask_client_context(cluster_config, logs_dir=None):
        client_calls.append((cluster_config, logs_dir))
        yield

    monkeypatch.setattr(dex, "dask_client_context", fake_dask_client_context)

    cluster_config = (
        "{'name': 'local', 'logs_dir': 'reports/dask-logs', "
        "'args': {'n_workers': 1, 'threads_per_worker': 1, 'processes': False}}"
    )
    result = CliRunner().invoke(
        cli,
        [
            "inspect",
            "--path",
            str(parquet_dir),
            "--survey-name",
            "PARQUET_CLUSTER",
            "--dask-cluster",
            cluster_config,
        ],
    )

    assert result.exit_code == 0
    assert client_calls == []
    assert (tmp_path / "reports" / "PARQUET_CLUSTER" / "inspect_report.json").exists()


def test_inspect_rejects_invalid_dask_cluster_option(tmp_path):
    """Ensure invalid --dask-cluster values fail before running inspect."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")

    result = CliRunner().invoke(cli, ["inspect", "--path", str(csv), "--dask-cluster", "not-a-dict"])

    assert result.exit_code != 0
    assert "Expected 'local' or a dict literal" in result.output


def test_inspect_rejects_simple_slurm_dask_cluster_option(tmp_path):
    """Ensure SLURM requires explicit executor args when set from CLI."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")

    result = CliRunner().invoke(cli, ["inspect", "--path", str(csv), "--dask-cluster", "slurm"])

    assert result.exit_code != 0
    assert "SLURM Dask clusters require a dict literal with args" in result.output


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


def test_inspect_path_accepts_column_selection(tmp_path, monkeypatch):
    """Ensure inspect --path can limit reports with repeated --column-selection."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z,extra\n1,10.0,-1.0,0.1,5\n2,11.0,-1.1,0.2,6\n")

    result = CliRunner().invoke(
        cli,
        [
            "inspect",
            "--path",
            str(csv),
            "--survey-name",
            "SELECTED",
            "--column-selection",
            "ra",
            "--column-selection",
            "z",
        ],
    )

    assert result.exit_code == 0
    report = json.loads((tmp_path / "reports" / "SELECTED" / "inspect_report.json").read_text())
    assert report["n_columns"] == 5
    assert report["n_columns_selected"] == 2
    assert report["columns"] == ["ra", "z"]
    assert report["sample"][0] == {"ra": 10.0, "z": 0.1}
    assert set(report["numeric_stats"]) == {"ra", "z"}
    assert "object_id" not in report["numeric_stats"]


def test_inspect_path_accepts_stats_mode_and_sample_max_columns(tmp_path, monkeypatch):
    """Ensure inspect --path can control stats and sample column count."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z,extra\n1,10.0,-1.0,0.1,5\n2,11.0,-1.1,0.2,6\n")

    result = CliRunner().invoke(
        cli,
        [
            "inspect",
            "--path",
            str(csv),
            "--survey-name",
            "CLI_LIMITS",
            "--stats-mode",
            "none",
            "--sample-max-columns",
            "2",
        ],
    )

    assert result.exit_code == 0
    report = json.loads((tmp_path / "reports" / "CLI_LIMITS" / "inspect_report.json").read_text())
    assert len(report["sample"][0]) == 2
    assert report["numeric_stats"] == {}
    assert report["categorical_uniques"] == {}
    assert "Tabular statistics were skipped because stats_mode='none'." in report["warnings"]


def test_inspect_path_accepts_column_selection_list(tmp_path, monkeypatch):
    """Ensure inspect --path can limit reports with --column-selection-list."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z\n1,10.0,-1.0,0.1\n")

    result = CliRunner().invoke(
        cli,
        [
            "inspect",
            "--path",
            str(csv),
            "--survey-name",
            "SELECTED_LIST",
            "--column-selection-list",
            '["object_id", "z"]',
        ],
    )

    assert result.exit_code == 0
    report = json.loads((tmp_path / "reports" / "SELECTED_LIST" / "inspect_report.json").read_text())
    assert report["n_columns"] == 4
    assert report["n_columns_selected"] == 2
    assert report["columns"] == ["object_id", "z"]


def test_inspect_rejects_mixed_column_selection_options(tmp_path):
    """Ensure the two column-selection option styles cannot be mixed."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")

    result = CliRunner().invoke(
        cli,
        [
            "inspect",
            "--path",
            str(csv),
            "--column-selection",
            "object_id",
            "--column-selection-list",
            '["object_id", "z"]',
        ],
    )

    assert result.exit_code != 0
    assert "Use either --column-selection repeatedly or --column-selection-list once" in result.output


def test_inspect_rejects_missing_column_selection(tmp_path, monkeypatch):
    """Ensure invalid selected columns fail with a clear error."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")

    result = CliRunner().invoke(
        cli,
        ["inspect", "--path", str(csv), "--column-selection-list", "object_id,missing"],
    )

    assert result.exit_code != 0
    assert "column_selection contains columns not present in input: missing" in result.output
    assert "--column-selection/--column-selection-list" in result.output


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


def test_curate_command(tmp_path):
    """Ensure curate runs from YAML config and writes Parquet output."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,0.1\n")
    output_dir = tmp_path / "curated"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"input_file: {csv}\n"
        f"output_dir: {output_dir}\n"
        "coordinates:\n"
        "  ra_column: ra\n"
        "  dec_column: dec\n"
        "redshift:\n"
        "  column: z\n"
    )

    result = CliRunner().invoke(cli, ["curate", str(cfg)])

    assert result.exit_code == 0
    assert "Curated catalog written to" in result.output
    assert "Starting curate" in result.output
    assert "Reading input catalog" in result.output
    assert "Validating RA, Dec, and redshift columns" in result.output
    assert "Writing single Parquet file" in result.output
    assert "Curate completed" in result.output
    assert sorted(output_dir.glob("*.parquet"))


def test_curate_command_dry_run_does_not_write_output(tmp_path):
    """Ensure curate dry-run validates schema without writing output."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,0.1\n")
    output_dir = tmp_path / "curated"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"input_file: {csv}\n"
        f"output_dir: {output_dir}\n"
        "coordinates:\n"
        "  ra_column: ra\n"
        "  dec_column: dec\n"
        "redshift:\n"
        "  column: z\n"
    )

    result = CliRunner().invoke(cli, ["curate", str(cfg), "--dry-run"])

    assert result.exit_code == 0
    assert "Curate dry-run OK" in result.output
    assert str(output_dir) in result.output
    assert not output_dir.exists()


def test_curate_command_dry_run_rejects_missing_selected_column(tmp_path):
    """Ensure curate dry-run catches schema-level column errors."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,0.1\n")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"input_file: {csv}\n"
        f"output_dir: {tmp_path / 'curated'}\n"
        "column_selection:\n"
        "  - ra\n"
        "  - dec\n"
        "  - z\n"
        "  - missing\n"
        "coordinates:\n"
        "  ra_column: ra\n"
        "  dec_column: dec\n"
        "redshift:\n"
        "  column: z\n"
    )

    result = CliRunner().invoke(cli, ["curate", str(cfg), "--dry-run"])

    assert result.exit_code != 0
    assert "Curate input is missing required columns: missing" in result.output


def test_curate_command_can_suppress_info_logs(tmp_path):
    """Ensure the global log-level option controls terminal stage messages."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,0.1\n")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"input_file: {csv}\n"
        f"output_dir: {tmp_path / 'curated'}\n"
        "coordinates:\n"
        "  ra_column: ra\n"
        "  dec_column: dec\n"
        "redshift:\n"
        "  column: z\n"
    )

    result = CliRunner().invoke(cli, ["--log-level", "WARNING", "curate", str(cfg)])

    assert result.exit_code == 0
    assert "Starting curate" not in result.output
    assert "Curated catalog written to" in result.output


@pytest.mark.parametrize("command", ["qa", "validate-flags", "run"])
def test_planned_commands(command, tmp_path):
    """Ensure later-phase commands are present and return a stable placeholder."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("input_file: example.csv\n")

    result = CliRunner().invoke(cli, [command, str(cfg)])

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
