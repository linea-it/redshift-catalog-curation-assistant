import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


@contextmanager
def fake_dask_client_context(cluster_config, logs_dir=None):
    """Stand in for a Dask client without starting a real cluster."""
    yield


def test_default_patterns_cover_common_sample_columns():
    """Verify default patterns cover reusable column names without broad DEC false positives."""
    import redshift_catalog_curation_assistant.inspect.inspect as insp

    matches = insp.candidate_columns(
        [
            "OBSRA",
            "OBSDEC",
            "ra_j2000_h",
            "dec_j2000_d",
            "Alpha_J2000",
            "Delta_J2000",
            "z_helio",
            "qual",
            "ZWARN",
            "ZWARNING",
            "ZWARNING_NOQSO",
            "ETA_TYPE",
            "MEAN_DELTA_X",
            "lines_spe_rank_rank0_halpha",
            "mer_fluxerr_z_ext_decam_1fwhm_aper",
            "lines_spe_line_id_rank0_halpha",
            "physparam_sfhtype",
        ],
        insp.PATTERNS,
    )

    assert matches["ra"] == ["OBSRA", "ra_j2000_h", "Alpha_J2000"]
    assert matches["dec"] == ["OBSDEC", "dec_j2000_d", "Delta_J2000"]
    assert matches["redshift"] == ["z_helio"]
    assert matches["quality"] == ["qual", "ZWARN", "ZWARNING", "ZWARNING_NOQSO"]
    assert matches["redshift_error"] == []
    assert matches["id"] == []
    assert matches["object_type"] == ["ETA_TYPE"]


def test_inspect_sample(tmp_path, monkeypatch):
    """Verify inspect writes a report with core catalog metadata."""
    # Create a small CSV and a config
    data = "id,ra,dec,z_phot,VI_quality\n1,10.0,0.1,0.5,3\n2,11.0,0.2,0.6,4\n"
    csv = tmp_path / "sample.csv"
    csv.write_text(data)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"input_file: {csv}\nsurvey_name: TEST\n")
    monkeypatch.chdir(tmp_path)

    # Run inspect
    import redshift_catalog_curation_assistant.inspect.inspect as insp

    outdir = insp.run_inspect(cfg)

    out = outdir / "inspect_report.json"
    assert out.exists()
    report = json.loads(out.read_text())
    assert report["n_rows"] == 2
    assert report["n_columns"] == 5
    assert report["n_columns_selected"] == 5
    assert report["candidates"]["ra"] == ["ra"]
    assert report["candidates"]["dec"] == ["dec"]
    assert report["candidates"]["redshift"] == ["z_phot"]
    assert report["candidates"]["quality"] == ["VI_quality"]
    assert (Path("reports") / "TEST" / "inspect_report.md").exists()


def test_inspect_markdown_includes_json_report_sections(tmp_path, monkeypatch):
    """Verify the Markdown report exposes the useful JSON report content."""
    data = "id,ra,dec,z,kind\n1,10.0,0.1,0.5,galaxy\n2,11.0,0.2,0.6,qso\n"
    csv = tmp_path / "sample.csv"
    csv.write_text(data)
    cfg = {
        "input_file": str(csv),
        "survey_name": "MD_REPORT",
        "stats_mode": "all",
    }
    monkeypatch.chdir(tmp_path)

    import redshift_catalog_curation_assistant.inspect.inspect as insp

    outdir = insp.run_inspect_config(cfg)
    markdown = (outdir / "inspect_report.md").read_text()

    assert "## Columns" in markdown
    assert "| 4 | z | float64 |" in markdown
    assert "## Numeric statistics" in markdown
    assert "| z | 2 | 0 | 0.55 |" in markdown
    assert "## Categorical unique values" in markdown
    assert '| kind | ["galaxy", "qso"] |' in markdown
    assert "## Sample rows" in markdown
    assert '"kind": "galaxy"' in markdown
    assert "## Warnings" in markdown
    assert "No warnings." in markdown


def test_inspect_markdown_includes_real_warnings(tmp_path, monkeypatch):
    """Verify warnings are written to Markdown, not only JSON."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n2,\n")
    cfg = {
        "input_file": str(csv),
        "survey_name": "MD_WARNINGS",
        "column_selection": ["object_id", "z"],
    }
    monkeypatch.chdir(tmp_path)

    import redshift_catalog_curation_assistant.inspect.inspect as insp

    outdir = insp.run_inspect_config(cfg)
    markdown = (outdir / "inspect_report.md").read_text()

    assert "## Warnings" in markdown
    assert "- No RA candidate columns were found." in markdown
    assert "- No DEC candidate columns were found." in markdown
    assert "- Report was limited to 2 selected columns out of 2 available columns." in markdown


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "Inspect config is empty"),
        (None, "Inspect config must be a YAML mapping"),
        ([], "Inspect config must be a YAML mapping"),
        ({"survey_name": "NO_INPUT"}, "Inspect config requires 'input_file'"),
        ({"input_file": ""}, "input_file must be a non-empty path string"),
        ({"input_file": 12}, "input_file must be a non-empty path string"),
        ({"input_file": "sample.csv", "survey_name": ""}, "survey_name must be a non-empty string"),
        ({"input_file": "sample.csv", "survey_name": 7}, "survey_name must be a non-empty string"),
        ({"input_file": "sample.csv", "output_dir": ""}, "output_dir must be a non-empty path string"),
        ({"input_file": "sample.csv", "output_dir": 7}, "output_dir must be a non-empty path string"),
        ({"input_file": "sample.csv", "fits_hdu": 0}, "fits_hdu must be a positive integer"),
        ({"input_file": "sample.csv", "unique_limit": -1}, "unique_limit must be an integer >= 0"),
        (
            {"input_file": "sample.csv", "sample_max_columns": -1},
            "sample_max_columns must be an integer >= 0",
        ),
        ({"input_file": "sample.csv", "stats_mode": "bad"}, "stats_mode must be one of"),
        (
            {"input_file": "sample.csv", "column_names": "ra,dec,z"},
            "column_names must be a list of strings",
        ),
        (
            {"input_file": "sample.csv", "column_selection": "ra,dec,z"},
            "column_selection must be a list of column names",
        ),
        (
            {"input_file": "sample.csv", "column_patterns": []},
            "column_patterns must be a mapping",
        ),
        (
            {"input_file": "sample.csv", "column_patterns": {"ra": "RA"}},
            "column_patterns.ra must be a list of regex strings",
        ),
        (
            {"input_file": "sample.csv", "parquet_stats_batch_size": 0},
            "parquet_stats_batch_size must be a positive integer",
        ),
        (
            {"input_file": "sample.csv", "fits_stats_batch_size": "many"},
            "fits_stats_batch_size must be a positive integer",
        ),
        (
            {"input_file": "sample.csv", "dask_threshold_mb": -1},
            "dask_threshold_mb must be a number >= 0 or null",
        ),
        (
            {"input_file": "sample.csv", "dask_cluster": "local"},
            "dask_cluster must be a mapping",
        ),
        (
            {"input_file": "sample.csv", "allow_large_raw_inspect": "yes"},
            "allow_large_raw_inspect must be a boolean",
        ),
    ],
)
def test_inspect_config_validation_errors(config, message):
    """Verify invalid inspect configs fail with explicit messages."""
    import redshift_catalog_curation_assistant.inspect.inspect as insp

    with pytest.raises(ValueError, match=message):
        insp.run_inspect_config(config)


def test_inspect_empty_yaml_fails_with_explicit_message(tmp_path):
    """Verify empty YAML configs fail before low-level KeyError/TypeError exceptions."""
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("")

    import redshift_catalog_curation_assistant.inspect.inspect as insp

    with pytest.raises(ValueError, match="Inspect config is empty"):
        insp.run_inspect(cfg)


def test_inspect_config_output_dir_writes_reports_to_custom_directory(tmp_path, monkeypatch):
    """Verify output_dir overrides the default reports/<survey> location."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,0.1\n")
    output_dir = tmp_path / "custom-inspect"

    import redshift_catalog_curation_assistant.inspect.inspect as insp

    outdir = insp.run_inspect_config(
        {
            "input_file": str(csv),
            "survey_name": "CUSTOM_OUTPUT",
            "output_dir": str(output_dir),
        }
    )

    assert outdir == output_dir
    assert (output_dir / "inspect_report.json").exists()
    assert (output_dir / "inspect_report.md").exists()
    assert not (tmp_path / "reports" / "CUSTOM_OUTPUT").exists()


def test_inspect_semantic_warnings_for_missing_ambiguous_and_null_columns(tmp_path, monkeypatch):
    """Verify semantic catalog warnings are audit-only diagnostics."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "semantic.csv"
    csv.write_text(
        "RA_TEXT,dec,z,z_spec,best_z,kind\n"
        "10.0,south,,0.10,0.11,galaxy\n"
        "11.0,south,,0.20,0.21,qso\n"
        "12.0,north,0.30,0.30,0.31,galaxy\n"
        "13.0,north,,0.40,0.41,qso\n"
    )
    cfg = {
        "input_file": str(csv),
        "survey_name": "SEMANTIC_WARNINGS",
        "stats_mode": "all",
    }

    import redshift_catalog_curation_assistant.inspect as insp

    outdir = insp.run_inspect_config(cfg)
    report = json.loads((outdir / "inspect_report.json").read_text())

    assert "No RA candidate columns were found." in report["warnings"]
    assert "Multiple redshift candidate columns were found: z, z_spec, best_z." in report["warnings"]
    assert "DEC candidate columns are not numeric: dec." in report["warnings"]
    assert "Redshift candidate columns have many null values: z (75.0% null)." in report["warnings"]
    assert "No redshift candidate columns were found." not in report["warnings"]


def test_inspect_sample_with_dask_threshold(tmp_path, monkeypatch):
    """Verify inspect can generate the same core report through Dask."""
    data = "id,ra,dec,z_phot,kind\n1,10.0,0.1,0.5,galaxy\n2,11.0,0.2,0.6,qso\n"
    csv = tmp_path / "sample.csv"
    csv.write_text(data)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"input_file: {csv}\n"
        "survey_name: TEST_DASK\n"
        "stats_mode: all\n"
        "dask_threshold_mb: 0.000001\n"
        "allow_large_raw_inspect: true\n"
        "dask_cluster:\n"
        "  name: local\n"
        "  args:\n"
        "    n_workers: 1\n"
        "    threads_per_worker: 1\n"
        "    processes: false\n"
        "    dashboard_address: null\n"
    )
    monkeypatch.chdir(tmp_path)

    import redshift_catalog_curation_assistant.executor as dex
    import redshift_catalog_curation_assistant.inspect as insp

    client_calls = []

    @contextmanager
    def fake_dask_client_context(cluster_config, logs_dir=None):
        client_calls.append((cluster_config, logs_dir))
        yield

    monkeypatch.setattr(dex, "dask_client_context", fake_dask_client_context)

    outdir = insp.run_inspect(cfg)

    report = json.loads((outdir / "inspect_report.json").read_text())
    assert client_calls == [
        (
            {
                "name": "local",
                "args": {
                    "n_workers": 1,
                    "threads_per_worker": 1,
                    "memory_limit": "6GB",
                    "processes": False,
                    "dashboard_address": None,
                },
            },
            None,
        )
    ]
    assert report["n_rows"] == 2
    assert report["n_columns"] == 5
    assert report["n_columns_selected"] == 5
    assert report["numeric_stats"]["z_phot"]["count"] == 2
    assert report["categorical_uniques"]["kind"] == ["galaxy", "qso"]
    assert report["sample"][0]["id"] == 1


def test_inspect_large_raw_input_recommends_prepare(tmp_path, monkeypatch):
    """Ensure large raw inputs are routed through prepare by default."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "large.csv"
    csv.write_text("object_id,z\n1,0.1\n")

    import redshift_catalog_curation_assistant.inspect as insp

    with pytest.raises(ValueError, match="Run prepare first"):
        insp.run_inspect_config(
            {
                "input_file": str(csv),
                "survey_name": "RAW_LARGE",
                "dask_threshold_mb": 0.000001,
            }
        )


def test_inspect_large_compressed_input_recommends_decompression(tmp_path, monkeypatch):
    """Ensure large compressed raw inputs fail with a decompression hint."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "large.csv.gz"
    csv.write_bytes(b"compressed")

    import redshift_catalog_curation_assistant.inspect as insp

    with pytest.raises(ValueError, match="gzip -dk"):
        insp.run_inspect_config(
            {
                "input_file": str(csv),
                "survey_name": "RAW_COMPRESSED",
                "dask_threshold_mb": 0.000001,
            }
        )


def test_inspect_slurm_defaults_logs_to_report_logs_dir(tmp_path):
    """Ensure inspect SLURM logs live under the report directory by default."""
    import redshift_catalog_curation_assistant.inspect.inspect as insp

    logs_dir = insp._dask_logs_dir({"name": "slurm"}, tmp_path / "reports" / "SURVEY")

    assert logs_dir == tmp_path / "reports" / "SURVEY" / "logs"


def test_gather_stats_handles_non_native_numeric_byte_order():
    """Verify FITS-style big-endian numeric columns do not break stats."""
    import redshift_catalog_curation_assistant.inspect as insp

    df = pd.DataFrame(
        {
            "object_id": np.array([1, 2], dtype=">i8"),
            "z": np.array([0.1, 0.2], dtype=">f8"),
        }
    )

    stats = insp.gather_stats(df)

    assert stats["object_id"]["count"] == 2
    assert stats["object_id"]["mean"] == 1.5
    assert stats["z"]["count"] == 2
    assert stats["z"]["mean"] == pytest.approx(0.15)


def test_gather_stats_reports_null_counts_without_nan_values():
    """Verify empty numeric columns are reported as JSON-friendly null stats."""
    import redshift_catalog_curation_assistant.inspect as insp

    stats = insp.gather_stats(pd.DataFrame({"empty": [np.nan, np.nan], "single": [1.0, np.nan]}))

    assert stats["empty"] == {
        "count": 0,
        "null_count": 2,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
    }
    assert stats["single"] == {
        "count": 1,
        "null_count": 1,
        "mean": 1.0,
        "std": None,
        "min": 1.0,
        "max": 1.0,
    }


def test_inspect_csv_defaults_to_candidate_stats(tmp_path, monkeypatch):
    """Verify generic inspect controls apply outside the Parquet path."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "wide.csv"
    csv.write_text(
        "object_id,ra,dec,z,extra_0,extra_1,extra_2\n1,10.0,-1.0,0.1,5,6,7\n2,11.0,-1.1,0.2,8,9,10\n"
    )
    cfg = {
        "input_file": str(csv),
        "survey_name": "WIDE_CSV",
        "sample_max_columns": 3,
    }

    import redshift_catalog_curation_assistant.inspect as insp

    outdir = insp.run_inspect_config(cfg)
    report = json.loads((outdir / "inspect_report.json").read_text())

    assert len(report["sample"][0]) == 3
    assert set(report["numeric_stats"]) == {"object_id", "ra", "dec", "z"}
    assert report["warnings"]


def test_inspect_config_column_selection_limits_report_columns(tmp_path, monkeypatch):
    """Verify column_selection limits sample, candidates, dtypes, and stats."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "wide.csv"
    csv.write_text("object_id,ra,dec,z,kind,extra\n1,10.0,-1.0,0.1,galaxy,5\n2,11.0,-1.1,0.2,qso,6\n")
    cfg = {
        "input_file": str(csv),
        "survey_name": "SELECTED_CSV",
        "column_selection": ["object_id", "z", "kind"],
        "stats_mode": "all",
    }

    import redshift_catalog_curation_assistant.inspect as insp

    outdir = insp.run_inspect_config(cfg)
    report = json.loads((outdir / "inspect_report.json").read_text())

    assert report["n_columns"] == 6
    assert report["n_columns_selected"] == 3
    assert report["columns"] == ["object_id", "z", "kind"]
    assert set(report["dtypes"]) == {"object_id", "z", "kind"}
    assert report["candidates"]["ra"] == []
    assert set(report["numeric_stats"]) == {"object_id", "z"}
    assert report["categorical_uniques"]["kind"] == ["galaxy", "qso"]
    assert report["sample"][0] == {"object_id": 1, "z": 0.1, "kind": "galaxy"}
    assert "Report was limited to 3 selected columns out of 6 available columns." in report["warnings"]


def test_inspect_config_column_selection_rejects_missing_columns(tmp_path, monkeypatch):
    """Verify missing selected columns fail before report generation."""
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")

    import redshift_catalog_curation_assistant.inspect as insp

    with pytest.raises(ValueError, match="column_selection contains columns not present in input: missing"):
        insp.run_inspect_config(
            {
                "input_file": str(csv),
                "survey_name": "BAD_SELECTION",
                "column_selection": ["object_id", "missing"],
            }
        )


def test_inspect_wide_parquet_uses_limited_pyarrow_strategy(tmp_path, monkeypatch):
    """Verify wide Parquet inspection avoids materializing all columns for sample/stats."""
    monkeypatch.chdir(tmp_path)
    data = {
        "object_id": [1, 2, 3],
        "ra": [10.0, 11.0, 12.0],
        "dec": [-1.0, -1.1, -1.2],
        "z": [0.1, 0.2, 0.3],
        "kind": ["galaxy", "qso", "galaxy"],
    }
    data.update({f"extra_{idx}": [idx, idx + 1, idx + 2] for idx in range(8)})
    parquet = tmp_path / "wide.parquet"
    pd.DataFrame(data).to_parquet(parquet)

    cfg = {
        "input_file": str(parquet),
        "survey_name": "WIDE_PARQUET",
        "sample_max_columns": 4,
        "parquet_stats_batch_size": 2,
    }

    import redshift_catalog_curation_assistant.inspect as insp

    outdir = insp.run_inspect_config(cfg)
    report = json.loads((outdir / "inspect_report.json").read_text())

    assert report["n_rows"] == 3
    assert report["n_columns"] == 13
    assert report["n_columns_selected"] == 13
    assert len(report["sample"][0]) == 4
    assert report["candidates"]["ra"] == ["ra"]
    assert report["candidates"]["dec"] == ["dec"]
    assert report["candidates"]["redshift"] == ["z"]
    assert set(report["numeric_stats"]) == {"object_id", "ra", "dec", "z"}
    assert report["numeric_stats"]["z"]["mean"] == pytest.approx(0.2)
    assert report["warnings"]


def test_inspect_parquet_parallel_fragment_stats(tmp_path, monkeypatch):
    """Verify Parquet stats can be computed through Dask fragment tasks."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "partitioned.parquet"
    path.mkdir()
    pd.DataFrame({"ra": [10.0, 11.0], "dec": [-1.0, -1.1], "z": [0.1, 0.2]}).to_parquet(
        path / "part0.parquet"
    )
    pd.DataFrame({"ra": [12.0, 13.0], "dec": [-1.2, -1.3], "z": [0.3, 0.4]}).to_parquet(
        path / "part1.parquet"
    )

    import redshift_catalog_curation_assistant.executor as executor
    import redshift_catalog_curation_assistant.inspect as insp

    monkeypatch.setattr(executor, "dask_client_context", fake_dask_client_context)

    outdir = insp.run_inspect_config(
        {
            "input_file": str(path),
            "survey_name": "PARQUET_PARALLEL",
            "parallel_stats": True,
            "parquet_stats_batch_size": 2,
        }
    )
    report = json.loads((outdir / "inspect_report.json").read_text())

    assert report["numeric_stats"]["z"]["count"] == 4
    assert report["numeric_stats"]["z"]["mean"] == pytest.approx(0.25)
    assert report["numeric_stats"]["ra"]["max"] == pytest.approx(13.0)


def test_inspect_fits_uses_selective_sample_and_stats(tmp_path, monkeypatch):
    """Verify FITS inspection can sample and compute candidate stats selectively."""
    from astropy.io import fits
    from astropy.table import Table

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "sample.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(
                Table(
                    {
                        "RA": [10.0, 11.0, 12.0],
                        "DEC": [-1.0, -1.1, -1.2],
                        "Z": [0.1, 0.2, 0.3],
                        "CLASS": ["GALAXY", "QSO", "GALAXY"],
                        "EXTRA": [1, 2, 3],
                    }
                ),
                name="CATALOG",
            ),
        ]
    ).writeto(path)
    cfg = {
        "input_file": str(path),
        "survey_name": "FITS_SELECTIVE",
        "fits_hdu": 1,
        "stats_mode": "candidates",
        "sample_max_columns": 3,
        "fits_stats_batch_size": 2,
    }

    import redshift_catalog_curation_assistant.inspect as insp

    outdir = insp.run_inspect_config(cfg)
    report = json.loads((outdir / "inspect_report.json").read_text())

    assert len(report["sample"][0]) == 3
    assert report["n_columns"] == 5
    assert report["n_columns_selected"] == 5
    assert set(report["numeric_stats"]) == {"RA", "DEC", "Z"}
    assert report["numeric_stats"]["Z"]["mean"] == pytest.approx(0.2)
    assert report["categorical_uniques"]["CLASS"] == ["GALAXY", "QSO"]


def test_inspect_fits_parallel_chunk_stats(tmp_path, monkeypatch):
    """Verify FITS stats can be computed through Dask row chunks."""
    from astropy.io import fits
    from astropy.table import Table

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "sample.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(
                Table(
                    {
                        "RA": [10.0, 11.0, 12.0, 13.0],
                        "DEC": [-1.0, -1.1, -1.2, -1.3],
                        "Z": [0.1, 0.2, 0.3, 0.4],
                        "CLASS": ["GALAXY", "QSO", "GALAXY", "STAR"],
                    }
                ),
                name="CATALOG",
            ),
        ]
    ).writeto(path)

    import redshift_catalog_curation_assistant.executor as executor
    import redshift_catalog_curation_assistant.inspect as insp

    monkeypatch.setattr(executor, "dask_client_context", fake_dask_client_context)

    outdir = insp.run_inspect_config(
        {
            "input_file": str(path),
            "survey_name": "FITS_PARALLEL",
            "fits_hdu": 1,
            "parallel_stats": True,
            "fits_stats_batch_size": 2,
            "fits_stats_chunk_rows": 2,
        }
    )
    report = json.loads((outdir / "inspect_report.json").read_text())

    assert report["numeric_stats"]["Z"]["count"] == 4
    assert report["numeric_stats"]["Z"]["mean"] == pytest.approx(0.25)
    assert report["categorical_uniques"]["CLASS"] == ["GALAXY", "QSO", "STAR"]


def test_inspect_large_fits_defaults_to_candidate_stats(tmp_path, monkeypatch):
    """Verify direct large FITS inspection defaults to candidate stats."""
    from astropy.io import fits
    from astropy.table import Table

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "large.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(Table({"RA": [10.0, 11.0], "Z": [0.1, 0.2]}), name="CATALOG"),
        ]
    ).writeto(path)

    import redshift_catalog_curation_assistant.inspect as insp

    outdir = insp.run_inspect_config({"input_file": str(path), "survey_name": "LARGE_FITS", "fits_hdu": 1})
    report = json.loads((outdir / "inspect_report.json").read_text())

    assert set(report["numeric_stats"]) == {"RA", "Z"}
    assert report["categorical_uniques"] == {}


def test_inspect_hats_catalog_uses_lsdb_with_dask_client(tmp_path, monkeypatch):
    """Verify HATS inspection opens LSDB catalogs under a Dask client context."""
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(repo_root)
    calls = []

    @contextmanager
    def tracking_dask_client_context(cluster_config, logs_dir=None):
        calls.append({"cluster_config": cluster_config, "logs_dir": logs_dir})
        yield

    import redshift_catalog_curation_assistant.executor as executor
    import redshift_catalog_curation_assistant.inspect as insp

    monkeypatch.setattr(executor, "dask_client_context", tracking_dask_client_context)

    outdir = insp.run_inspect_config(
        {
            "input_file": "tests/data/raw/elaisfbmc_sample",
            "survey_name": "ELAISFBMC_COLLECTION",
            "output_dir": str(tmp_path / "hats-report"),
            "stats_mode": "candidates",
            "sample_max_columns": 8,
            "sample_seed": 1,
            "column_patterns": {
                "ra": ["^RAdeg$"],
                "dec": ["^DEdeg$"],
                "redshift": ["^zbest$"],
                "quality": ["^zbest_quality$"],
            },
        }
    )
    report = json.loads((outdir / "inspect_report.json").read_text())

    assert calls
    assert report["input_format"] == "hats"
    assert report["n_rows"] == 3762
    assert report["n_columns"] == 85
    assert report["candidates"]["ra"] == ["RAdeg"]
    assert report["candidates"]["dec"] == ["DEdeg"]
    assert report["candidates"]["redshift"] == ["zbest"]
    assert report["numeric_stats"]["zbest"]["count"] > 0
    assert len(report["sample"]) <= 5
    assert set(report["sample"][0]).issubset(set(report["columns"]))


@pytest.mark.parametrize(
    ("config_path", "survey", "n_rows"),
    [
        ("configs/inspect/desi_deep_pilot.example.yaml", "DESI_DEEP_PILOT", 20),
        ("configs/inspect/elaisfbmc.example.yaml", "ELAISFBMC_COLLECTION", 3762),
        ("configs/inspect/synthetic.example.yaml", "SYNTHETIC_REDSHIFT", 5),
    ],
)
def test_inspect_versioned_sample_configs(config_path, survey, n_rows, tmp_path, monkeypatch):
    """Verify versioned sample configs produce inspect reports."""
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(repo_root)

    import redshift_catalog_curation_assistant.executor as executor
    import redshift_catalog_curation_assistant.inspect as insp

    monkeypatch.setattr(executor, "dask_client_context", fake_dask_client_context)

    config = yaml.safe_load(Path(config_path).read_text())
    config["output_dir"] = str(tmp_path / "inspect-report")
    runtime_config = tmp_path / "inspect.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False))

    outdir = insp.run_inspect(runtime_config)
    report = json.loads((outdir / "inspect_report.json").read_text())

    assert report["survey"] == survey
    assert report["n_rows"] == n_rows
    assert report["n_columns"] > 0
