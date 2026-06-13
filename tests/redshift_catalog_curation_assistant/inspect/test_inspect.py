import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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
                    "memory_limit": "2GB",
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


@pytest.mark.parametrize(
    ("config_path", "survey", "n_rows"),
    [
        ("configs/inspect/2dfgrs.example.yaml", "2DFGRS", 1000),
        ("configs/inspect/2dflens.example.yaml", "2DFLENS", 1000),
        ("configs/inspect/6dfgs.example.yaml", "6DFGS", 1000),
        ("configs/inspect/desi_deep_pilot.example.yaml", "DESI_DEEP_PILOT", 1000),
    ],
)
def test_inspect_versioned_sample_configs(config_path, survey, n_rows, tmp_path, monkeypatch):
    """Verify versioned sample configs produce inspect reports."""
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(repo_root)

    import redshift_catalog_curation_assistant.inspect as insp

    outdir = insp.run_inspect(Path(config_path))
    report = json.loads((outdir / "inspect_report.json").read_text())

    assert report["survey"] == survey
    assert report["n_rows"] == n_rows
    assert report["n_columns"] > 0
