import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_default_patterns_cover_common_sample_columns():
    """Verify default patterns cover reusable column names without broad DEC false positives."""
    import redshift_catalog_curation_assistant.inspect as insp

    matches = insp.candidate_columns(
        [
            "OBSRA",
            "OBSDEC",
            "ra_j2000_h",
            "dec_j2000_d",
            "z_helio",
            "qual",
            "ETA_TYPE",
            "MEAN_DELTA_X",
        ],
        insp.PATTERNS,
    )

    assert matches["ra"] == ["OBSRA", "ra_j2000_h"]
    assert matches["dec"] == ["OBSDEC", "dec_j2000_d"]
    assert matches["redshift"] == ["z_helio"]
    assert matches["quality"] == ["qual"]
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
    import redshift_catalog_curation_assistant.inspect as insp

    outdir = insp.run_inspect(cfg)

    out = outdir / "inspect_report.json"
    assert out.exists()
    report = json.loads(out.read_text())
    assert report["n_rows"] == 2
    assert report["n_columns"] == 5
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
        "dask_threshold_mb: 0.000001\n"
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
                    "memory_limit": "1GB",
                    "processes": False,
                    "dashboard_address": None,
                },
            },
            None,
        )
    ]
    assert report["n_rows"] == 2
    assert report["n_columns"] == 5
    assert report["numeric_stats"]["z_phot"]["count"] == 2
    assert report["categorical_uniques"]["kind"] == ["galaxy", "qso"]
    assert report["sample"][0]["id"] == 1


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


@pytest.mark.parametrize(
    ("config_path", "survey", "n_rows"),
    [
        ("configs/2dfgrs.sample.yaml", "2DFGRS", 1000),
        ("configs/2dflens.sample.yaml", "2DFLENS", 1000),
        ("configs/6dfgs.sample.yaml", "6DFGS", 1000),
        ("configs/desi_deep_pilot.example.yaml", "DESI_DEEP_PILOT", 1000),
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
