import json
from pathlib import Path

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
