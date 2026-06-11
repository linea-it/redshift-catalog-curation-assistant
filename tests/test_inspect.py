from pathlib import Path


def test_inspect_sample(tmp_path):
    # Create a small CSV and a config
    data = "id,ra,dec,z_phot,VI_quality\n1,10.0,0.1,0.5,3\n2,11.0,0.2,0.6,4\n"
    csv = tmp_path / "sample.csv"
    csv.write_text(data)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"input_file: {csv}\nsurvey_name: TEST\n")

    # Run inspect
    import redshift_catalog_curation_assistant.inspect as insp

    insp.run_inspect(cfg)

    out = Path("reports") / "TEST" / "inspect_report.json"
    assert out.exists()
