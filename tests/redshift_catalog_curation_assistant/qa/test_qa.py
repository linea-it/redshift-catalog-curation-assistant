import json
from datetime import date

import pytest

from redshift_catalog_curation_assistant.qa import dry_run_qa_config, generate_qa_notebook


def test_generate_qa_notebook_reads_local_parquet_and_uses_configured_header_images(tmp_path):
    """Ensure QA notebook generation is driven by local input and configured image attributes."""
    output_notebook = tmp_path / "qa.ipynb"

    generated = generate_qa_notebook(
        {
            "title": "C3R2 DR3",
            "input_file": "curated/c3r2_dr3",
            "output_notebook": str(output_notebook),
            "header_images": [
                {
                    "path": "tests/data/images/logo_linea.svg",
                    "width": 100,
                    "style": "padding: 40px",
                },
                {
                    "path": "tests/data/images/logo_rubin.png",
                    "width": 180,
                },
            ],
            "plots": {
                "redshift": {
                    "column": "zspec",
                    "range": [0, 4],
                }
            },
        }
    )

    assert generated == output_notebook
    notebook = json.loads(output_notebook.read_text())
    header = "".join(notebook["cells"][0]["source"])
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]

    assert '<img align="left" src = "tests/data/images/logo_linea.svg" width=100' in header
    assert 'style="padding: 40px"' in header
    assert '<img align="left" src = "tests/data/images/logo_rubin.png" width=180>' in header
    assert "df = pd.read_parquet('" in "\n".join(sources)
    assert "curated/c3r2_dr3" in "\n".join(sources)
    assert "from pzserver import PzServer" not in "\n".join(sources)


def test_generate_qa_notebook_defaults_last_verified_run_to_today(tmp_path):
    """Ensure generated notebooks include the current run date by default."""
    output_notebook = tmp_path / "qa.ipynb"

    generate_qa_notebook(
        {
            "title": "Default Date QA",
            "input_file": "curated/catalog.parquet",
            "output_notebook": str(output_notebook),
        }
    )

    notebook = json.loads(output_notebook.read_text())
    header = "".join(notebook["cells"][0]["source"])

    assert f"Last verified run: **{date.today().isoformat()}**" in header


def test_generate_qa_notebook_can_override_last_verified_run(tmp_path):
    """Ensure configs can still pin a manual verification date."""
    output_notebook = tmp_path / "qa.ipynb"

    generate_qa_notebook(
        {
            "title": "Pinned Date QA",
            "input_file": "curated/catalog.parquet",
            "last_verified_run": "2025-11-11",
            "output_notebook": str(output_notebook),
        }
    )

    notebook = json.loads(output_notebook.read_text())
    header = "".join(notebook["cells"][0]["source"])

    assert "Last verified run: **2025-11-11**" in header


def test_generate_qa_notebook_supports_hats_input(tmp_path):
    """Ensure HATS curate outputs can be opened locally in generated notebooks."""
    output_notebook = tmp_path / "qa.ipynb"

    generate_qa_notebook(
        {
            "title": "HATS QA",
            "input_file": "curated/hats_catalog",
            "input_format": "hats",
            "output_notebook": str(output_notebook),
        }
    )

    notebook = json.loads(output_notebook.read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]

    assert any("import lsdb" in source for source in sources)
    assert any("catalog = lsdb.open_catalog('" in source for source in sources)
    assert any("curated/hats_catalog" in source for source in sources)


def test_generate_qa_notebook_can_use_input_basename(tmp_path):
    """Ensure users can opt out of absolute input paths in generated read cells."""
    output_notebook = tmp_path / "qa.ipynb"

    generate_qa_notebook(
        {
            "title": "Relative QA",
            "input_file": "curated/c3r2_dr3.parquet",
            "include_absolute_input_path": False,
            "output_notebook": str(output_notebook),
        }
    )

    notebook = json.loads(output_notebook.read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]

    assert "df = pd.read_parquet('c3r2_dr3.parquet')" in sources


def test_generate_qa_notebook_supports_multiple_spatial_footprints(tmp_path):
    """Ensure spatial plots can include multiple validated footprint curve files."""
    region_footprint = tmp_path / "region.csv"
    region_footprint.write_text(
        "\n".join(
            [
                "region_id,ring_type,vertex_id,ra_deg,dec_deg",
                "0,exterior,0,10.0,-1.0",
                "0,exterior,1,11.0,-1.0",
            ]
        )
    )
    limit_footprint = tmp_path / "limit.csv"
    limit_footprint.write_text(
        "\n".join(
            [
                ",ra_center,dec_limit",
                "0,10.0,20.0",
                "1,11.0,21.0",
            ]
        )
    )
    output_notebook = tmp_path / "qa.ipynb"

    generate_qa_notebook(
        {
            "title": "Footprint QA",
            "input_file": "curated/catalog",
            "output_notebook": str(output_notebook),
            "plots": {
                "spatial": {
                    "ra_column": "ra",
                    "dec_column": "dec",
                    "footprints": [
                        {
                            "path": str(region_footprint),
                            "label": "Region footprint",
                            "color": "red",
                        },
                        {
                            "path": str(limit_footprint),
                            "label": "Limit footprint",
                            "color": "orange",
                        },
                    ],
                }
            },
        }
    )

    notebook = json.loads(output_notebook.read_text())
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert f"footprint_0 = pd.read_csv('{region_footprint.resolve()}')" in sources
    assert 'for _, footprint_region in footprint_0.groupby("region_id")' in sources
    assert "Region footprint" in sources
    assert f"footprint_1 = pd.read_csv('{limit_footprint.resolve()}')" in sources
    assert 'footprint_1 = footprint_1.sort_values("ra_center")' in sources
    assert "Limit footprint" in sources


def test_generate_qa_notebook_can_use_footprint_basenames(tmp_path):
    """Ensure include_absolute_input_path also controls footprint paths."""
    footprint = tmp_path / "footprint.csv"
    footprint.write_text(
        "\n".join(
            [
                "region_id,ring_type,vertex_id,ra_deg,dec_deg",
                "0,exterior,0,10.0,-1.0",
                "0,exterior,1,11.0,-1.0",
            ]
        )
    )
    output_notebook = tmp_path / "qa.ipynb"

    generate_qa_notebook(
        {
            "title": "Footprint QA",
            "input_file": "curated/catalog.parquet",
            "include_absolute_input_path": False,
            "output_notebook": str(output_notebook),
            "plots": {
                "spatial": {
                    "ra_column": "ra",
                    "dec_column": "dec",
                    "footprints": [
                        {
                            "path": str(footprint),
                        },
                    ],
                }
            },
        }
    )

    notebook = json.loads(output_notebook.read_text())
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "df = pd.read_parquet('catalog.parquet')" in sources
    assert "footprint_0 = pd.read_csv('footprint.csv')" in sources
    assert str(footprint.resolve()) not in sources


def test_qa_config_rejects_unsupported_footprint_columns(tmp_path):
    """Ensure malformed footprint files fail with the expected schema description."""
    footprint = tmp_path / "bad_footprint.csv"
    footprint.write_text("ra,dec\n10.0,-1.0\n")

    with pytest.raises(ValueError, match="Expected either columns region_id, ring_type, vertex_id"):
        dry_run_qa_config(
            {
                "input_file": "curated/catalog",
                "plots": {
                    "spatial": {
                        "footprints": [
                            {
                                "path": str(footprint),
                            }
                        ]
                    }
                },
            }
        )


def test_qa_config_rejects_more_than_two_header_images():
    """Ensure the notebook header stays constrained to at most two configured images."""
    with pytest.raises(ValueError, match="at most two images"):
        dry_run_qa_config(
            {
                "input_file": "curated/catalog",
                "header_images": ["one.png", "two.png", "three.png"],
            }
        )


def test_qa_config_requires_local_input_file():
    """Ensure QA notebook generation is not configured around remote product names."""
    with pytest.raises(ValueError, match="requires input_file"):
        dry_run_qa_config({"title": "Missing input"})
