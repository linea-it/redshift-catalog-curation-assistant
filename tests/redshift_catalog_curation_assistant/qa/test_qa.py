import json
import subprocess
import warnings
from datetime import date

import lsdb
import numpy as np
import pandas as pd
import pytest

from redshift_catalog_curation_assistant.qa import dry_run_qa_config, generate_qa_notebook, run_qa_config


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        (
            {"hats_catalog/collection.properties": ""},
            ("hats", "hats_catalog"),
        ),
        (
            {"catalog/c3r2_dr3.parquet": ""},
            ("parquet", "catalog/c3r2_dr3.parquet"),
        ),
        (
            {
                "catalog/part.0.parquet": "",
                "catalog/part.1.parquet": "",
            },
            ("parquet", "catalog"),
        ),
        (
            {"catalog.csv": ""},
            ("csv", "catalog.csv"),
        ),
    ],
)
def test_pzserver_detection_source_recognizes_supported_unzipped_layouts(tmp_path, layout, expected):
    """Ensure archive autodetection resolves a single supported extracted input."""
    from redshift_catalog_curation_assistant.qa.qa import _pzserver_detection_source

    for relative_path, contents in layout.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    namespace = {
        "Path": type(tmp_path),
        "archive_members": [path for path in tmp_path.rglob("*") if path.is_file()],
        "archive_path": tmp_path / "product.zip",
    }

    exec(_pzserver_detection_source(), namespace)

    detected = namespace["qa_input_file"].relative_to(tmp_path).as_posix()
    assert (namespace["qa_input_format"], detected) == expected


def test_pzserver_detection_source_rejects_ambiguous_unzipped_layouts(tmp_path):
    """Ensure archive autodetection fails when multiple supported inputs remain."""
    from redshift_catalog_curation_assistant.qa.qa import _pzserver_detection_source

    for relative_path in ("first.csv", "second.csv"):
        path = tmp_path / relative_path
        path.write_text("", encoding="utf-8")

    namespace = {
        "Path": type(tmp_path),
        "archive_members": [path for path in tmp_path.rglob("*") if path.is_file()],
        "archive_path": tmp_path / "product.zip",
    }

    with pytest.raises(ValueError, match="Expected exactly one HATS, Parquet, or CSV input"):
        exec(_pzserver_detection_source(), namespace)


def test_generate_qa_notebook_reads_local_parquet_and_uses_configured_header_images(tmp_path):
    """Ensure QA notebook generation is driven by local input and configured image attributes."""
    output_notebook = tmp_path / "qa.ipynb"
    first_logo = tmp_path / "first.svg"
    second_logo = tmp_path / "second.png"
    first_logo.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    second_logo.write_bytes(b"fixture")

    generated = generate_qa_notebook(
        {
            "title": "C3R2 DR3",
            "input_file": "curated/c3r2_dr3",
            "output_notebook": str(output_notebook),
            "header_images": [
                {
                    "path": str(first_logo),
                    "width": 100,
                    "style": "padding: 40px",
                },
                {
                    "path": str(second_logo),
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

    assert f'<img align="left" src = "{first_logo}" width=100' in header
    assert 'style="padding: 40px"' in header
    assert f'<img align="left" src = "{second_logo}" width=180>' in header
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
    assert any("catalog.aggregate_column_statistics()" in source for source in sources)
    assert any("curated/hats_catalog" in source for source in sources)
    assert any("max-height: 520px; overflow: auto" in source for source in sources)


def test_generate_qa_notebook_downloads_optional_pzserver_product(tmp_path):
    """Ensure PZ Server products are downloaded before normal catalog loading."""
    output_notebook = tmp_path / "qa.ipynb"
    generate_qa_notebook(
        {
            "input_file": str(tmp_path / "downloaded_data" / "303_c3r2_dr3" / "catalog.parquet"),
            "output_notebook": str(output_notebook),
            "include_absolute_input_path": False,
            "pzs_prod_name": "303_c3r2_dr3",
            "pzs_token_path": str(tmp_path / "token.txt"),
            "pzs_host": "pz",
            "pzs_download_dir": str(tmp_path / "downloaded_data"),
            "pzs_overwrite": True,
        }
    )

    notebook = json.loads(output_notebook.read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    combined = "\n".join(sources)
    assert "from pzserver import PzServer" in combined
    assert "pz_server = PzServer(token=token, host='pz')" in combined
    assert "prod_name = '303_c3r2_dr3'" in combined
    assert "download_path = download_root / prod_name" in combined
    assert "pzs_overwrite = True" in combined
    assert "shutil.rmtree(download_path)" in combined
    assert "pz_server.download_product(product_id=prod_name, save_in=download_path)" in combined
    assert "archive.extractall(download_path)" in combined
    assert "qa_input_file = 'downloaded_data/303_c3r2_dr3/catalog.parquet'" in combined
    assert "qa_input_format = 'parquet'" in combined
    assert "qa_data = qa_open_data()" in combined


def test_generate_qa_notebook_autodetects_unzipped_pzserver_input(tmp_path):
    """Ensure PZ Server inputs can be detected from the extracted archive contents."""
    output_notebook = tmp_path / "qa.ipynb"

    generate_qa_notebook(
        {
            "output_notebook": str(output_notebook),
            "include_absolute_input_path": False,
            "pzs_prod_name": "314_example_hats_product",
            "pzs_token_path": str(tmp_path / "token.txt"),
            "pzs_host": "pz",
            "pzs_download_dir": str(tmp_path / "downloaded_data"),
        }
    )

    notebook = json.loads(output_notebook.read_text())
    combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "collection.properties" in combined
    assert "hats.properties" in combined
    assert "qa_input_format, qa_input_file = candidates[0]" in combined
    assert "**Data access mode:** resolved at runtime after PZ Server download" in combined
    assert "**Input size:** determined from the detected local input after unzip" not in combined
    assert "**Lazy threshold:** 100 MB" not in combined
    assert "**Lazy executor when needed:**" not in combined
    assert "def qa_open_data(columns=None):" in combined
    assert "return dd.read_parquet(qa_input_file, columns=columns)" in combined
    assert "return dd.read_csv(qa_input_file, usecols=columns)" in combined
    assert "return lsdb.open_catalog(qa_input_file, columns=columns)" in combined
    assert "qa_input_size_bytes = qa_path_size_bytes(qa_input_file)" in combined
    assert "qa_access_mode = 'lazy'" in combined
    assert "qa_access_mode = 'in_memory'" in combined
    assert "qa_runtime_summary = pd.Series({" in combined
    assert "'input_size_mb': qa_input_size_bytes / (1024 * 1024)" in combined
    assert "qa_runtime_summary['dask_executor'] = 'not used'" in combined
    assert "qa_data = qa_open_data()" in combined


def test_qa_config_requires_complete_pzserver_configuration():
    """Ensure partially configured PZ Server access fails validation."""
    with pytest.raises(ValueError, match="pzs_token_path"):
        dry_run_qa_config({"input_file": "catalog.parquet", "pzs_prod_name": "123_product"})


def test_qa_config_rejects_non_boolean_pzserver_overwrite():
    """Ensure product replacement requires an explicit boolean setting."""
    with pytest.raises(ValueError, match="pzs_overwrite"):
        dry_run_qa_config(
            {
                "input_file": "catalog.parquet",
                "pzs_prod_name": "123_product",
                "pzs_token_path": "token.txt",
                "pzs_host": "pz",
                "pzs_overwrite": "yes",
            }
        )


def test_qa_config_rejects_pzserver_input_format_override_without_input_file():
    """Ensure PZ override format is only accepted alongside an explicit input path."""
    with pytest.raises(ValueError, match="input_format override requires an input_file override"):
        dry_run_qa_config(
            {
                "pzs_prod_name": "123_product",
                "pzs_token_path": "token.txt",
                "pzs_host": "pz",
                "input_format": "parquet",
            }
        )


def test_generate_qa_notebook_supports_force_compute_with_pzserver_autodetection(tmp_path):
    """Ensure runtime autodetection can still force a full in-memory read after download."""
    output_notebook = tmp_path / "qa.ipynb"

    generate_qa_notebook(
        {
            "output_notebook": str(output_notebook),
            "pzs_prod_name": "123_product",
            "pzs_token_path": str(tmp_path / "token.txt"),
            "pzs_host": "pz",
            "force_compute": True,
        }
    )

    notebook = json.loads(output_notebook.read_text())
    combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "qa_force_compute = True" in combined
    assert "qa_access_mode = 'forced_in_memory'" in combined


def test_generate_qa_notebook_defers_manual_pzserver_input_mode_until_download(tmp_path):
    """Ensure hybrid PZ Server configs resolve access mode at runtime when the local path appears later."""
    output_notebook = tmp_path / "qa.ipynb"

    generate_qa_notebook(
        {
            "output_notebook": str(output_notebook),
            "include_absolute_input_path": False,
            "pzs_prod_name": "314_example_hats_product",
            "pzs_token_path": str(tmp_path / "token.txt"),
            "pzs_host": "pz",
            "pzs_download_dir": str(tmp_path / "downloaded_data"),
            "input_file": str(tmp_path / "downloaded_data" / "314_example_hats_product" / "hats_catalog"),
            "input_format": "hats",
        }
    )

    notebook = json.loads(output_notebook.read_text())
    combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "**Data access mode:** resolved at runtime after PZ Server download" in combined
    assert "qa_input_file = 'downloaded_data/314_example_hats_product/hats_catalog'" in combined
    assert "qa_input_format = 'hats'" in combined
    assert "qa_input_size_bytes = qa_path_size_bytes(qa_input_file)" in combined
    assert "qa_runtime_summary = pd.Series({" in combined


def test_generate_qa_notebook_compiles_for_pzserver_autodetection_with_plots(tmp_path):
    """Ensure the automatic PZ Server path emits syntactically valid code across QA sections."""
    output_notebook = tmp_path / "qa.ipynb"

    generate_qa_notebook(
        {
            "output_notebook": str(output_notebook),
            "pzs_prod_name": "123_product",
            "pzs_token_path": str(tmp_path / "token.txt"),
            "pzs_host": "pz",
            "plots": {
                "spatial": {"ra_column": "ra", "dec_column": "dec"},
                "redshift": {"column": "z", "range": [0, 4]},
                "quality": {"column": "quality"},
            },
        }
    )

    notebook = json.loads(output_notebook.read_text())
    code_sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]

    for index, source in enumerate(code_sources):
        compile(source, f"qa-auto-cell-{index}", "exec")


def test_generate_qa_notebook_can_use_relative_input_path(tmp_path):
    """Ensure users can emit an executable relative path in generated read cells."""
    output_notebook = tmp_path / "reports" / "qa.ipynb"
    input_file = tmp_path / "curated" / "c3r2_dr3.parquet"

    generate_qa_notebook(
        {
            "title": "Relative QA",
            "input_file": str(input_file),
            "include_absolute_input_path": False,
            "output_notebook": str(output_notebook),
        }
    )

    notebook = json.loads(output_notebook.read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]

    assert "df = pd.read_parquet('../curated/c3r2_dr3.parquet')" in sources


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
    input_file = tmp_path / "catalog.parquet"

    generate_qa_notebook(
        {
            "title": "Footprint QA",
            "input_file": str(input_file),
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


def test_run_qa_config_can_execute_notebook_and_export_html(tmp_path, monkeypatch):
    """Ensure optional HTML generation executes a copy and leaves the source notebook blank."""
    import nbclient
    import nbconvert

    execute_calls = []

    class FakeNotebookClient:
        def __init__(self, notebook, timeout, kernel_name, resources, km=None, shutdown_kernel=None):
            self.notebook = notebook
            self.timeout = timeout
            self.kernel_name = kernel_name
            self.resources = resources
            self.km = km
            self.shutdown_kernel = shutdown_kernel

        def execute(self, **kwargs):
            execute_calls.append(kwargs)
            self.notebook["cells"][0]["source"] = ["# Executed QA\n"]

    class FakeHTMLExporter:
        def from_notebook_node(self, notebook):
            return "".join(notebook["cells"][0]["source"]), {}

    monkeypatch.setattr(nbclient, "NotebookClient", FakeNotebookClient)
    monkeypatch.setattr(nbconvert, "HTMLExporter", FakeHTMLExporter)

    input_file = tmp_path / "catalog.csv"
    input_file.write_text("ra,dec,z\n10.0,-1.0,0.1\n11.0,-1.5,0.2\n")
    output_notebook = tmp_path / "reports" / "qa.ipynb"

    artifacts = run_qa_config(
        {
            "title": "Executed QA",
            "input_file": str(input_file),
            "input_format": "csv",
            "include_absolute_input_path": False,
            "output_notebook": str(output_notebook),
            "generate_html": True,
            "html_execution_timeout": 120,
        }
    )

    assert artifacts == {
        "notebook": output_notebook,
        "html": output_notebook.with_suffix(".html"),
    }
    assert artifacts["html"].exists()
    assert "Executed QA" in artifacts["html"].read_text(encoding="utf-8")
    assert len(execute_calls) == 1
    assert execute_calls[0]["independent"] is True
    assert execute_calls[0]["env"]["JPY_PARENT_PID"] == "1"
    assert execute_calls[0]["env"]["IPYTHONDIR"]
    assert "JPY_INTERRUPT_EVENT" not in execute_calls[0]["env"]
    assert "IPY_INTERRUPT_EVENT" not in execute_calls[0]["env"]
    assert execute_calls[0]["stdin"] is subprocess.DEVNULL
    assert execute_calls[0]["stdout"] is subprocess.DEVNULL
    assert execute_calls[0]["stderr"] is subprocess.DEVNULL

    notebook = json.loads(output_notebook.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)


def test_qa_html_generation_rejects_missing_input_file(tmp_path):
    """Ensure optional execution fails before starting when notebook paths are invalid."""
    missing_input = tmp_path / "missing.csv"

    with pytest.raises(ValueError, match="input_file does not exist"):
        run_qa_config(
            {
                "title": "Missing Input QA",
                "input_file": str(missing_input),
                "input_format": "csv",
                "output_notebook": str(tmp_path / "qa.ipynb"),
                "generate_html": True,
            }
        )


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


def test_qa_warns_when_valid_footprint_schema_has_no_usable_curve(tmp_path):
    """Ensure insufficient footprint points produce a notebook warning."""
    footprint = tmp_path / "short_footprint.csv"
    footprint.write_text("region_id,ring_type,vertex_id,ra_deg,dec_deg\n0,exterior,0,10.0,-1.0\n")
    output_notebook = tmp_path / "qa.ipynb"

    generate_qa_notebook(
        {
            "input_file": "catalog.parquet",
            "output_notebook": str(output_notebook),
            "plots": {"spatial": {"footprints": [{"path": str(footprint), "label": "Short curve"}]}},
        }
    )

    notebook = json.loads(output_notebook.read_text())
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "Short curve: footprint has no curve with at least two finite points." in sources


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


@pytest.mark.parametrize(
    ("categorical", "message"),
    [
        ({"column": "TYPE"}, "must be a list"),
        (["TYPE"], r"categorical\[0\] must be a mapping"),
        ([{}], r"categorical\[0\]\.column"),
    ],
)
def test_qa_config_validates_categorical_plots(categorical, message):
    """Ensure generic categorical plots have an unambiguous list schema."""
    with pytest.raises(ValueError, match=message):
        dry_run_qa_config(
            {
                "input_file": "curated/catalog.parquet",
                "plots": {"categorical": categorical},
            }
        )


@pytest.mark.parametrize("rotation", ["45", True, [45]])
def test_qa_config_rejects_invalid_categorical_label_rotation(rotation):
    """Ensure category label rotation is an explicit numeric angle."""
    with pytest.raises(ValueError, match="label_rotation must be a number"):
        dry_run_qa_config(
            {
                "input_file": "curated/catalog.parquet",
                "plots": {"categorical": [{"column": "TYPE", "label_rotation": rotation}]},
            }
        )


@pytest.mark.parametrize("lazy", [False, True])
def test_qa_generates_missing_values_warnings_and_categorical_plots(tmp_path, lazy):
    """Ensure new QA sections generate valid code in both access modes."""
    input_file = tmp_path / "catalog.parquet"
    pd.DataFrame(
        {
            "ra": [np.nan, 400.0],
            "dec": [np.nan, -100.0],
            "z": [10.0, np.nan],
            "quality": [np.nan, np.nan],
            "TYPE": ["GALAXY", None],
            "empty": [np.nan, np.nan],
        }
    ).to_parquet(input_file)
    suffix = "lazy" if lazy else "memory"
    output_notebook = tmp_path / f"qa-{suffix}.ipynb"

    generate_qa_notebook(
        {
            "title": f"Warnings {suffix}",
            "input_file": str(input_file),
            "output_notebook": str(output_notebook),
            "large_input_threshold_mb": 0.000001 if lazy else 100,
            "dask_cluster": {
                "name": "local",
                "args": {"processes": False, "n_workers": 1, "threads_per_worker": 1},
            },
            "plots": {
                "spatial": {"ra_column": "ra", "dec_column": "dec"},
                "redshift": {"column": "z", "range": [0, 1]},
                "quality": {"column": "quality", "label_rotation": 90},
                "categorical": [{"column": "TYPE", "title": "Object types", "label_rotation": 45}],
            },
        }
    )

    notebook = json.loads(output_notebook.read_text())
    code_sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    sources = "\n".join(code_sources)
    for index, source in enumerate(code_sources):
        compile(source, f"qa-cell-{index}", "exec")
    assert "Missing %" in sources
    assert "is entirely null" in sources
    assert "values outside [0, 360)" in sources
    assert "values outside [-90, 90]" in sources
    assert "has no values inside histogram range [0, 1]" in sources
    assert "has no non-null categories" in sources
    assert "Object types" in sources
    assert "plt.xticks(rotation=90)" in sources
    assert "plt.xticks(rotation=45)" in sources


def test_generate_qa_notebook_uses_lazy_partition_aggregations_for_large_parquet(tmp_path):
    """Ensure large Parquet QA never materializes the complete dataframe."""
    input_file = tmp_path / "catalog.parquet"
    pd.DataFrame(
        {
            "ra": [10.0, 20.0, 30.0],
            "dec": [-1.0, 0.0, 1.0],
            "z": [0.1, 0.2, 0.3],
            "z_err": [0.01, 0.02, 0.03],
            "quality": [3, 4, 4],
        }
    ).to_parquet(input_file)
    output_notebook = tmp_path / "reports" / "qa.ipynb"

    generate_qa_notebook(
        {
            "input_file": str(input_file),
            "output_notebook": str(output_notebook),
            "include_absolute_input_path": False,
            "large_input_threshold_mb": 0.000001,
            "dask_cluster": {
                "name": "local",
                "args": {"processes": False, "n_workers": 2},
            },
            "plots": {
                "spatial": {"ra_column": "ra", "dec_column": "dec"},
                "redshift": {"column": "z", "range": [0, 1]},
                "redshift_error": {"column": "z_err"},
                "quality": {"column": "quality"},
            },
        }
    )

    notebook = json.loads(output_notebook.read_text())
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "**Data access mode:** lazy partition aggregation" in sources
    assert "df = dd.read_parquet('../catalog.parquet')" in sources
    assert "dd.read_parquet('/" not in sources
    assert "df = pd.read_parquet" not in sources
    assert "qa_cluster = create_dask_cluster(" in sources
    assert "'processes': False" in sources
    assert "qa_client = Client(qa_cluster)" in sources
    assert "qa_client.close()" in sources
    assert "qa_cluster.close()" in sources
    assert "qa_histogram2d(plot_data, 'ra', 'dec'" in sources
    assert "qa_histogram1d(" in sources
    assert "qa_value_counts(plot_data, 'quality')" in sources
    assert "columns=['ra', 'dec']" in sources
    assert "del plot_data" in sources


def test_generate_qa_notebook_force_compute_overrides_large_input_mode(tmp_path):
    """Ensure users can explicitly accept full in-memory loading for large inputs."""
    input_file = tmp_path / "catalog.parquet"
    pd.DataFrame({"ra": [10.0], "dec": [-1.0], "z": [0.1]}).to_parquet(input_file)
    output_notebook = tmp_path / "qa.ipynb"

    generate_qa_notebook(
        {
            "input_file": str(input_file),
            "output_notebook": str(output_notebook),
            "large_input_threshold_mb": 0.000001,
            "force_compute": True,
        }
    )

    notebook = json.loads(output_notebook.read_text())
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "**Data access mode:** forced in-memory" in sources
    assert f"df = pd.read_parquet('{input_file.resolve()}')" in sources
    assert "force_compute=True" in sources
    assert "dd.read_parquet" not in sources
    assert "create_dask_cluster" not in sources


def test_generate_qa_notebook_uses_public_lazy_hats_operations(tmp_path):
    """Ensure large HATS notebooks use public LSDB projection and aggregation APIs."""
    output_notebook = tmp_path / "qa.ipynb"
    hats_path = "tests/data/raw/elaisfbmc_sample"

    generate_qa_notebook(
        {
            "input_file": hats_path,
            "input_format": "hats",
            "output_notebook": str(output_notebook),
            "include_absolute_input_path": False,
            "large_input_threshold_mb": 0.000001,
            "plots": {"redshift": {"column": "zbest", "range": [0, 5]}},
        }
    )

    notebook = json.loads(output_notebook.read_text())
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "catalog = lsdb.open_catalog(" in sources
    assert "catalog.compute()" not in sources
    assert "catalog.aggregate_column_statistics()" in sources
    assert "columns=['zbest']" in sources
    assert "._ddf" not in sources
    assert "lsdb.open_catalog('/" not in sources


@pytest.mark.parametrize(
    ("kind", "lazy", "expected_snippets", "forbidden_snippets"),
    [
        (
            "parquet",
            False,
            ["df.describe()"],
            ["catalog.aggregate_column_statistics()", "df.describe().compute()"],
        ),
        (
            "csv",
            False,
            ["df.describe()"],
            ["catalog.aggregate_column_statistics()", "df.describe().compute()"],
        ),
        (
            "hats",
            False,
            ["catalog.aggregate_column_statistics()"],
            ["df.describe()", "df.describe().compute()"],
        ),
        (
            "parquet",
            True,
            ["df.describe().compute()"],
            ["catalog.aggregate_column_statistics()", "df.describe()\n"],
        ),
        (
            "csv",
            True,
            ["df.describe().compute()"],
            ["catalog.aggregate_column_statistics()", "df.describe()\n"],
        ),
        (
            "hats",
            True,
            ["catalog.aggregate_column_statistics()", "count_data = lsdb.open_catalog("],
            ["df.describe()", "df.describe().compute()"],
        ),
    ],
)
def test_generate_qa_notebook_uses_expected_basic_statistics_source_by_mode_and_format(
    tmp_path, kind, lazy, expected_snippets, forbidden_snippets
):
    """Ensure each mode/format combination emits the intended basic statistics operation."""
    output_notebook = tmp_path / "qa.ipynb"
    if kind == "parquet":
        input_file = tmp_path / "catalog.parquet"
        pd.DataFrame({"ra": [1.0], "dec": [2.0]}).to_parquet(input_file)
        config = {"input_file": str(input_file)}
    elif kind == "csv":
        input_file = tmp_path / "catalog.csv"
        input_file.write_text("ra,dec\n1.0,2.0\n", encoding="utf-8")
        config = {"input_file": str(input_file), "input_format": "csv"}
    else:
        input_file = "tests/data/raw/elaisfbmc_sample" if lazy else tmp_path / "tiny_hats"
        if not lazy:
            input_path = tmp_path / "tiny_hats"
            input_path.mkdir()
            (input_path / "collection.properties").write_text("", encoding="utf-8")
            input_file = input_path
        config = {"input_file": str(input_file), "input_format": "hats"}
    if lazy:
        config["large_input_threshold_mb"] = 0.000001

    generate_qa_notebook(
        {
            "output_notebook": str(output_notebook),
            **config,
        }
    )

    notebook = json.loads(output_notebook.read_text())
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for snippet in expected_snippets:
        assert snippet in sources
    for snippet in forbidden_snippets:
        assert snippet not in sources


@pytest.mark.parametrize("columns", [None, ["zbest"]])
def test_hats_sample_exposes_public_statistics_api_for_full_and_projected_catalogs(columns):
    """Ensure the public HATS statistics API works for the notebook open patterns."""
    kwargs = {"columns": columns} if columns is not None else {}
    catalog = lsdb.open_catalog("tests/data/raw/elaisfbmc_sample", **kwargs)

    assert hasattr(catalog, "aggregate_column_statistics")

    statistics = catalog.aggregate_column_statistics()
    assert hasattr(statistics, "to_html")
    assert not statistics.empty

    assert "zbest" in statistics.index


def test_generate_qa_notebook_uses_minimal_hats_column_projection_for_lazy_plots(tmp_path):
    """Ensure lazy HATS plot cells project only the columns they actually need."""
    output_notebook = tmp_path / "qa.ipynb"

    generate_qa_notebook(
        {
            "input_file": "tests/data/raw/elaisfbmc_sample",
            "input_format": "hats",
            "output_notebook": str(output_notebook),
            "include_absolute_input_path": False,
            "large_input_threshold_mb": 0.000001,
            "plots": {
                "spatial": {"ra_column": "RAdeg", "dec_column": "DEdeg"},
                "redshift": {"column": "zbest", "range": [0, 5]},
                "redshift_error": {"column": "n_valid_bands", "range": [0, 8]},
                "quality": {"column": "zbest_type"},
            },
        }
    )

    notebook = json.loads(output_notebook.read_text())
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "columns=['RAdeg', 'DEdeg']" in sources
    assert "columns=['zbest']" in sources
    assert "columns=['n_valid_bands']" in sources
    assert "columns=['zbest_type']" in sources


@pytest.mark.parametrize(
    ("config_value", "message"),
    [
        ({"large_input_threshold_mb": 0}, "large_input_threshold_mb"),
        ({"force_compute": "yes"}, "force_compute"),
        ({"dask_cluster": "local"}, "dask_cluster must be a mapping"),
        (
            {"dask_cluster": {"name": "slurm", "args": {}}},
            "SLURM Dask clusters require complete",
        ),
    ],
)
def test_qa_config_validates_large_input_options(config_value, message):
    """Ensure large-input controls reject ambiguous values."""
    with pytest.raises(ValueError, match=message):
        dry_run_qa_config({"input_file": "catalog.parquet", **config_value})


def test_lazy_histogram_derives_range_without_materializing_values():
    """Ensure lazy histograms can derive finite bin edges by partition."""
    import dask.dataframe as dd

    from redshift_catalog_curation_assistant.qa.qa import _lazy_helpers_source

    namespace = {"np": np, "pd": pd, "warnings": warnings}
    exec(_lazy_helpers_source(), namespace)
    source = dd.from_pandas(pd.DataFrame({"z": [0.1, 0.2, np.nan, 0.4]}), npartitions=1)

    counts, edges = namespace["qa_histogram1d"](source, "z", bins=3)

    assert counts.tolist() == [1, 1, 1]
    assert edges.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_lazy_quality_helpers_aggregate_missing_and_diagnostics():
    """Ensure lazy warning inputs are exact partition-wise reductions."""
    import dask.dataframe as dd

    from redshift_catalog_curation_assistant.qa.qa import _lazy_helpers_source

    namespace = {"np": np, "pd": pd, "warnings": warnings}
    exec(_lazy_helpers_source(), namespace)
    source = dd.from_pandas(
        pd.DataFrame(
            {
                "ra": [10.0, np.nan, 361.0],
                "dec": [0.0, np.inf, -91.0],
                "z": [0.1, np.nan, 4.0],
            }
        ),
        npartitions=2,
    )

    missing = namespace["qa_missing_counts"](source)
    spatial = namespace["qa_spatial_diagnostics"](source, "ra", "dec")
    numeric = namespace["qa_numeric_diagnostics"](source, "z", [0, 1])

    assert missing.to_dict() == {"dec": 0, "ra": 1, "z": 1}
    assert spatial.to_dict() == {
        "dec_outside_range": 1,
        "nonfinite_dec": 1,
        "nonfinite_ra": 1,
        "ra_outside_range": 1,
        "valid_pairs": 2,
    }
    assert numeric.to_dict() == {"finite": 2, "in_range": 1}
