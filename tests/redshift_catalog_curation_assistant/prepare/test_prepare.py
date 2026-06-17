import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
import yaml

from redshift_catalog_curation_assistant.prepare import PrepareError, prepare_catalog


@contextmanager
def fake_dask_client_context(cluster_config, logs_dir=None):
    """Stand in for a distributed Dask client during unit tests."""
    yield


def test_prepare_small_csv_writes_single_parquet(tmp_path):
    """Ensure small single-file inputs can be materialized without Dask."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n2,0.2\n")
    output_dir = tmp_path / "prepared"

    prepared = prepare_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "large_file_threshold_mb": 1,
        }
    )

    assert prepared == output_dir
    parts = sorted(output_dir.glob("*.parquet"))
    assert len(parts) == 1
    manifest = json.loads((output_dir / "_redshift_curator_manifest.json").read_text())
    assert manifest["requested_output_mode"] == "auto"
    assert manifest["output_mode"] == "single"


def test_prepare_rejects_hats_input_as_already_supported(tmp_path):
    """Ensure prepare does not rewrite HATS inputs that are already supported downstream."""
    hats_dir = tmp_path / "catalog"
    hats_dir.mkdir()
    (hats_dir / "hats.properties").write_text("obs_regime=Optical\n")

    with pytest.raises(PrepareError, match="Input is already a HATS catalog"):
        prepare_catalog(
            {
                "input_file": str(hats_dir),
                "output_dir": str(tmp_path / "prepared"),
            }
        )


def test_prepare_small_csv_can_write_hats_collection(tmp_path, monkeypatch):
    """Ensure small inputs can be converted to HATS through LSDB."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z\n1,10.0,-1.0,0.1\n2,11.0,-1.1,0.2\n")
    output_dir = tmp_path / "prepared_hats"
    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)

    prepared = prepare_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "output_format": "hats",
            "large_file_threshold_mb": 1,
            "hats": {
                "catalog_name": "toy",
                "ra_column": "ra",
                "dec_column": "dec",
                "margin_threshold": 5.0,
            },
        }
    )

    assert prepared == output_dir
    assert (output_dir / "collection.properties").exists()
    assert (output_dir / "toy" / "hats.properties").exists()
    assert (output_dir / "toy_5arcs" / "hats.properties").exists()
    manifest = json.loads((output_dir / "_redshift_curator_manifest.json").read_text())
    assert manifest["partition_format"] == "hats"
    assert manifest["hats"]["ra_column"] == "ra"
    assert manifest["hats"]["dec_column"] == "dec"
    assert manifest["hats"]["hats_output_with_margin"] is True
    assert manifest["hats"]["margin_threshold"] == 5.0


def test_prepare_small_hats_output_can_disable_margin(tmp_path, monkeypatch):
    """Ensure users can write HATS output without a margin cache."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z\n1,10.0,-1.0,0.1\n2,11.0,-1.1,0.2\n")
    output_dir = tmp_path / "prepared_hats"
    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)

    prepared = prepare_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "output_format": "hats",
            "large_file_threshold_mb": 1,
            "hats": {
                "catalog_name": "toy",
                "ra_column": "ra",
                "dec_column": "dec",
                "hats_output_with_margin": False,
                "margin_threshold": 5.0,
            },
        }
    )

    assert prepared == output_dir
    assert (output_dir / "collection.properties").exists()
    assert (output_dir / "toy" / "hats.properties").exists()
    assert not list(output_dir.glob("*_arcs"))
    manifest = json.loads((output_dir / "_redshift_curator_manifest.json").read_text())
    assert manifest["hats"]["hats_output_with_margin"] is False
    assert manifest["hats"]["margin_threshold"] is None


def test_prepare_hats_rejects_zero_margin_threshold(tmp_path):
    """Ensure margin threshold zero is rejected when HATS margins are enabled."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z\n1,10.0,-1.0,0.1\n")

    with pytest.raises(PrepareError, match="margin_threshold cannot be 0"):
        prepare_catalog(
            {
                "input_file": str(csv),
                "output_dir": str(tmp_path / "prepared_hats"),
                "output_format": "hats",
                "hats": {
                    "catalog_name": "toy",
                    "ra_column": "ra",
                    "dec_column": "dec",
                    "margin_threshold": 0,
                },
            }
        )


def test_prepare_hats_requires_standard_coordinate_ranges(tmp_path, monkeypatch):
    """Ensure HATS output rejects coordinates outside the standard degree ranges."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z\n1,360.0,-1.0,0.1\n2,11.0,-1.1,0.2\n")
    output_dir = tmp_path / "prepared_hats"
    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)

    with pytest.raises(
        PrepareError,
        match="HATS output requires RA/Dec already in standard degree ranges.*Use prepare to Parquet",
    ):
        prepare_catalog(
            {
                "input_file": str(csv),
                "output_dir": str(output_dir),
                "output_format": "hats",
                "large_file_threshold_mb": 1,
                "hats": {
                    "catalog_name": "toy",
                    "ra_column": "ra",
                    "dec_column": "dec",
                },
            }
        )


def test_prepare_hats_requires_numeric_coordinates(tmp_path, monkeypatch):
    """Ensure HATS output rejects non-numeric coordinate columns."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z\n1,10:00:00,-1.0,0.1\n2,11:00:00,-1.1,0.2\n")
    output_dir = tmp_path / "prepared_hats"
    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)

    with pytest.raises(
        PrepareError,
        match="HATS output requires RA/Dec already in standard degree ranges.*must be numeric",
    ):
        prepare_catalog(
            {
                "input_file": str(csv),
                "output_dir": str(output_dir),
                "output_format": "hats",
                "large_file_threshold_mb": 1,
                "hats": {
                    "catalog_name": "toy",
                    "ra_column": "ra",
                    "dec_column": "dec",
                },
            }
        )


def test_prepare_large_csv_uses_dask_and_writes_partitioned_parquet(tmp_path, monkeypatch):
    """Ensure large CSV files are written through the Dask path."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n2,0.2\n")
    output_dir = tmp_path / "prepared"
    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)

    prepared = prepare_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "large_file_threshold_mb": 0,
            "target_partition_size_mb": 1,
            "dask_cluster": {
                "name": "local",
                "args": {
                    "processes": False,
                },
            },
        }
    )

    assert prepared == output_dir
    assert sorted(output_dir.glob("*.parquet"))


def test_prepare_large_csv_can_write_hats_via_intermediate_parquet(tmp_path, monkeypatch):
    """Ensure large HATS output uses temporary Parquet and removes it after import."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z\n1,10.0,-1.0,0.1\n2,11.0,-1.1,0.2\n")
    output_dir = tmp_path / "prepared_hats"
    calls = []

    def fake_hats_import(parquet_dir, hats_output_dir, config, client, cluster_config):
        calls.append((parquet_dir, hats_output_dir, config, client))
        assert sorted(parquet_dir.glob("*.parquet"))
        (hats_output_dir / "collection.properties").write_text("collection=toy\n")
        catalog_dir = hats_output_dir / "toy"
        catalog_dir.mkdir()
        (catalog_dir / "hats.properties").write_text("catalog=toy\n")

    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)
    monkeypatch.setattr(prep, "_run_hats_import_from_parquet", fake_hats_import)

    prepared = prepare_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "output_format": "hats",
            "large_file_threshold_mb": 0,
            "target_partition_size_mb": 1,
            "hats": {
                "catalog_name": "toy",
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "dask_cluster": {
                "name": "local",
                "args": {
                    "processes": False,
                },
            },
        }
    )

    assert prepared == output_dir
    assert calls
    assert not list(tmp_path.glob(".prepared_hats-parquet-*"))
    manifest = json.loads((output_dir / "_redshift_curator_manifest.json").read_text())
    assert manifest["partition_format"] == "hats"
    assert manifest["output_mode"] == "hats"


def test_prepare_large_hats_validates_coordinates_before_import(tmp_path, monkeypatch):
    """Ensure invalid large-input coordinates stop before hats_import runs."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z\n1,-1.0,-1.0,0.1\n2,11.0,-1.1,0.2\n")
    output_dir = tmp_path / "prepared_hats"
    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)

    def fail_hats_import(*args):
        raise AssertionError("hats_import should not run when coordinates are invalid")

    monkeypatch.setattr(prep, "_run_hats_import_from_parquet", fail_hats_import)

    with pytest.raises(
        PrepareError,
        match="HATS output requires RA/Dec already in standard degree ranges.*observed range",
    ):
        prepare_catalog(
            {
                "input_file": str(csv),
                "output_dir": str(output_dir),
                "output_format": "hats",
                "large_file_threshold_mb": 0,
                "target_partition_size_mb": 1,
                "hats": {
                    "catalog_name": "toy",
                    "ra_column": "ra",
                    "dec_column": "dec",
                },
                "dask_cluster": {
                    "name": "local",
                    "args": {
                        "processes": False,
                    },
                },
            }
        )


def test_prepare_parquet_directory_hats_validates_coordinates_before_import(tmp_path, monkeypatch):
    """Ensure Parquet directory to HATS validates coordinates before hats_import."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    parquet_dir = tmp_path / "prepared_parquet"
    parquet_dir.mkdir()
    pd.DataFrame(
        {
            "object_id": [1, 2],
            "ra": [10.0, 11.0],
            "dec": [-90.0, -1.1],
            "z": [0.1, 0.2],
        }
    ).to_parquet(parquet_dir / "part0.parquet")
    output_dir = tmp_path / "prepared_hats"
    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)

    def fail_hats_import(*args):
        raise AssertionError("hats_import should not run when coordinates are invalid")

    monkeypatch.setattr(prep, "_run_hats_import_from_parquet", fail_hats_import)

    with pytest.raises(
        PrepareError,
        match="HATS output requires RA/Dec already in standard degree ranges.*observed range",
    ):
        prepare_catalog(
            {
                "input_file": str(parquet_dir),
                "output_dir": str(output_dir),
                "output_format": "hats",
                "hats": {
                    "catalog_name": "toy",
                    "ra_column": "ra",
                    "dec_column": "dec",
                },
            }
        )


def test_prepare_multi_file_auto_writes_partitioned_parquet(tmp_path, monkeypatch):
    """Ensure multi-file catalogs are treated as partitioned output in auto mode."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    first = tmp_path / "part1.csv"
    second = tmp_path / "part2.csv"
    first.write_text("object_id,z\n1,0.1\n")
    second.write_text("object_id,z\n2,0.2\n")
    output_dir = tmp_path / "prepared"
    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)

    prepared = prepare_catalog(
        {
            "input_files": [str(first), str(second)],
            "output_dir": str(output_dir),
            "large_file_threshold_mb": 1,
            "dask_cluster": {
                "name": "local",
                "args": {
                    "processes": False,
                },
            },
        }
    )

    assert prepared == output_dir
    assert sorted(output_dir.glob("*.parquet"))


def test_prepare_multi_file_rejects_schema_mismatch(tmp_path):
    """Ensure multi-file inputs must describe one logical catalog."""
    first = tmp_path / "part1.csv"
    second = tmp_path / "part2.csv"
    first.write_text("object_id,z\n1,0.1\n")
    second.write_text("object_id,z,quality\n2,0.2,3\n")

    with pytest.raises(PrepareError, match="same schema"):
        prepare_catalog(
            {
                "input_files": [str(first), str(second)],
                "output_dir": str(tmp_path / "prepared"),
            }
        )


def test_prepare_slurm_defaults_logs_to_output_logs_dir(tmp_path, monkeypatch):
    """Ensure SLURM prepare logs live next to the prepared dataset by default."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")
    output_dir = tmp_path / "prepared"
    client_calls = []

    @contextmanager
    def capture_dask_client_context(cluster_config, logs_dir=None):
        client_calls.append((cluster_config, logs_dir))
        yield

    monkeypatch.setattr(prep, "dask_client_context", capture_dask_client_context)

    prepare_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "large_file_threshold_mb": 0,
            "dask_cluster": {
                "name": "slurm",
                "args": {
                    "instance": {
                        "cores": 1,
                        "memory": "2GB",
                    },
                    "scale": {
                        "minimum_jobs": 1,
                    },
                },
            },
        }
    )

    assert client_calls[0][1] == output_dir / "logs"


def test_prepare_slurm_preserves_explicit_logs_dir(tmp_path, monkeypatch):
    """Ensure explicit SLURM logs_dir overrides the output-local default."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n")
    output_dir = tmp_path / "prepared"
    logs_dir = tmp_path / "custom-logs"
    client_calls = []

    @contextmanager
    def capture_dask_client_context(cluster_config, logs_dir=None):
        client_calls.append((cluster_config, logs_dir))
        yield

    monkeypatch.setattr(prep, "dask_client_context", capture_dask_client_context)

    prepare_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "large_file_threshold_mb": 0,
            "dask_cluster": {
                "name": "slurm",
                "logs_dir": str(logs_dir),
                "args": {
                    "instance": {
                        "cores": 1,
                        "memory": "2GB",
                    },
                    "scale": {
                        "minimum_jobs": 1,
                    },
                },
            },
        }
    )

    assert client_calls[0][1] == logs_dir


def test_prepare_rejects_large_single_output_without_opt_in(tmp_path):
    """Ensure single output for large inputs requires explicit opt-in."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n2,0.2\n")

    with pytest.raises(PrepareError, match="allow_large_single_output"):
        prepare_catalog(
            {
                "input_file": str(csv),
                "output_dir": str(tmp_path / "prepared"),
                "large_file_threshold_mb": 0,
                "output_mode": "single",
            }
        )


def test_prepare_large_single_output_with_opt_in_writes_one_part(tmp_path, monkeypatch):
    """Ensure users can explicitly request one Parquet part for large inputs."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,z\n1,0.1\n2,0.2\n")
    output_dir = tmp_path / "prepared"
    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)

    prepare_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "large_file_threshold_mb": 0,
            "output_mode": "single",
            "allow_large_single_output": True,
            "dask_cluster": {
                "name": "local",
                "args": {
                    "processes": False,
                },
            },
        }
    )

    assert len(sorted(output_dir.glob("*.parquet"))) == 1


def test_prepare_parquet_preserves_arrow_schema_for_object_like_columns(tmp_path, monkeypatch):
    """Ensure Parquet inputs with Arrow list/binary columns can be repartitioned."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import redshift_catalog_curation_assistant.prepare.prepare as prep

    parquet = tmp_path / "sample.parquet"
    table = pa.table(
        {
            "object_id": pa.array([1, 2], type=pa.int64()),
            "label": pa.array([b"GALAXY", b"QSO"], type=pa.binary()),
            "pdf": pa.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], type=pa.list_(pa.float32(), 3)),
        }
    )
    pq.write_table(table, parquet)
    output_dir = tmp_path / "prepared"
    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)

    prepare_catalog(
        {
            "input_file": str(parquet),
            "output_dir": str(output_dir),
            "large_file_threshold_mb": 0,
            "dask_cluster": {
                "name": "local",
                "args": {
                    "processes": False,
                },
            },
        }
    )

    output_schema = pq.read_schema(sorted(output_dir.glob("*.parquet"))[0])
    assert output_schema.field("label").type == table.schema.field("label").type
    assert output_schema.field("pdf").type == table.schema.field("pdf").type


def test_prepare_parquet_splits_input_row_groups(tmp_path):
    """Ensure Parquet prepare does not collapse row groups into one Dask partition."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import redshift_catalog_curation_assistant.prepare.prepare as prep

    parquet = tmp_path / "sample.parquet"
    writer = pq.ParquetWriter(parquet, pa.schema([("object_id", pa.int64()), ("z", pa.float64())]))
    for index in range(3):
        writer.write_table(
            pa.table(
                {
                    "object_id": pa.array([index], type=pa.int64()),
                    "z": pa.array([index / 10], type=pa.float64()),
                }
            )
        )
    writer.close()

    df, input_format = prep._tabular_to_dask_dataframe(
        [parquet],
        {
            "target_partition_size_mb": 1,
        },
    )

    assert input_format == "parquet"
    assert df.npartitions == 3


def test_prepare_parquet_does_not_repartition_before_cluster(tmp_path, monkeypatch):
    """Ensure Parquet inputs avoid driver-side repartition size estimation."""
    import dask.dataframe as dd
    import pandas as pd

    import redshift_catalog_curation_assistant.prepare.prepare as prep

    parquet = tmp_path / "sample.parquet"
    pd.DataFrame({"object_id": [1, 2], "z": [0.1, 0.2]}).to_parquet(parquet)
    output_dir = tmp_path / "prepared"
    original_repartition = dd.DataFrame.repartition

    def fail_on_partition_size(self, *args, **kwargs):
        if "partition_size" in kwargs:
            raise AssertionError("Parquet prepare should not repartition by partition_size before cluster")
        return original_repartition(self, *args, **kwargs)

    monkeypatch.setattr(dd.DataFrame, "repartition", fail_on_partition_size)
    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)

    prepare_catalog(
        {
            "input_file": str(parquet),
            "output_dir": str(output_dir),
            "large_file_threshold_mb": 0,
            "dask_cluster": {
                "name": "local",
                "args": {
                    "processes": False,
                },
            },
        }
    )

    assert sorted(output_dir.glob("*.parquet"))


def test_prepare_large_compressed_file_requires_decompression(tmp_path):
    """Ensure large compressed files fail before Dask processing."""
    compressed = tmp_path / "sample.csv.gz"
    compressed.write_bytes(b"compressed")

    with pytest.raises(PrepareError, match="Decompress the file first"):
        prepare_catalog(
            {
                "input_file": str(compressed),
                "output_dir": str(tmp_path / "prepared"),
                "large_file_threshold_mb": 0,
            }
        )


def test_prepare_fits_uses_fitsio_chunks(tmp_path, monkeypatch):
    """Ensure FITS files can be chunked and written as partitioned Parquet."""
    from astropy.io import fits
    from astropy.table import Table

    import redshift_catalog_curation_assistant.prepare.prepare as prep

    path = tmp_path / "sample.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(Table({"object_id": [1, 2], "z": [0.1, 0.2]}), name="CATALOG"),
        ]
    ).writeto(path)
    output_dir = tmp_path / "prepared"
    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)

    prepared = prepare_catalog(
        {
            "input_file": str(path),
            "output_dir": str(output_dir),
            "large_file_threshold_mb": 0,
            "fits_hdu": 1,
            "chunk_size_rows": 1,
            "dask_cluster": {
                "name": "local",
                "args": {
                    "processes": False,
                },
            },
        }
    )

    assert prepared == output_dir
    assert sorted(output_dir.glob("*.parquet"))


def test_prepare_fits_chunk_size_is_capped_by_target_partition_size():
    """Ensure FITS chunks are bounded by target partition size as well as row count."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    chunk_size = prep._fits_effective_chunk_size(
        configured_chunk_size=200_000,
        row_size_bytes=3_452,
        target_bytes=100 * 1024 * 1024,
    )

    assert chunk_size == 30_375


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/prepare/2dfgrs.example.yaml",
        "configs/prepare/2dflens.example.yaml",
        "configs/prepare/2mrs.example.yaml",
        "configs/prepare/6dfgs.example.yaml",
        "configs/prepare/desi_deep_pilot.example.yaml",
        "configs/prepare/euclid_parquet_sample.example.yaml",
        "configs/prepare/sdss_dr19.example.yaml",
        "configs/prepare/synthetic.example.yaml",
        "configs/prepare/synthetic_hats.example.yaml",
    ],
)
def test_prepare_versioned_sample_configs(config_path, tmp_path, monkeypatch):
    """Verify every versioned prepare config remains executable."""
    import redshift_catalog_curation_assistant.prepare.prepare as prep

    monkeypatch.setattr(prep, "dask_client_context", fake_dask_client_context)

    config = yaml.safe_load(Path(config_path).read_text())
    config["output_dir"] = str(tmp_path / Path(config_path).stem)
    config["dask_cluster"] = None

    output_dir = prepare_catalog(config)

    if config.get("output_format") == "hats":
        assert (output_dir / "collection.properties").exists()
    else:
        assert sorted(output_dir.glob("*.parquet"))
    assert (output_dir / "_redshift_curator_manifest.json").exists()
