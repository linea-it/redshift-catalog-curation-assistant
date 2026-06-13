import json
from contextlib import contextmanager

import pytest

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
