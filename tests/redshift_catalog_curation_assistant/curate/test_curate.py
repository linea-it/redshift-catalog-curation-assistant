import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from redshift_catalog_curation_assistant.curate import CurateError, curate_catalog
from redshift_catalog_curation_assistant.curate.curate import load_curate_config


@contextmanager
def fake_dask_client_context(cluster_config, logs_dir=None):
    """Stand in for a distributed Dask client during unit tests."""
    yield


def minimal_curate_config() -> dict:
    """Return a minimal config that reaches schema validation before file reads."""
    return {
        "input_file": "missing.csv",
        "output_dir": "missing-output",
        "coordinates": {
            "ra_column": "ra",
            "dec_column": "dec",
        },
        "redshift": {
            "column": "z",
        },
    }


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"input_files": ["a.csv", "b.csv"]}, "either input_file or input_files"),
        ({"input_file": 12}, "input_file must be a non-empty path string"),
        ({"input_file": None, "input_files": [12]}, "either input_file or input_files"),
        ({"column_names": "ra,dec,z"}, "column_names must be a non-empty list"),
        ({"column_names": ["ra", ""]}, "column_names must contain only non-empty strings"),
        ({"column_selection": ["ra", 1]}, "column_selection must be a list"),
        ({"coordinates": "ra,dec"}, "coordinates.ra_column and coordinates.dec_column"),
        ({"redshift": "z"}, "redshift.column"),
        ({"redshift": {"column": "z", "allow_blueshifts": "yes"}}, "redshift.allow_blueshifts"),
        ({"redshift": {"column": "z", "filters": "z < 9"}}, "redshift.filters must be a list"),
        ({"redshift": {"column": "z", "filters": [{"op": "between", "value": 9}]}}, "redshift.filters"),
        ({"redshift": {"column": "z", "filters": [{"op": "<", "value": "9"}]}}, "redshift.filters"),
        ({"overwrite": "yes"}, "overwrite must be true or false"),
        ({"allow_large_single_output": 1}, "allow_large_single_output must be true or false"),
        ({"large_file_threshold_mb": "100"}, "large_file_threshold_mb must be a non-negative number"),
        ({"target_partition_size_mb": 0}, "target_partition_size_mb must be a positive number"),
        ({"chunk_size_rows": 1.5}, "chunk_size_rows must be a positive integer"),
        ({"part_prefix": ""}, "part_prefix must be a non-empty string"),
    ],
)
def test_curate_rejects_invalid_config_fields(update, message):
    """Verify curate config schema errors fail before input files are read."""
    config = minimal_curate_config()
    config.update(update)

    with pytest.raises(CurateError, match=message):
        curate_catalog(config)


@pytest.mark.parametrize(
    ("transformation", "message"),
    [
        ("cast", "transformations\\[0\\] must be a mapping"),
        ({}, "transformations\\[0\\].type must be a non-empty string"),
        ({"type": "unknown"}, "Unsupported transformation type: unknown"),
        ({"type": "cast", "column": "quality"}, "cast transformations require dtype"),
        ({"type": "add_constant_column", "name": ""}, "add_constant_column transformations require name"),
        (
            {"type": "ra_hms_to_degrees", "output_column": "ra"},
            "ra_hms_to_degrees transformations require hours",
        ),
        (
            {"type": "dec_dms_to_degrees", "output_column": "dec", "degrees": "d"},
            "dec_dms_to_degrees transformations require arcminutes",
        ),
        ({"type": "velocity_to_redshift"}, "velocity_to_redshift transformations require velocity_column"),
        (
            {"type": "coalesce_redshift", "output_column": "z", "columns": []},
            "coalesce_redshift transformations require columns",
        ),
        (
            {
                "type": "coalesce_redshift",
                "output_column": "z",
                "columns": ["z_spec", "z_phot"],
                "allow_blueshifts": "yes",
            },
            "coalesce_redshift allow_blueshifts",
        ),
        (
            {
                "type": "coalesce_redshift",
                "output_column": "z",
                "columns": ["z_spec", "z_phot"],
                "label_column": "",
            },
            "coalesce_redshift label_column",
        ),
        (
            {
                "type": "coalesce_redshift",
                "output_column": "z",
                "columns": ["z_spec", "z_phot"],
                "labels": ["s"],
            },
            "coalesce_redshift labels must have the same length",
        ),
        (
            {
                "type": "coalesce_redshift",
                "output_column": "z",
                "columns": ["z_spec", "z_phot"],
                "labels": ["s", ""],
            },
            "coalesce_redshift labels must be a list",
        ),
        (
            {"type": "skycoord_to_degrees", "ra_column": "ra"},
            "skycoord_to_degrees transformations require dec_column",
        ),
    ],
)
def test_curate_rejects_invalid_transformation_configs(transformation, message):
    """Verify transformation-specific config errors are explicit."""
    config = minimal_curate_config()
    config["transformations"] = [transformation]

    with pytest.raises(CurateError, match=message):
        curate_catalog(config)


def test_curate_small_csv_selects_casts_adds_constant_and_writes_single_parquet(tmp_path):
    """Verify small raw inputs can be curated in memory."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z,quality,extra\n1,10.0,-1.0,0.1,3,drop\n")
    output_dir = tmp_path / "curated"

    curated = curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "column_selection": ["object_id", "ra", "dec", "z", "quality"],
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z",
                "invalid_policy": "flag",
            },
            "transformations": [
                {"type": "add_constant_column", "name": "survey_name", "value": "SAMPLE"},
                {"type": "cast", "column": "quality", "dtype": "float64"},
            ],
        }
    )

    assert curated == output_dir
    parts = sorted(output_dir.glob("*.parquet"))
    assert len(parts) == 1
    df = pd.read_parquet(parts[0])
    assert df.columns.tolist() == ["object_id", "ra", "dec", "z", "quality", "survey_name"]
    assert df["quality"].dtype == "float64"
    assert df["survey_name"].tolist() == ["SAMPLE"]
    manifest = json.loads((output_dir / "_redshift_curator_curation_manifest.json").read_text())
    assert manifest["output_mode"] == "single"
    assert manifest["columns"] == df.columns.tolist()


def test_curate_generated_columns_can_be_written_first(tmp_path):
    """Verify generated columns can be placed before selected source columns."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z\n1,10.0,-1.0,0.1\n")
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "column_selection": ["object_id", "ra", "dec", "z"],
            "generated_columns_position": "first",
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z",
                "invalid_policy": "flag",
            },
            "transformations": [
                {"type": "add_constant_column", "name": "survey_name", "value": "SAMPLE"},
            ],
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df.columns.tolist() == ["survey_name", "object_id", "ra", "dec", "z"]


def test_curate_empty_selection_keeps_all_columns_and_appends_generated(tmp_path):
    """Verify empty selection keeps source columns and appends generated columns."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z\n1,10.0,-1.0,0.1\n")
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "column_selection": [],
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z",
                "invalid_policy": "flag",
            },
            "transformations": [
                {"type": "add_constant_column", "name": "survey_name", "value": "SAMPLE"},
            ],
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df.columns.tolist() == ["object_id", "ra", "dec", "z", "survey_name"]


def test_curate_small_csv_pushes_down_required_columns(tmp_path, monkeypatch):
    """Verify explicit selections avoid reading unused CSV columns."""
    import redshift_catalog_curation_assistant.curate.curate as cur

    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z,quality,unused\n1,10.0,-1.0,0.1,3,drop\n")
    output_dir = tmp_path / "curated"
    original_read_csv = pd.read_csv
    usecols_calls = []

    def capture_read_csv(*args, **kwargs):
        if args and args[0] == csv:
            usecols_calls.append(kwargs.get("usecols"))
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(cur.pd, "read_csv", capture_read_csv)

    curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "column_selection": ["object_id", "ra", "dec", "z"],
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z",
            },
        }
    )

    assert ["object_id", "ra", "dec", "z"] in usecols_calls
    assert all(call is None or "unused" not in call for call in usecols_calls)


def test_curate_large_parquet_pushes_down_required_columns(tmp_path, monkeypatch):
    """Verify prepared Parquet inputs are read with a columns projection."""
    import dask.dataframe as dd

    import redshift_catalog_curation_assistant.curate.curate as cur

    parquet = tmp_path / "sample.parquet"
    pd.DataFrame({"object_id": [1], "ra": [10.0], "dec": [-1.0], "z": [0.1], "unused": ["drop"]}).to_parquet(
        parquet
    )
    output_dir = tmp_path / "curated"
    read_calls = []

    def fake_read_parquet(path_strings, **kwargs):
        read_calls.append(kwargs)
        columns = kwargs.get("columns")
        frame = pd.read_parquet(parquet, columns=columns)
        return dd.from_pandas(frame, npartitions=1)

    monkeypatch.setattr(dd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(cur, "dask_client_context", fake_dask_client_context)

    curate_catalog(
        {
            "input_file": str(parquet),
            "output_dir": str(output_dir),
            "overwrite": True,
            "large_file_threshold_mb": 0,
            "column_selection": ["object_id", "ra", "dec", "z"],
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z",
            },
            "dask_cluster": {
                "name": "local",
                "args": {"processes": False},
            },
        }
    )

    assert read_calls[0]["columns"] == ["object_id", "ra", "dec", "z"]


def test_curate_rejects_selected_output_without_validated_coordinates(tmp_path):
    """Verify selected outputs must retain validated RA/DEC columns."""
    csv = tmp_path / "sample.csv"
    csv.write_text("object_id,ra,dec,z\n1,10.0,-1.0,0.1\n")

    with pytest.raises(CurateError, match="column_selection must include validated coordinate and redshift"):
        curate_catalog(
            {
                "input_file": str(csv),
                "output_dir": str(tmp_path / "curated"),
                "column_selection": ["object_id"],
                "coordinates": {
                    "ra_column": "ra",
                    "dec_column": "dec",
                },
                "redshift": {
                    "column": "z",
                },
            }
        )


def test_curate_rejects_invalid_coordinate_ranges(tmp_path):
    """Verify RA/DEC validation stops invalid curated outputs early."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n361.0,-91.0,0.1\n")

    with pytest.raises(CurateError, match="RA column 'ra' is outside the required range"):
        curate_catalog(
            {
                "input_file": str(csv),
                "output_dir": str(tmp_path / "curated"),
                "coordinates": {
                    "ra_column": "ra",
                    "dec_column": "dec",
                },
                "redshift": {
                    "column": "z",
                },
            }
        )


def test_curate_rejects_non_numeric_coordinates_with_conversion_hint(tmp_path):
    """Verify string coordinates fail with a useful conversion hint."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\nnot-ra,-1.0,0.1\n")

    with pytest.raises(CurateError, match="Use a supported coordinate conversion transformation"):
        curate_catalog(
            {
                "input_file": str(csv),
                "output_dir": str(tmp_path / "curated"),
                "coordinates": {
                    "ra_column": "ra",
                    "dec_column": "dec",
                },
                "redshift": {
                    "column": "z",
                },
            }
        )


def test_curate_2dfgrs_hms_dms_conversion_from_headerless_fixture(tmp_path):
    """Verify HMS/DMS coordinate conversions on the 2dFGRS sample fixture."""
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": "tests/data/raw/2dfgrs_sample.idz.gz",
            "output_dir": str(output_dir),
            "overwrite": True,
            "column_names": [
                "serial",
                "spectra",
                "name",
                "UKST",
                "ra_b1950_h",
                "ra_b1950_m",
                "ra_b1950_s",
                "dec_b1950_d",
                "dec_b1950_m",
                "dec_b1950_s",
                "ra_j2000_h",
                "ra_j2000_m",
                "ra_j2000_s",
                "dec_j2000_d",
                "dec_j2000_m",
                "dec_j2000_s",
                "BJG",
                "BJSEL",
                "BJG_OLD",
                "BJSELOLD",
                "GALEXT",
                "SB_BJ",
                "SR_R",
                "z",
                "z_helio",
                "obsrun",
                "quality",
                "abemma",
                "Z_ABS",
                "KBESTR",
                "R_CRCOR",
                "Z_EMI",
                "NMBEST",
                "SNR",
                "ETA_TYPE",
            ],
            "column_selection": ["serial", "ra_j2000_deg", "dec_b1950_deg", "z"],
            "coordinates": {
                "ra_column": "ra_j2000_deg",
                "dec_column": "dec_b1950_deg",
            },
            "redshift": {
                "column": "z",
                "invalid_policy": "flag",
            },
            "transformations": [
                {
                    "type": "ra_hms_to_degrees",
                    "output_column": "ra_j2000_deg",
                    "hours": "ra_j2000_h",
                    "minutes": "ra_j2000_m",
                    "seconds": "ra_j2000_s",
                },
                {
                    "type": "dec_dms_to_degrees",
                    "output_column": "dec_b1950_deg",
                    "degrees": "dec_b1950_d",
                    "arcminutes": "dec_b1950_m",
                    "arcseconds": "dec_b1950_s",
                },
            ],
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df.columns.tolist() == ["serial", "ra_j2000_deg", "dec_b1950_deg", "z"]
    assert df["ra_j2000_deg"].between(0.0, 360.0, inclusive="left").all()
    assert ((df["dec_b1950_deg"] > -90.0) & (df["dec_b1950_deg"] < 90.0)).all()
    assert df["z"].min() >= -1.0


def test_curate_6dfgs_skycoord_conversion(tmp_path):
    """Verify SkyCoord hourangle/degree conversion on the 6dFGS sample fixture."""
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": "tests/data/raw/6dfgs_sample.csv.gz",
            "output_dir": str(output_dir),
            "overwrite": True,
            "column_selection": ["SPECID", "RA_deg", "DEC_deg", "Z"],
            "coordinates": {
                "ra_column": "RA_deg",
                "dec_column": "DEC_deg",
            },
            "redshift": {
                "column": "Z",
                "invalid_policy": "flag",
            },
            "transformations": [
                {
                    "type": "skycoord_to_degrees",
                    "ra_column": "OBSRA",
                    "dec_column": "OBSDEC",
                    "output_ra_column": "RA_deg",
                    "output_dec_column": "DEC_deg",
                    "ra_unit": "hourangle",
                    "dec_unit": "deg",
                }
            ],
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df["RA_deg"].between(0.0, 360.0, inclusive="left").all()
    assert ((df["DEC_deg"] > -90.0) & (df["DEC_deg"] < 90.0)).all()


def test_curate_velocity_to_redshift_conversion(tmp_path):
    """Verify velocity and velocity-error conversion to redshift."""
    csv = tmp_path / "velocity.csv"
    csv.write_text("ra,dec,V,EV\n10.0,-1.0,299792.458,299.792458\n")
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "column_selection": ["ra", "dec", "redshift", "redshift_err"],
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "redshift",
            },
            "transformations": [
                {
                    "type": "velocity_to_redshift",
                    "velocity_column": "V",
                    "output_column": "redshift",
                    "velocity_error_column": "EV",
                    "error_output_column": "redshift_err",
                }
            ],
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df["redshift"].iloc[0] == pytest.approx(1.0)
    assert df["redshift_err"].iloc[0] == pytest.approx(0.001)


def test_curate_rejects_invalid_redshift_by_default(tmp_path):
    """Verify redshift validation fails early by default."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,99.0\n")

    with pytest.raises(CurateError, match="Redshift column 'z' is outside the required range"):
        curate_catalog(
            {
                "input_file": str(csv),
                "output_dir": str(tmp_path / "curated"),
                "coordinates": {
                    "ra_column": "ra",
                    "dec_column": "dec",
                },
                "redshift": {
                    "column": "z",
                },
            }
        )


def test_curate_can_flag_invalid_redshifts(tmp_path):
    """Verify users can opt into mapping invalid redshifts to the standard flag."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,0.1\n11.0,-1.1,99.0\n12.0,-1.2,-0.2\n")
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z",
                "invalid_policy": "flag",
            },
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df["z"].tolist() == [0.1, -1.0, -1.0]


def test_curate_allows_small_blueshifts_by_default(tmp_path):
    """Verify the default redshift range preserves the existing blueshift allowance."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,-0.005\n")
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z",
            },
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df["z"].tolist() == [-0.005]


def test_curate_can_disallow_blueshifts(tmp_path):
    """Verify allow_blueshifts false tightens the standard redshift range."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,-0.005\n")

    with pytest.raises(CurateError, match=r"required range \(0.0, 20.0\)"):
        curate_catalog(
            {
                "input_file": str(csv),
                "output_dir": str(tmp_path / "curated"),
                "coordinates": {
                    "ra_column": "ra",
                    "dec_column": "dec",
                },
                "redshift": {
                    "column": "z",
                    "allow_blueshifts": False,
                },
            }
        )


def test_curate_can_flag_blueshifts_when_disallowed(tmp_path):
    """Verify disallowed blueshifts can be mapped to the invalid-redshift flag."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,0.1\n11.0,-1.1,-0.005\n12.0,-1.2,0.0\n")
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z",
                "allow_blueshifts": False,
                "invalid_policy": "flag",
            },
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df["z"].tolist() == [0.1, -1.0, -1.0]


def test_curate_filters_redshift_range(tmp_path):
    """Verify users can filter curated rows by final redshift comparisons."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,0.5\n11.0,-1.1,1.6\n12.0,-1.2,2.0\n13.0,-1.3,9.0\n")
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z",
                "filters": [
                    {"op": ">", "value": 1.6},
                    {"op": "<", "value": 9},
                ],
            },
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df["z"].tolist() == [2.0]


def test_curate_redshift_filters_exclude_flagged_invalids(tmp_path):
    """Verify flagged invalid redshifts are removed when any redshift filter is configured."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,0.5\n11.0,-1.1,99.0\n12.0,-1.2,-0.2\n")
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z",
                "invalid_policy": "flag",
                "filters": [
                    {"op": "<", "value": 9},
                ],
            },
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df["z"].tolist() == [0.5]


def test_curate_coalesces_redshift_columns_by_validity(tmp_path):
    """Verify z_final can be generated from prioritized redshift candidates."""
    csv = tmp_path / "sample.csv"
    csv.write_text(
        "ra,dec,z_spec,z_phot\n" "10.0,-1.0,0.10,0.20\n" "11.0,-1.1,99.00,0.30\n" "12.0,-1.2,-0.2,99.00\n"
    )
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "column_selection": ["ra", "dec", "z_final"],
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z_final",
                "invalid_policy": "flag",
            },
            "transformations": [
                {
                    "type": "coalesce_redshift",
                    "output_column": "z_final",
                    "columns": ["z_spec", "z_phot"],
                }
            ],
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df["z_final"].tolist() == [0.1, 0.3, -1.0]


def test_curate_coalesce_redshift_writes_source_labels(tmp_path):
    """Verify coalesce_redshift can record which candidate supplied the final redshift."""
    csv = tmp_path / "sample.csv"
    csv.write_text(
        "ra,dec,z_spec,z_phot\n" "10.0,-1.0,0.10,0.20\n" "11.0,-1.1,99.00,0.30\n" "12.0,-1.2,-0.2,99.00\n"
    )
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "column_selection": ["ra", "dec", "z_final", "z_source"],
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z_final",
                "invalid_policy": "flag",
            },
            "transformations": [
                {
                    "type": "coalesce_redshift",
                    "output_column": "z_final",
                    "columns": ["z_spec", "z_phot"],
                    "labels": ["s", "p"],
                    "label_column": "z_source",
                    "invalid_label": "none",
                }
            ],
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df["z_final"].tolist() == [0.1, 0.3, -1.0]
    assert df["z_source"].tolist() == ["s", "p", "none"]


def test_curate_coalesce_redshift_label_defaults_to_column_names(tmp_path):
    """Verify coalesce labels default to candidate column names."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z_spec,z_phot\n10.0,-1.0,99.00,0.20\n")
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "column_selection": ["ra", "dec", "z_final", "z_source"],
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z_final",
                "invalid_policy": "flag",
            },
            "transformations": [
                {
                    "type": "coalesce_redshift",
                    "output_column": "z_final",
                    "columns": ["z_spec", "z_phot"],
                    "label_column": "z_source",
                }
            ],
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df["z_source"].tolist() == ["z_phot"]


def test_curate_coalesce_redshift_respects_disallowed_blueshifts(tmp_path):
    """Verify coalesce_redshift uses redshift.allow_blueshifts by default."""
    csv = tmp_path / "sample.csv"
    csv.write_text("ra,dec,z_spec,z_phot\n10.0,-1.0,-0.005,0.20\n11.0,-1.1,-0.004,-0.003\n")
    output_dir = tmp_path / "curated"

    curate_catalog(
        {
            "input_file": str(csv),
            "output_dir": str(output_dir),
            "overwrite": True,
            "column_selection": ["ra", "dec", "z_final"],
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z_final",
                "allow_blueshifts": False,
                "invalid_policy": "flag",
            },
            "transformations": [
                {
                    "type": "coalesce_redshift",
                    "output_column": "z_final",
                    "columns": ["z_spec", "z_phot"],
                }
            ],
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert df["z_final"].tolist() == [0.2, -1.0]


def test_curate_multi_parquet_large_input_writes_partitioned_output(tmp_path, monkeypatch):
    """Verify prepared multi-Parquet inputs can be curated through the large-input path."""
    import redshift_catalog_curation_assistant.curate.curate as cur

    first = tmp_path / "part1.parquet"
    second = tmp_path / "part2.parquet"
    pd.DataFrame({"object_id": [1], "ra": [10.0], "dec": [-1.0], "z": [0.1]}).to_parquet(first)
    pd.DataFrame({"object_id": [2], "ra": [11.0], "dec": [-1.1], "z": [0.2]}).to_parquet(second)
    output_dir = tmp_path / "curated"
    monkeypatch.setattr(cur, "dask_client_context", fake_dask_client_context)

    curated = curate_catalog(
        {
            "input_files": [str(first), str(second)],
            "output_dir": str(output_dir),
            "overwrite": True,
            "large_file_threshold_mb": 0,
            "coordinates": {
                "ra_column": "ra",
                "dec_column": "dec",
            },
            "redshift": {
                "column": "z",
            },
            "dask_cluster": {
                "name": "local",
                "args": {
                    "processes": False,
                },
            },
        }
    )

    assert curated == output_dir
    assert sorted(output_dir.glob("*.parquet"))
    manifest = json.loads((output_dir / "_redshift_curator_curation_manifest.json").read_text())
    assert manifest["output_mode"] == "partitioned"


def test_curate_large_raw_input_requires_prepare(tmp_path):
    """Verify large non-Parquet inputs fail with a prepare hint."""
    csv = tmp_path / "large.csv"
    csv.write_text("ra,dec,z\n10.0,-1.0,0.1\n")

    with pytest.raises(CurateError, match="Run redshift-curator prepare first"):
        curate_catalog(
            {
                "input_file": str(csv),
                "output_dir": str(tmp_path / "curated"),
                "large_file_threshold_mb": 0,
                "coordinates": {
                    "ra_column": "ra",
                    "dec_column": "dec",
                },
                "redshift": {
                    "column": "z",
                },
            }
        )


def test_curate_multi_file_rejects_schema_mismatch(tmp_path):
    """Verify multi-file inputs must describe one logical catalog."""
    first = tmp_path / "part1.csv"
    second = tmp_path / "part2.csv"
    first.write_text("ra,dec,z\n10.0,-1.0,0.1\n")
    second.write_text("ra,dec,quality\n11.0,-1.1,3\n")

    with pytest.raises(CurateError, match="same schema"):
        curate_catalog(
            {
                "input_files": [str(first), str(second)],
                "output_dir": str(tmp_path / "curated"),
                "coordinates": {
                    "ra_column": "ra",
                    "dec_column": "dec",
                },
                "redshift": {
                    "column": "z",
                },
            }
        )


def test_curate_2mrs_velocity_fixture(tmp_path):
    """Verify 2MRS velocity conversion on the versioned FITS sample."""
    path = Path("tests/data/raw/2mrs_sample.fits")
    output_dir = tmp_path / "curated"
    curate_catalog(
        {
            "input_file": str(path),
            "output_dir": str(output_dir),
            "overwrite": True,
            "column_selection": ["RA", "DEC", "redshift", "redshift_err"],
            "coordinates": {
                "ra_column": "RA",
                "dec_column": "DEC",
            },
            "redshift": {
                "column": "redshift",
            },
            "transformations": [
                {
                    "type": "velocity_to_redshift",
                    "velocity_column": "V",
                    "output_column": "redshift",
                    "velocity_error_column": "EV",
                    "error_output_column": "redshift_err",
                }
            ],
        }
    )

    df = pd.read_parquet(sorted(output_dir.glob("*.parquet"))[0])
    assert {"redshift", "redshift_err"}.issubset(df.columns)


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/curate/synthetic.example.yaml",
        "configs/curate/2dfgrs.example.yaml",
        "configs/curate/2dflens.example.yaml",
        "configs/curate/2mrs.example.yaml",
        "configs/curate/6dfgs.example.yaml",
        "configs/curate/desi_deep_pilot.example.yaml",
        "configs/curate/euclid_parquet_sample.example.yaml",
        "configs/curate/sdss_dr19.example.yaml",
    ],
)
def test_curate_versioned_sample_configs(config_path, tmp_path):
    """Verify versioned curate configs remain executable on small fixtures."""
    config = load_curate_config(Path(config_path))
    config["output_dir"] = str(tmp_path / Path(config_path).stem)

    output_dir = curate_catalog(config)

    assert sorted(output_dir.glob("*.parquet"))
    assert (output_dir / "_redshift_curator_curation_manifest.json").exists()
