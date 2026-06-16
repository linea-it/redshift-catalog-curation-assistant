import json
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from ..executor import dask_client_context, dask_cluster_config
from ..io import DEFAULT_DASK_THRESHOLD_BYTES, read_table
from ..prepare.prepare import (
    DEFAULT_TARGET_PARTITION_SIZE_MB,
    OUTPUT_MODES,
    PARQUET_SUFFIXES,
    _data_suffix,
    _schema_for_path,
    _target_partition_size,
)

GENERATED_COLUMNS_POSITION = {"first", "last"}
SUPPORTED_TRANSFORMATION_TYPES = {
    "add_constant_column",
    "cast",
    "coalesce_redshift",
    "dec_dms_to_degrees",
    "ra_hms_to_degrees",
    "skycoord_to_degrees",
    "velocity_to_redshift",
}
RA_MIN_DEG = 0.0
RA_MAX_DEG = 360.0
DEC_MIN_DEG = -90.0
DEC_MAX_DEG = 90.0
REDSHIFT_MIN = -0.01
REDSHIFT_MAX = 15.0
REDSHIFT_INVALID_POLICIES = {"fail", "flag"}
DEFAULT_INVALID_REDSHIFT_VALUE = -1.0


class CurateError(ValueError):
    """Raised when local curation cannot continue safely."""


def load_curate_config(path: Path) -> dict[str, Any]:
    """Load a YAML curate configuration file."""
    with open(path, "r") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def _as_paths(paths: list[str | Path]) -> list[Path]:
    return [Path(path) for path in paths]


def _input_paths(config: dict[str, Any]) -> list[Path]:
    if "input_file" in config and "input_files" in config:
        raise CurateError("curate config accepts either input_file or input_files, not both.")
    if "input_files" in config:
        paths = config["input_files"]
        if not isinstance(paths, list | tuple) or not paths:
            raise CurateError("input_files must be a non-empty list of paths.")
        if not all(isinstance(path, str | Path) and str(path).strip() for path in paths):
            raise CurateError("input_files must contain only non-empty path strings.")
        return _as_paths(cast(list[str | Path], list(paths)))
    if "input_file" not in config:
        raise CurateError("curate config requires input_file or input_files.")
    input_file = config["input_file"]
    if not isinstance(input_file, str | Path) or not str(input_file).strip():
        raise CurateError("input_file must be a non-empty path string.")
    return [Path(input_file)]


def _total_input_size(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if path.is_dir():
            total += sum(part.stat().st_size for part in path.rglob("*.parquet"))
        else:
            total += path.stat().st_size
    return total


def _bytes_from_mb(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(float(value) * 1024 * 1024)


def _large_file_threshold_bytes(config: dict[str, Any]) -> int:
    return _bytes_from_mb(
        config.get("large_file_threshold_mb", config.get("dask_threshold_mb")),
        DEFAULT_DASK_THRESHOLD_BYTES,
    )


def _output_mode(config: dict[str, Any]) -> str:
    output_mode = str(config.get("output_mode", "auto")).lower()
    if output_mode not in OUTPUT_MODES:
        raise CurateError("output_mode must be one of: auto, single, partitioned.")
    return output_mode


def _generated_columns_position(config: dict[str, Any]) -> str:
    position = str(config.get("generated_columns_position", "last")).lower()
    if position not in GENERATED_COLUMNS_POSITION:
        raise CurateError("generated_columns_position must be one of: first, last.")
    return position


def _is_parquet_input(path: Path) -> bool:
    if path.is_dir():
        return (path / "_metadata").exists() or any(path.glob("*.parquet")) or any(path.glob("*.pq"))
    return _data_suffix(path) in PARQUET_SUFFIXES


def _all_parquet_inputs(paths: list[Path]) -> bool:
    return all(_is_parquet_input(path) for path in paths)


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise CurateError(f"Output directory already exists: {output_dir}. Set overwrite: true.")
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _column_selection(config: dict[str, Any]) -> list[str]:
    selection = config.get("column_selection", [])
    if isinstance(selection, dict):
        selection = selection.get("keep_columns", [])
    if selection is None:
        return []
    if not isinstance(selection, list | tuple) or not all(isinstance(column, str) for column in selection):
        raise CurateError("column_selection must be a list of column names or keep_columns mapping.")
    return list(dict.fromkeys(column.strip() for column in selection if column.strip()))


def _coordinate_config(config: dict[str, Any]) -> tuple[str, str]:
    coordinates = config.get("coordinates", config.get("coordinate_validation"))
    if not isinstance(coordinates, dict):
        raise CurateError(
            "curate config requires coordinates.ra_column and coordinates.dec_column for output validation."
        )
    ra_column = coordinates.get("ra_column")
    dec_column = coordinates.get("dec_column")
    if not isinstance(ra_column, str) or not ra_column.strip():
        raise CurateError("coordinates.ra_column must be a non-empty string.")
    if not isinstance(dec_column, str) or not dec_column.strip():
        raise CurateError("coordinates.dec_column must be a non-empty string.")
    return ra_column, dec_column


def _redshift_config(config: dict[str, Any]) -> tuple[str, str, float]:
    redshift = config.get("redshift", config.get("redshift_validation"))
    if not isinstance(redshift, dict):
        raise CurateError("curate config requires redshift.column for output validation.")
    column = redshift.get("column")
    if not isinstance(column, str) or not column.strip():
        raise CurateError("redshift.column must be a non-empty string.")
    invalid_policy = str(redshift.get("invalid_policy", "fail")).lower()
    if invalid_policy not in REDSHIFT_INVALID_POLICIES:
        raise CurateError("redshift.invalid_policy must be one of: fail, flag.")
    invalid_value = redshift.get("invalid_value", DEFAULT_INVALID_REDSHIFT_VALUE)
    if not isinstance(invalid_value, int | float) or isinstance(invalid_value, bool):
        raise CurateError("redshift.invalid_value must be numeric.")
    return column, invalid_policy, float(invalid_value)


def _validate_optional_bool(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if value is not None and not isinstance(value, bool):
        raise CurateError(f"{key} must be true or false.")


def _validate_optional_positive_number(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CurateError(f"{key} must be a positive number.")
    if value <= 0:
        raise CurateError(f"{key} must be a positive number.")


def _validate_optional_non_negative_number(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CurateError(f"{key} must be a non-negative number.")
    if value < 0:
        raise CurateError(f"{key} must be a non-negative number.")


def _validate_optional_positive_int(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise CurateError(f"{key} must be a positive integer.")
    if value <= 0:
        raise CurateError(f"{key} must be a positive integer.")


def _validate_optional_string(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise CurateError(f"{key} must be a non-empty string.")


def _validate_column_names(config: dict[str, Any]) -> None:
    column_names = config.get("column_names")
    if column_names is None:
        return
    if not isinstance(column_names, list | tuple) or not column_names:
        raise CurateError("column_names must be a non-empty list of strings.")
    if not all(isinstance(column, str) and column.strip() for column in column_names):
        raise CurateError("column_names must contain only non-empty strings.")


def _require_string(transformation: dict[str, Any], transform_type: str, key: str) -> None:
    value = transformation.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CurateError(f"{transform_type} transformations require {key} to be a non-empty string.")


def _validate_add_constant_column(transformation: dict[str, Any]) -> None:
    _require_string(transformation, "add_constant_column", "name")


def _validate_cast(transformation: dict[str, Any]) -> None:
    _require_string(transformation, "cast", "column")
    _require_string(transformation, "cast", "dtype")


def _validate_ra_hms_to_degrees(transformation: dict[str, Any]) -> None:
    for key in ["output_column", "hours", "minutes", "seconds"]:
        _require_string(transformation, "ra_hms_to_degrees", key)


def _validate_dec_dms_to_degrees(transformation: dict[str, Any]) -> None:
    for key in ["output_column", "degrees", "arcminutes", "arcseconds"]:
        _require_string(transformation, "dec_dms_to_degrees", key)


def _validate_velocity_to_redshift(transformation: dict[str, Any]) -> None:
    _require_string(transformation, "velocity_to_redshift", "velocity_column")
    output_column = transformation.get("output_column")
    if output_column is not None and (not isinstance(output_column, str) or not output_column.strip()):
        raise CurateError(
            "velocity_to_redshift transformations require output_column to be a non-empty string."
        )
    velocity_error_column = transformation.get("velocity_error_column")
    error_output_column = transformation.get("error_output_column")
    if velocity_error_column is not None and (
        not isinstance(velocity_error_column, str) or not velocity_error_column.strip()
    ):
        raise CurateError(
            "velocity_to_redshift transformations require velocity_error_column to be a non-empty string."
        )
    if error_output_column is not None and (
        not isinstance(error_output_column, str) or not error_output_column.strip()
    ):
        raise CurateError(
            "velocity_to_redshift transformations require error_output_column to be a non-empty string."
        )


def _validate_coalesce_redshift(transformation: dict[str, Any]) -> None:
    _require_string(transformation, "coalesce_redshift", "output_column")
    columns = transformation.get("columns")
    if not isinstance(columns, list | tuple) or not columns:
        raise CurateError(
            "coalesce_redshift transformations require columns to be a non-empty list of strings."
        )
    if not all(isinstance(column, str) and column.strip() for column in columns):
        raise CurateError(
            "coalesce_redshift transformations require columns to contain only non-empty strings."
        )
    invalid_value = transformation.get("invalid_value", DEFAULT_INVALID_REDSHIFT_VALUE)
    if isinstance(invalid_value, bool) or not isinstance(invalid_value, int | float):
        raise CurateError("coalesce_redshift invalid_value must be numeric.")


def _validate_skycoord_to_degrees(transformation: dict[str, Any]) -> None:
    for key in ["ra_column", "dec_column", "output_ra_column", "output_dec_column"]:
        _require_string(transformation, "skycoord_to_degrees", key)
    for key in ["ra_unit", "dec_unit"]:
        value = transformation.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise CurateError(f"skycoord_to_degrees transformations require {key} to be a non-empty string.")


TRANSFORMATION_CONFIG_VALIDATORS = {
    "add_constant_column": _validate_add_constant_column,
    "cast": _validate_cast,
    "coalesce_redshift": _validate_coalesce_redshift,
    "dec_dms_to_degrees": _validate_dec_dms_to_degrees,
    "ra_hms_to_degrees": _validate_ra_hms_to_degrees,
    "skycoord_to_degrees": _validate_skycoord_to_degrees,
    "velocity_to_redshift": _validate_velocity_to_redshift,
}


def _validate_transformations(transformations: Any) -> None:
    if transformations is None:
        return
    if not isinstance(transformations, list | tuple):
        raise CurateError("transformations must be a list.")
    for index, transformation in enumerate(transformations):
        if not isinstance(transformation, dict):
            raise CurateError(f"transformations[{index}] must be a mapping.")
        transform_type = transformation.get("type")
        if not isinstance(transform_type, str) or not transform_type.strip():
            raise CurateError(f"transformations[{index}].type must be a non-empty string.")
        if transform_type not in SUPPORTED_TRANSFORMATION_TYPES:
            raise CurateError(f"Unsupported transformation type: {transform_type}.")
        TRANSFORMATION_CONFIG_VALIDATORS[transform_type](transformation)


def _validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise CurateError("curate config must be a YAML mapping.")
    if not config:
        raise CurateError("curate config is empty.")
    _input_paths(config)
    output_dir = config.get("output_dir")
    if not isinstance(output_dir, str | Path) or not str(output_dir).strip():
        raise CurateError("curate config requires a non-empty output_dir.")
    _validate_optional_bool(config, "overwrite")
    _validate_optional_bool(config, "allow_large_single_output")
    _validate_optional_non_negative_number(config, "large_file_threshold_mb")
    _validate_optional_non_negative_number(config, "dask_threshold_mb")
    _validate_optional_positive_number(config, "target_partition_size_mb")
    _validate_optional_positive_int(config, "chunk_size_rows")
    _validate_optional_string(config, "part_prefix")
    _validate_column_names(config)
    _coordinate_config(config)
    _redshift_config(config)
    _column_selection(config)
    _output_mode(config)
    _generated_columns_position(config)
    _validate_transformations(config.get("transformations", []))
    return config


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _transformation_generated_columns(transformation: dict[str, Any]) -> list[str]:
    transform_type = transformation.get("type")
    if transform_type == "add_constant_column":
        return [transformation["name"]] if isinstance(transformation.get("name"), str) else []
    if transform_type in {"ra_hms_to_degrees", "dec_dms_to_degrees", "coalesce_redshift"}:
        if isinstance(transformation.get("output_column"), str):
            return [transformation["output_column"]]
        return []
    if transform_type == "velocity_to_redshift":
        generated = []
        if isinstance(transformation.get("output_column"), str):
            generated.append(transformation["output_column"])
        elif "velocity_column" in transformation:
            generated.append("redshift")
        if transformation.get("velocity_error_column") is not None:
            if isinstance(transformation.get("error_output_column"), str):
                generated.append(transformation["error_output_column"])
            else:
                generated.append("redshift_err")
        return generated
    if transform_type == "skycoord_to_degrees":
        return [
            column
            for column in [transformation.get("output_ra_column"), transformation.get("output_dec_column")]
            if isinstance(column, str)
        ]
    return []


def _transformation_required_columns(transformation: dict[str, Any]) -> list[str]:
    transform_type = transformation.get("type")
    if transform_type == "cast":
        return [transformation["column"]] if isinstance(transformation.get("column"), str) else []
    if transform_type == "ra_hms_to_degrees":
        return [
            column
            for column in [
                transformation.get("hours"),
                transformation.get("minutes"),
                transformation.get("seconds"),
            ]
            if isinstance(column, str)
        ]
    if transform_type == "dec_dms_to_degrees":
        return [
            column
            for column in [
                transformation.get("degrees"),
                transformation.get("arcminutes"),
                transformation.get("arcseconds"),
            ]
            if isinstance(column, str)
        ]
    if transform_type == "velocity_to_redshift":
        return [
            column
            for column in [transformation.get("velocity_column"), transformation.get("velocity_error_column")]
            if isinstance(column, str)
        ]
    if transform_type == "coalesce_redshift":
        columns = transformation.get("columns")
        if isinstance(columns, list | tuple):
            return [column for column in columns if isinstance(column, str)]
        return []
    if transform_type == "skycoord_to_degrees":
        return [
            column
            for column in [transformation.get("ra_column"), transformation.get("dec_column")]
            if isinstance(column, str)
        ]
    return []


def _generated_columns(config: dict[str, Any]) -> list[str]:
    return _ordered_unique(
        [
            column
            for transformation in config.get("transformations") or []
            for column in _transformation_generated_columns(transformation)
        ]
    )


def _required_input_columns(config: dict[str, Any]) -> list[str] | None:
    selection = _column_selection(config)
    if not selection:
        return None

    generated_columns = set(_generated_columns(config))
    ra_column, dec_column = _coordinate_config(config)
    redshift_column, _invalid_policy, _invalid_value = _redshift_config(config)
    required = [column for column in selection if column not in generated_columns]
    required.extend(
        column for column in [ra_column, dec_column, redshift_column] if column not in generated_columns
    )
    for transformation in config.get("transformations") or []:
        required.extend(_transformation_required_columns(transformation))
    return _ordered_unique(required)


def _fits_to_dataframe(path: Path, fits_hdu: int, columns: list[str] | None) -> pd.DataFrame:
    import fitsio

    with fitsio.FITS(path) as fits_file:
        data = fits_file[fits_hdu].read(columns=columns)

    names = list(data.dtype.names or [])
    df_dict = {}
    for name in names:
        column = data[name]
        if column.ndim == 1:
            values = np.asarray(column)
            if values.dtype.byteorder not in ("=", "|"):
                values = values.astype(values.dtype.newbyteorder("="), copy=False)
            df_dict[name] = values
        else:
            df_dict[name] = pd.Series([np.asarray(value).tolist() for value in column], dtype="object")
    return pd.DataFrame(df_dict)


def _read_headerless_table(path: Path, config: dict[str, Any], columns: list[str] | None) -> pd.DataFrame:
    detected_columns = pd.read_csv(path, sep=r"\s+", comment="#", header=None, nrows=1).shape[1]
    column_names = config.get("column_names")
    if not column_names:
        raise CurateError("Headerless curate inputs require column_names.")
    if len(column_names) != detected_columns:
        raise CurateError(
            f"column_names has {len(column_names)} entries, but {path} has {detected_columns} columns."
        )
    return pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=column_names, usecols=columns)


def _read_small_table(path: Path, config: dict[str, Any], columns: list[str] | None) -> pd.DataFrame:
    if columns is None:
        frame = read_table(
            path,
            fits_hdu=int(config.get("fits_hdu", 1)),
            column_names=config.get("column_names"),
            dask_threshold_bytes=None,
        )
        return frame

    suffix = _data_suffix(path)
    try:
        if path.is_dir() or suffix in PARQUET_SUFFIXES:
            return pd.read_parquet(path, columns=columns)
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path, usecols=columns)
        if suffix in {".dat", ".idz"}:
            return _read_headerless_table(path, config, columns)
        if suffix in {".fits", ".fit", ".fts"}:
            return _fits_to_dataframe(path, int(config.get("fits_hdu", 1)), columns)
    except (KeyError, ValueError) as exc:
        raise CurateError(f"Curate input is missing required columns: {', '.join(columns)}.") from exc
    raise CurateError(f"Unsupported curate input format: {suffix or path}")


def _validate_matching_input_schemas(paths: list[Path], config: dict[str, Any]) -> None:
    if len(paths) <= 1:
        return
    schemas = {}
    for path in paths:
        if path.is_dir():
            import pyarrow.dataset as ds

            schemas[path] = list(ds.dataset(path, format="parquet").schema.names)
        else:
            schemas[path] = _schema_for_path(path, _data_suffix(path), config)

    first_path = paths[0]
    first_schema = schemas[first_path]
    mismatches = {path: schema for path, schema in schemas.items() if schema != first_schema}
    if not mismatches:
        return

    lines = ["Multi-file curate inputs must describe one logical catalog with the same schema."]
    lines.append(f"{first_path}: {', '.join(first_schema)}")
    for path, schema in mismatches.items():
        lines.append(f"{path}: {', '.join(schema)}")
    raise CurateError("\n".join(lines))


def _read_small_inputs(paths: list[Path], config: dict[str, Any]) -> pd.DataFrame:
    _validate_matching_input_schemas(paths, config)
    columns = _required_input_columns(config)
    frames = []
    for path in paths:
        frames.append(_read_small_table(path, config, columns))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _validate_matching_dataframes(frames: list[pd.DataFrame], paths: list[Path]) -> None:
    first_columns = list(frames[0].columns)
    mismatches = [
        (path, list(frame.columns))
        for path, frame in zip(paths[1:], frames[1:], strict=False)
        if list(frame.columns) != first_columns
    ]
    if not mismatches:
        return

    lines = ["Multi-file curate inputs must describe one logical catalog with the same schema."]
    lines.append(f"{paths[0]}: {', '.join(first_columns)}")
    for path, columns in mismatches:
        lines.append(f"{path}: {', '.join(columns)}")
    raise CurateError("\n".join(lines))


def _read_large_parquet_inputs(paths: list[Path], config: dict[str, Any]) -> Any:
    import dask.dataframe as dd

    path_strings = [str(path) for path in paths]
    return dd.read_parquet(
        path_strings,
        columns=_required_input_columns(config),
        split_row_groups=True,
        blocksize=_target_partition_size(config),
        aggregate_files=False,
    )


def _read_input(paths: list[Path], config: dict[str, Any], threshold_bytes: int) -> tuple[Any, bool]:
    total_size = _total_input_size(paths)
    if total_size < threshold_bytes:
        return _read_small_inputs(paths, config), False

    if not _all_parquet_inputs(paths):
        raise CurateError(
            "Large raw inputs must be prepared as Parquet before curate. "
            "Run redshift-curator prepare first, then use the prepared Parquet directory."
        )
    return _read_large_parquet_inputs(paths, config), True


def _ensure_columns(df: Any, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise CurateError(f"{context} references missing columns: {', '.join(missing)}.")


def _cast_column(df: Any, transformation: dict[str, Any]) -> tuple[Any, list[str]]:
    column = transformation.get("column")
    dtype = transformation.get("dtype")
    if not isinstance(column, str) or not isinstance(dtype, str):
        raise CurateError("cast transformations require column and dtype strings.")
    _ensure_columns(df, [column], "cast")
    df[column] = df[column].astype(dtype)
    return df, []


def _add_constant_column(df: Any, transformation: dict[str, Any]) -> tuple[Any, list[str]]:
    name = transformation.get("name")
    if not isinstance(name, str) or not name.strip():
        raise CurateError("add_constant_column transformations require a non-empty name.")
    df[name] = transformation.get("value")
    return df, [name]


def _ra_hms_to_degrees(df: Any, transformation: dict[str, Any]) -> tuple[Any, list[str]]:
    output_column = transformation.get("output_column")
    hours = transformation.get("hours")
    minutes = transformation.get("minutes")
    seconds = transformation.get("seconds")
    if not all(isinstance(value, str) for value in [output_column, hours, minutes, seconds]):
        raise CurateError("ra_hms_to_degrees requires output_column, hours, minutes, and seconds.")
    output_column = cast(str, output_column)
    hours = cast(str, hours)
    minutes = cast(str, minutes)
    seconds = cast(str, seconds)
    _ensure_columns(df, [hours, minutes, seconds], "ra_hms_to_degrees")
    df[output_column] = 15.0 * (df[hours] + df[minutes] / 60.0 + df[seconds] / 3600.0)
    return df, [output_column]


def _dec_dms_to_degrees(df: Any, transformation: dict[str, Any]) -> tuple[Any, list[str]]:
    output_column = transformation.get("output_column")
    degrees = transformation.get("degrees")
    arcminutes = transformation.get("arcminutes")
    arcseconds = transformation.get("arcseconds")
    if not all(isinstance(value, str) for value in [output_column, degrees, arcminutes, arcseconds]):
        raise CurateError("dec_dms_to_degrees requires output_column, degrees, arcminutes, and arcseconds.")
    output_column = cast(str, output_column)
    degrees = cast(str, degrees)
    arcminutes = cast(str, arcminutes)
    arcseconds = cast(str, arcseconds)
    _ensure_columns(df, [degrees, arcminutes, arcseconds], "dec_dms_to_degrees")
    df[output_column] = np.sign(df[degrees]) * (
        df[degrees].abs() + df[arcminutes] / 60.0 + df[arcseconds] / 3600.0
    )
    return df, [output_column]


def _velocity_to_redshift(df: Any, transformation: dict[str, Any]) -> tuple[Any, list[str]]:
    from astropy import constants as const
    from astropy import units as u

    velocity_column = transformation.get("velocity_column")
    output_column = transformation.get("output_column", "redshift")
    velocity_error_column = transformation.get("velocity_error_column")
    error_output_column = transformation.get("error_output_column", "redshift_err")
    if not isinstance(velocity_column, str) or not isinstance(output_column, str):
        raise CurateError("velocity_to_redshift requires velocity_column and output_column strings.")
    velocity_column = cast(str, velocity_column)
    output_column = cast(str, output_column)
    _ensure_columns(df, [velocity_column], "velocity_to_redshift")
    c_kms = const.c.to(u.km / u.s).value
    df[output_column] = df[velocity_column] / c_kms
    generated = [output_column]
    if velocity_error_column is not None:
        if not isinstance(velocity_error_column, str) or not isinstance(error_output_column, str):
            raise CurateError("velocity_to_redshift error columns must be strings.")
        velocity_error_column = cast(str, velocity_error_column)
        error_output_column = cast(str, error_output_column)
        _ensure_columns(df, [velocity_error_column], "velocity_to_redshift")
        df[error_output_column] = df[velocity_error_column] / c_kms
        generated.append(error_output_column)
    return df, generated


def _valid_redshift_mask(series: Any) -> Any:
    return (series > REDSHIFT_MIN) & (series < REDSHIFT_MAX)


def _coalesce_redshift(df: Any, transformation: dict[str, Any]) -> tuple[Any, list[str]]:
    output_column = transformation.get("output_column")
    columns = transformation.get("columns")
    invalid_value = transformation.get("invalid_value", DEFAULT_INVALID_REDSHIFT_VALUE)
    if not isinstance(output_column, str) or not output_column.strip():
        raise CurateError("coalesce_redshift requires a non-empty output_column.")
    if (
        not isinstance(columns, list | tuple)
        or not columns
        or not all(isinstance(column, str) for column in columns)
    ):
        raise CurateError("coalesce_redshift requires a non-empty columns list.")
    if not isinstance(invalid_value, int | float) or isinstance(invalid_value, bool):
        raise CurateError("coalesce_redshift invalid_value must be numeric.")
    output_column = cast(str, output_column)
    columns = cast(list[str], list(columns))
    _ensure_columns(df, columns, "coalesce_redshift")

    result = df[columns[0]].where(_valid_redshift_mask(df[columns[0]]), invalid_value)
    for column in columns[1:]:
        candidate = df[column].where(_valid_redshift_mask(df[column]), invalid_value)
        result = result.where(result != invalid_value, candidate)
    df[output_column] = result
    return df, [output_column]


def _skycoord_to_degrees(df: Any, transformation: dict[str, Any]) -> tuple[Any, list[str]]:
    from astropy.coordinates import SkyCoord

    ra_column = transformation.get("ra_column")
    dec_column = transformation.get("dec_column")
    output_ra_column = transformation.get("output_ra_column")
    output_dec_column = transformation.get("output_dec_column")
    ra_unit = transformation.get("ra_unit", "hourangle")
    dec_unit = transformation.get("dec_unit", "deg")
    required = [ra_column, dec_column, output_ra_column, output_dec_column]
    if not all(isinstance(value, str) for value in required):
        raise CurateError(
            "skycoord_to_degrees requires ra_column, dec_column, output_ra_column, and output_dec_column."
        )
    ra_column = cast(str, ra_column)
    dec_column = cast(str, dec_column)
    output_ra_column = cast(str, output_ra_column)
    output_dec_column = cast(str, output_dec_column)
    ra_unit = cast(str, ra_unit)
    dec_unit = cast(str, dec_unit)
    _ensure_columns(df, [ra_column, dec_column], "skycoord_to_degrees")

    if hasattr(df, "map_partitions"):
        meta = df._meta.assign(**{output_ra_column: np.float64(), output_dec_column: np.float64()})

        def convert_partition(partition: pd.DataFrame) -> pd.DataFrame:
            coords = SkyCoord(ra=partition[ra_column], dec=partition[dec_column], unit=(ra_unit, dec_unit))
            partition = partition.copy()
            partition[output_ra_column] = coords.ra.deg
            partition[output_dec_column] = coords.dec.deg
            return partition

        df = df.map_partitions(convert_partition, meta=meta)
        return df, [output_ra_column, output_dec_column]

    coords = SkyCoord(ra=df[ra_column], dec=df[dec_column], unit=(ra_unit, dec_unit))
    df[output_ra_column] = coords.ra.deg
    df[output_dec_column] = coords.dec.deg
    return df, [output_ra_column, output_dec_column]


TRANSFORMATIONS = {
    "add_constant_column": _add_constant_column,
    "cast": _cast_column,
    "ra_hms_to_degrees": _ra_hms_to_degrees,
    "dec_dms_to_degrees": _dec_dms_to_degrees,
    "velocity_to_redshift": _velocity_to_redshift,
    "coalesce_redshift": _coalesce_redshift,
    "skycoord_to_degrees": _skycoord_to_degrees,
}


def _apply_transformations(df: Any, config: dict[str, Any]) -> tuple[Any, list[str]]:
    generated_columns = []
    for transformation in config.get("transformations") or []:
        if not isinstance(transformation, dict):
            raise CurateError("Each transformation must be a mapping.")
        transform_type = transformation.get("type")
        if not isinstance(transform_type, str):
            raise CurateError("Each transformation requires a type string.")
        handler = TRANSFORMATIONS.get(transform_type)
        if handler is None:
            raise CurateError(f"Unsupported transformation type: {transform_type}.")
        df, generated = handler(df, transformation)
        generated_columns.extend(generated)
    return df, list(dict.fromkeys(generated_columns))


def _series_min_max(series: Any) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="raise")
    if hasattr(values, "compute"):
        values = values.compute()
    return float(values.min()), float(values.max())


def _validate_coordinate_range(
    df: Any, column: str, lower: float, upper: float, label: str, include_lower: bool
) -> None:
    _ensure_columns(df, [column], f"{label} validation")
    try:
        min_value, max_value = _series_min_max(df[column])
    except Exception as exc:
        raise CurateError(
            f"{label} column '{column}' must be numeric degrees before curation output. "
            "Use a supported coordinate conversion transformation, then set coordinates."
        ) from exc
    lower_invalid = min_value < lower if include_lower else min_value <= lower
    if lower_invalid or max_value >= upper:
        lower_bracket = "[" if include_lower else "("
        raise CurateError(
            f"{label} column '{column}' is outside the required range {lower_bracket}{lower}, {upper}). "
            f"Observed range: [{min_value}, {max_value}]. "
            "Use a supported coordinate conversion transformation and validate the generated column."
        )


def _validate_coordinates(df: Any, config: dict[str, Any]) -> tuple[str, str]:
    ra_column, dec_column = _coordinate_config(config)
    _validate_coordinate_range(df, ra_column, RA_MIN_DEG, RA_MAX_DEG, "RA", include_lower=True)
    _validate_coordinate_range(df, dec_column, DEC_MIN_DEG, DEC_MAX_DEG, "DEC", include_lower=False)
    return ra_column, dec_column


def _redshift_min_max(series: Any) -> tuple[float, float, int]:
    values = pd.to_numeric(series, errors="raise")
    mask = ~_valid_redshift_mask(values)
    if hasattr(values, "compute"):
        values = values.compute()
        mask = mask.compute()
    return float(values.min()), float(values.max()), int(mask.sum())


def _flag_invalid_redshifts(df: Any, column: str, invalid_value: float) -> Any:
    df[column] = df[column].where(_valid_redshift_mask(df[column]), invalid_value)
    return df


def _validate_redshift(df: Any, config: dict[str, Any]) -> tuple[Any, str]:
    column, invalid_policy, invalid_value = _redshift_config(config)
    _ensure_columns(df, [column], "redshift validation")
    try:
        min_value, max_value, invalid_count = _redshift_min_max(df[column])
    except Exception as exc:
        raise CurateError(
            f"Redshift column '{column}' must be numeric before curation output. "
            "Use a supported redshift transformation, such as velocity_to_redshift or coalesce_redshift."
        ) from exc

    if invalid_count and invalid_policy == "fail":
        raise CurateError(
            f"Redshift column '{column}' is outside the required range ({REDSHIFT_MIN}, {REDSHIFT_MAX}). "
            f"Observed range: [{min_value}, {max_value}]. "
            "Use a supported redshift transformation, such as velocity_to_redshift or coalesce_redshift, "
            "or set redshift.invalid_policy: flag to map invalid values to -1."
        )
    if invalid_count:
        df = _flag_invalid_redshifts(df, column, invalid_value)
    return df, column


def _final_columns(
    df: Any,
    config: dict[str, Any],
    generated_columns: list[str],
    ra_column: str,
    dec_column: str,
    redshift_column: str,
) -> list[str]:
    selection = _column_selection(config)
    if not selection:
        base_columns = [column for column in df.columns if column not in generated_columns]
    else:
        _ensure_columns(df, selection, "column_selection")
        missing_required = [
            column for column in [ra_column, dec_column, redshift_column] if column not in selection
        ]
        if missing_required:
            raise CurateError(
                "column_selection must include validated coordinate and redshift columns: "
                + ", ".join(missing_required)
                + "."
            )
        base_columns = selection

    generated_not_selected = [column for column in generated_columns if column not in base_columns]
    if _generated_columns_position(config) == "first":
        return [*generated_not_selected, *base_columns]
    return [*base_columns, *generated_not_selected]


def _write_single_parquet(df: Any, output_dir: Path, prefix: str) -> int:
    output_path = output_dir / f"{prefix}-part0.parquet"
    if hasattr(df, "compute"):
        df = df.compute()
    df.to_parquet(output_path, index=False)
    return 1


def _write_partitioned_parquet(df: Any, output_dir: Path, prefix: str, output_mode: str) -> int:
    import dask.dataframe as dd

    if not hasattr(df, "npartitions"):
        df = dd.from_pandas(df, npartitions=1)
    if output_mode == "single":
        df = df.repartition(npartitions=1)
    n_parts = int(df.npartitions)
    n_digits = max(len(str(n_parts)), 1)
    df.to_parquet(
        output_dir,
        engine="pyarrow",
        write_index=False,
        overwrite=True,
        name_function=lambda index: f"{prefix}-part{index:0{n_digits}d}.parquet",
    )
    return n_parts


def _resolve_output_mode(config: dict[str, Any], is_small_input: bool) -> str:
    output_mode = _output_mode(config)
    if output_mode == "auto":
        return "single" if is_small_input else "partitioned"
    if (
        output_mode == "single"
        and not is_small_input
        and not bool(config.get("allow_large_single_output", False))
    ):
        raise CurateError(
            "output_mode='single' can concentrate a large catalog into one partition. "
            "Set allow_large_single_output: true if this is intentional."
        )
    return output_mode


def _write_manifest(
    output_dir: Path,
    input_paths: list[Path],
    output_mode: str,
    n_partitions: int,
    final_columns: list[str],
    config: dict[str, Any],
) -> None:
    manifest = {
        "source_paths": [str(path) for path in input_paths],
        "partition_format": "parquet",
        "output_mode": output_mode,
        "n_partitions": n_partitions,
        "columns": final_columns,
        "coordinates": config["coordinates"],
        "redshift": config["redshift"],
        "transformations": config.get("transformations", []),
        "large_file_threshold_mb": float(
            config.get("large_file_threshold_mb", config.get("dask_threshold_mb", 100))
        ),
        "target_partition_size_mb": float(
            config.get("target_partition_size_mb", DEFAULT_TARGET_PARTITION_SIZE_MB)
        ),
    }
    (output_dir / "_redshift_curator_curation_manifest.json").write_text(json.dumps(manifest, indent=2))


def curate_catalog(config: dict[str, Any]) -> Path:
    """Curate a local catalog and write a Parquet dataset."""
    cfg = _validate_config(config)
    input_paths = _input_paths(cfg)
    output_dir = Path(cfg["output_dir"])
    threshold_bytes = _large_file_threshold_bytes(cfg)
    is_small_input = _total_input_size(input_paths) < threshold_bytes
    prefix = str(cfg.get("part_prefix") or output_dir.name)

    df, is_dask = _read_input(input_paths, cfg, threshold_bytes)
    df, generated_columns = _apply_transformations(df, cfg)
    ra_column, dec_column = _validate_coordinates(df, cfg)
    df, redshift_column = _validate_redshift(df, cfg)
    final_columns = _final_columns(df, cfg, generated_columns, ra_column, dec_column, redshift_column)
    df = df[final_columns]

    output_mode = _resolve_output_mode(cfg, is_small_input=is_small_input)
    _prepare_output_dir(output_dir, overwrite=bool(cfg.get("overwrite", False)))
    if output_mode == "single" and not is_dask:
        n_partitions = _write_single_parquet(df, output_dir, prefix)
    else:
        cluster_config = dask_cluster_config(cfg)
        with dask_client_context(cluster_config, logs_dir=output_dir / "logs"):
            n_partitions = _write_partitioned_parquet(df, output_dir, prefix, output_mode)

    _write_manifest(output_dir, input_paths, output_mode, n_partitions, final_columns, cfg)
    return output_dir
