# redshift-catalog-curation-assistant

A simple, reproducible and extensible tool to aid in the technical curation of
heterogeneous redshift catalogs.

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/luigilcsilva/redshift-catalog-curation-assistant/smoke-test.yml)](https://github.com/luigilcsilva/redshift-catalog-curation-assistant/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/luigilcsilva/redshift-catalog-curation-assistant/branch/main/graph/badge.svg)](https://codecov.io/gh/luigilcsilva/redshift-catalog-curation-assistant)

This project was created following the LINCC Frameworks Python Project Template
(https://lincc-ppt.readthedocs.io/en/latest/).

The project is a local-first Python CLI. It helps a human curator inspect raw
catalogs, identify candidate columns, and prepare explicit curation rules before
later conversion, QA, validation, and cloud execution phases.

## Install

Create and activate the project environment, then install the package in editable
mode:

```bash
conda activate redshift-catalog-curation-assistant
python -m pip install -e '.[dev]'
```

## CLI

After installation, the phase-0 command surface is:

```bash
redshift-curator --help
redshift-curator inspect-fits tests/data/raw/desi_deep_pilot_sample.fits
redshift-curator prepare configs/prepare/desi_deep_pilot.example.yaml
redshift-curator inspect --path tests/data/raw/6dfgs_sample.csv.gz
redshift-curator inspect configs/inspect/synthetic.example.yaml
redshift-curator version
```

The planned commands are already present in the CLI so scripts can start using
the stable command names:

```bash
redshift-curator curate configs/inspect/synthetic.example.yaml
redshift-curator qa configs/inspect/synthetic.example.yaml
redshift-curator validate-flags configs/inspect/synthetic.example.yaml
redshift-curator run configs/inspect/synthetic.example.yaml
```

`inspect`, `inspect-fits`, and `prepare` are functional. The other commands are
placeholders for later phases.

## Example

The repository includes a small synthetic catalog at
`tests/data/raw/synthetic_redshift_catalog.csv` and a matching config at
`configs/inspect/synthetic.example.yaml`.

Running:

```bash
redshift-curator inspect configs/inspect/synthetic.example.yaml
```

writes:

```text
reports/SYNTHETIC_REDSHIFT/inspect_report.json
reports/SYNTHETIC_REDSHIFT/inspect_report.md
```

For FITS catalogs, inspect the HDU structure before choosing `fits_hdu`:

```bash
redshift-curator inspect-fits tests/data/raw/desi_deep_pilot_sample.fits
```

FITS inspection uses FITS headers first and avoids loading the selected HDU into
a pandas DataFrame. For large uncompressed FITS files, statistics and sample rows
are skipped by default unless configured explicitly. Large `.fits.gz` files are
rejected for HDU inspection because gzip-compressed FITS cannot be memory-mapped
efficiently; decompress them first, then run `inspect-fits` or `inspect` on the
uncompressed `.fits` file.

Large or repeated workflows should first normalize raw inputs to partitioned
Parquet:

```bash
redshift-curator prepare configs/prepare/desi_deep_pilot.example.yaml
redshift-curator prepare --path large_catalog.csv --output-dir prepared/large_catalog/ --overwrite
```

`prepare` reads small single files directly in memory, rejects large compressed
files with a decompression suggestion, and uses Dask/chunked reading to write
large CSV/TXT/DAT/IDZ, Parquet, and FITS inputs as multiple Parquet parts. Its
`output_mode` can be `auto`, `single`, or `partitioned`: `auto` writes one part
for small single-file inputs and partitioned output for large or multi-file
inputs. For large or multi-file inputs, `output_mode: single` also requires
`allow_large_single_output: true`.

For quick inspection without writing a YAML config first, use defaults from the
input path:

```bash
redshift-curator inspect --path tests/data/raw/6dfgs_sample.csv.gz
redshift-curator inspect --path tests/data/raw/desi_deep_pilot_sample.fits --fits-hdu 1
```

For raw inputs at or above 100 MB, `inspect` now recommends running `prepare`
first instead of inspecting the large source file directly. This keeps inspection
lightweight and makes the prepared Parquet dataset the canonical input for later
pipeline phases. Tune the threshold with `--dask-threshold-mb` or with
`dask_threshold_mb` in YAML:

```bash
redshift-curator prepare --path large_catalog.csv --output-dir reports/prepared/large_catalog.parquet --overwrite
redshift-curator inspect --path reports/prepared/large_catalog.parquet
```

Large compressed raw inputs are rejected with a decompression command suggestion
before `prepare`. Advanced users can still opt into direct raw inspection with
`--allow-large-raw-inspect` or `allow_large_raw_inspect: true` in YAML.

Partitioned Parquet datasets are also accepted as input directories and are
inspected through the PyArrow dataset path:

```bash
redshift-curator inspect --path converted_catalog/
```

Advanced users can configure the cluster in YAML:

```yaml
dask_threshold_mb: 250
dask_cluster:
  name: slurm
  # Optional. Defaults to the command output directory plus logs/.
  logs_dir: reports/slurm-logs
  args:
    instance:
      cores: 4
      processes: 2
      memory: 8GB
      queue: debug
      account: my-account
    scale:
      minimum_jobs: 1
      maximum_jobs: 4
```

The same executor schema can be passed through the CLI with `--dask-cluster`,
although YAML is usually easier to review. SLURM clusters must be passed as a
full dict with `args`; `--dask-cluster slurm` is intentionally rejected. SLURM
configs require `args.instance.cores`, `args.instance.memory`, and
`args.scale.minimum_jobs > 0`. If `logs_dir` is omitted, SLURM job logs are
written under the command output directory in `logs/`:

```bash
redshift-curator inspect --path large_catalog.parquet \
  --dask-cluster '{"name": "slurm", "logs_dir": "reports/slurm-logs", "args": {"instance": {"cores": 4, "processes": 1, "memory": "16GB", "queue": "cpu_bpglsst", "account": "hpc-bpglsst", "interface": "ib0"}, "scale": {"minimum_jobs": 1, "maximum_jobs": 4}}}'
```

For headerless whitespace files, pass column names explicitly. Short schemas can
use `--column-names`; longer schemas are usually easier to review in YAML:

```bash
redshift-curator inspect --path sample.dat.gz --column-names '["RA", "Dec", "z"]'
redshift-curator inspect configs/inspect/2dflens.example.yaml
```

To inspect only a reviewed subset of columns, pass `column_selection` in YAML:

```yaml
column_selection:
  - RA
  - DEC
  - Z
  - ZWARNING
```

The report keeps `n_columns` as the original catalog width and adds
`n_columns_selected` for the subset used by `columns`, `dtypes`, candidates,
sample rows, and statistics. The same selection can be passed through the CLI:

```bash
redshift-curator inspect --path catalog.fits \
  --column-selection RA --column-selection DEC --column-selection Z

redshift-curator inspect --path catalog.fits \
  --column-selection-list '["RA", "DEC", "Z"]'

redshift-curator inspect --path catalog.fits \
  --stats-mode none --sample-max-columns 25
```

## Development

Run tests and checks with:

```bash
pytest -q
pre-commit run --all-files
```

Large or redistributability-unclear source catalogs should stay outside git.
Small sample files that are safe to redistribute live under `tests/data/raw/`.

## Scientific Naming

Synthetic data in this repository is clearly named synthetic and is not
associated with any real survey. Real survey names are used only for real source
catalogs or local samples derived from those catalogs.

Files without embedded column names require explicit column names, either with
`column_names` in YAML or with `--column-name`/`--column-names` in the CLI. That
list is user-supplied metadata and must match the number of columns in the file.
Files that already include column names ignore `column_names`.

See `docs/sample-data.md` for the current local sample-data convention.
