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
redshift-curator inspect --path tests/data/raw/6dfgs_sample.csv.gz
redshift-curator inspect configs/synthetic.example.yaml
redshift-curator version
```

The planned commands are already present in the CLI so scripts can start using
the stable command names:

```bash
redshift-curator curate configs/synthetic.example.yaml
redshift-curator qa configs/synthetic.example.yaml
redshift-curator validate-flags configs/synthetic.example.yaml
redshift-curator run configs/synthetic.example.yaml
```

Only `inspect` is functional in phase 0. The other commands are placeholders for
later phases.

## Example

The repository includes a small synthetic catalog at
`tests/data/raw/synthetic_redshift_catalog.csv` and a matching config at
`configs/synthetic.example.yaml`.

Running:

```bash
redshift-curator inspect configs/synthetic.example.yaml
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

Large FITS files at or above the configured threshold are blocked by default
because the current FITS path loads the selected HDU into memory with astropy
and pandas. To keep that behavior explicitly, pass `--load-big-fits` or set
`load_big_fits: true` in YAML. For repeated work on large FITS catalogs, prefer
converting FITS to Parquet once and running later pipeline steps on the Parquet
data.

For quick inspection without writing a YAML config first, use defaults from the
input path:

```bash
redshift-curator inspect --path tests/data/raw/6dfgs_sample.csv.gz
redshift-curator inspect --path tests/data/raw/desi_deep_pilot_sample.fits --fits-hdu 1
```

For supported non-FITS tabular formats, files at or above 100 MB are read with
Dask by default. Dask uses a local cluster with 2 workers, 1 thread per worker,
and 1 GB per worker unless configured otherwise. Tune the threshold with
`--dask-threshold-mb` or with `dask_threshold_mb` in YAML; use `0` to disable
Dask:

```bash
redshift-curator inspect --path large_catalog.csv --dask-threshold-mb 250
```

Partitioned Parquet datasets are also accepted as input directories and are
always read with Dask:

```bash
redshift-curator inspect --path converted_catalog/
```

Advanced users can configure the cluster in YAML:

```yaml
dask_threshold_mb: 250
dask_cluster:
  name: slurm
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
`args.scale.minimum_jobs > 0`:

```bash
redshift-curator inspect --path large_catalog.parquet \
  --dask-cluster '{"name": "slurm", "logs_dir": "reports/slurm-logs", "args": {"instance": {"cores": 4, "processes": 1, "memory": "16GB", "queue": "cpu_bpglsst", "account": "hpc-bpglsst", "interface": "ib0"}, "scale": {"minimum_jobs": 1, "maximum_jobs": 4}}}'
```

For headerless whitespace files, pass column names explicitly. Short schemas can
use `--column-names`; longer schemas are usually easier to review in YAML:

```bash
redshift-curator inspect --path sample.dat.gz --column-names '["RA", "Dec", "z"]'
redshift-curator inspect configs/2dflens.sample.yaml
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

Files without embedded column names require a `column_names` list in the YAML
config. That list is user-supplied metadata and must match the number of columns
in the file. Files that already include column names ignore `column_names`.

See `docs/sample-data.md` for the current local sample-data convention.
