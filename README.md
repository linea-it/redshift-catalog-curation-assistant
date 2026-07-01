# redshift-catalog-curation-assistant

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/luigilcsilva/redshift-catalog-curation-assistant/smoke-test.yml)](https://github.com/luigilcsilva/redshift-catalog-curation-assistant/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/luigilcsilva/redshift-catalog-curation-assistant/branch/main/graph/badge.svg)](https://codecov.io/gh/luigilcsilva/redshift-catalog-curation-assistant)

This project was created following the LINCC Frameworks Python Project Template
(https://lincc-ppt.readthedocs.io/en/latest/).

A local-first CLI for technical curation of heterogeneous redshift catalogs.

The tool helps a human curator inspect catalog structure, prepare local inputs
as Parquet or HATS, apply explicit curation rules, and generate reproducible QA
notebooks. It does not make scientific decisions automatically.

## Install

```bash
conda activate redshift-catalog-curation-assistant
python -m pip install -e '.[dev]'
```

## Quick Start

```bash
redshift-curator --help
redshift-curator inspect configs/inspect/synthetic.example.yaml
redshift-curator inspect-fits tests/data/raw/desi_deep_pilot_sample.fits
redshift-curator prepare configs/prepare/2mrs.example.yaml
redshift-curator curate configs/curate/2mrs.example.yaml
redshift-curator qa configs/qa/2mrs.example.yaml
```

Functional commands:

- `inspect`: build JSON and Markdown inspection reports.
- `inspect-fits`: summarize FITS HDUs without loading table data.
- `prepare`: normalize raw inputs to Parquet datasets or HATS collections.
- `curate`: apply explicit curation rules and write Parquet datasets or HATS
  collections.
- `qa`: generate a local QA notebook and optionally execute it to produce HTML.

Planned command names are also present for later phases: `validate-flags` and
`run`.

## Current Workflow

For small catalogs, inspect and curate can read supported raw files directly.
For large, multi-file, or repeated workflows, prepare first. The full local
workflow is:

```bash
redshift-curator inspect configs/inspect/synthetic.example.yaml
redshift-curator prepare configs/prepare/2mrs.example.yaml
redshift-curator curate configs/curate/2mrs.example.yaml
redshift-curator qa configs/qa/2mrs.example.yaml
```

`prepare` accepts one file or a list of files representing one logical
catalog. Multi-file schemas are strict by default; `schema_policy: union` can
preserve optional columns and fill missing values with nulls. QA plots are
configured independently, so spatial, redshift, redshift-error, and quality
sections can be included only when supported by the catalog. QA notebooks also
include per-column missing-value statistics, objective data warnings, and
optional generic categorical count plots.

QA inputs larger than 100 MB use lazy Dask or LSDB partition aggregations by
default. Only the columns and aggregate bins required by each plot are computed
in memory. The threshold is configurable, and `force_compute: true` explicitly
restores full in-memory loading when required. Lazy QA can use the same local or
SLURM `dask_cluster` configuration as the other distributed pipeline stages.

Supported local examples live under:

- `configs/inspect/`
- `configs/prepare/`
- `configs/curate/`
- `configs/qa/`
- `tests/data/raw/`

## Documentation

Detailed documentation lives in `docs/`:

- `docs/inspect-schema.rst`
- `docs/prepare-schema.rst`
- `docs/curate-schema.rst`
- `docs/curate-transformations.rst`
- `docs/qa-schema.rst`
- `docs/large-data.rst`
- `docs/sample-data.md`

Acknowledgements for public data and other external materials used by test
fixtures and examples are maintained in
`tests/data/acknowledgements.md`. Keep this file updated when adding or changing
data assets.

## Development

```bash
pytest -q
pre-commit run --all-files
```

Large source catalogs and generated pipeline artifacts should not be committed.
Only small, redistributable source fixtures used by tests live under
`tests/data/raw/`; prepared and curated outputs are created in temporary test
directories.
