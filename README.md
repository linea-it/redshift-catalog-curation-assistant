# redshift-catalog-curation-assistant

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/luigilcsilva/redshift-catalog-curation-assistant/smoke-test.yml)](https://github.com/luigilcsilva/redshift-catalog-curation-assistant/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/luigilcsilva/redshift-catalog-curation-assistant/branch/main/graph/badge.svg)](https://codecov.io/gh/luigilcsilva/redshift-catalog-curation-assistant)

This project was created following the LINCC Frameworks Python Project Template
(https://lincc-ppt.readthedocs.io/en/latest/).

A local-first CLI for technical curation of heterogeneous redshift catalogs.

The tool helps a human curator inspect catalog structure, prepare large inputs
as Parquet, and apply explicit curation rules. It does not make scientific
decisions automatically.

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
redshift-curator prepare configs/prepare/desi_deep_pilot.example.yaml
redshift-curator curate configs/curate/synthetic.example.yaml
```

Functional commands:

- `inspect`: build JSON and Markdown inspection reports.
- `inspect-fits`: summarize FITS HDUs without loading table data.
- `prepare`: normalize raw inputs to Parquet datasets.
- `curate`: apply explicit curation rules and write Parquet datasets.

Planned command names are also present for later phases: `qa`,
`validate-flags`, and `run`.

## Current Workflow

For small catalogs, inspect and curate can read supported raw files directly.
For large or repeated workflows, prepare first:

```bash
redshift-curator prepare configs/prepare/sdss_dr19.example.yaml
redshift-curator inspect --path outputs/prepared/sdss_dr19.parquet
redshift-curator curate configs/curate/sdss_dr19.example.yaml
```

Supported local examples live under:

- `configs/inspect/`
- `configs/prepare/`
- `configs/curate/`
- `tests/data/raw/`

## Documentation

Detailed documentation lives in `docs/`:

- `docs/inspect-schema.rst`
- `docs/prepare-schema.rst`
- `docs/curate-schema.rst`
- `docs/curate-transformations.rst`
- `docs/large-data.rst`
- `docs/sample-data.md`

## Development

```bash
pytest -q
pre-commit run --all-files
```

Large source catalogs and generated artifacts should not be committed. Small,
redistributable fixtures used by tests live under `tests/data/raw/`.
