# redshift-catalog-curation-assistant

A simple, reproducible and extensible tool to aid in the technical curation of
heterogeneous redshift catalogs.

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
`examples/raw/synthetic_redshift_catalog.csv` and a matching config at
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

For quick inspection without writing a YAML config first, use defaults from the
input path:

```bash
redshift-curator inspect --path tests/data/raw/6dfgs_sample.csv.gz
redshift-curator inspect --path tests/data/raw/desi_deep_pilot_sample.fits --fits-hdu 1
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

Large or redistributability-unclear source catalogs should stay outside git, for
example under `scratch-folder-NOT-TRACKED/`. Small sample files that are safe to
redistribute live under `tests/data/raw/`.

## Scientific Naming

Synthetic data in this repository is clearly named synthetic and is not
associated with any real survey. Real survey names are used only for real source
catalogs or local samples derived from those catalogs.

Files without embedded column names require a `column_names` list in the YAML
config. That list is user-supplied metadata and must match the number of columns
in the file. Files that already include column names ignore `column_names`.

See `docs/sample-data.md` for the current local sample-data convention.
