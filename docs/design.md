# Design

Redshift Catalog Curation Assistant is a local-first Python CLI for technical
curation of heterogeneous redshift catalogs.

The project starts with file-based workflows driven by YAML configuration. The
CLI reads raw catalog files, produces auditable reports, and keeps human
curation decisions explicit in configuration files.

The core local workflow is built around these commands:

- `redshift-curator inspect config.yaml`
- `redshift-curator inspect-fits catalog.fits`
- `redshift-curator prepare config.yaml`
- `redshift-curator curate config.yaml`
- `redshift-curator qa config.yaml`
- `redshift-curator validate-flags config.yaml`
- `redshift-curator run config.yaml`

Phase 0 establishes packaging, CLI structure, tests, documentation, and small
example inputs. Phase 1 implements useful local inspection for CSV, Parquet,
FITS, and HATS catalogs. The local `prepare` workflow normalizes local inputs
to Parquet datasets or HATS collections, while rejecting HATS inputs because
they are already supported directly downstream. Phase 2 implements local
curation to explicit Parquet or HATS outputs: small raw inputs can be curated
in memory, while large raw inputs are expected to be prepared as Parquet first.
Expensive inspect statistics, large Parquet curation paths, and HATS conversion
can use Dask for local or HPC execution.
