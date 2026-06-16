# Design

Redshift Catalog Curation Assistant is a local-first Python CLI for technical
curation of heterogeneous redshift catalogs.

The project starts with file-based workflows driven by YAML configuration. The
CLI reads raw catalog files, produces auditable reports, and keeps human
curation decisions explicit in configuration files.

The core workflow is planned around these commands:

- `redshift-curator inspect config.yaml`
- `redshift-curator inspect-fits catalog.fits`
- `redshift-curator curate config.yaml`
- `redshift-curator qa config.yaml`
- `redshift-curator validate-flags config.yaml`
- `redshift-curator run config.yaml`

Phase 0 establishes packaging, CLI structure, tests, documentation, and small
example inputs. Phase 1 implements useful local inspection for CSV, Parquet, and
FITS catalogs. Phase 2 starts with local curation to explicit Parquet outputs:
small raw inputs can be curated in memory, while large inputs are expected to be
prepared as Parquet first.
