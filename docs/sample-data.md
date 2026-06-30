# Sample Data

The repository keeps scientific naming conservative.

Synthetic data must be named as synthetic data. The versioned synthetic fixture
at `tests/data/raw/synthetic_redshift_catalog.csv` is not associated with any
real survey.

Real survey names should only be used when the input file is a real catalog or a
sample derived from that real catalog. Small redistributable fixtures live under
`tests/data/raw/`. Larger exploratory source files should stay outside git.

Current source files used to generate the versioned fixtures:

- `2dflens_bestredshifts_goodz_withtypesandmags_final.dat.gz`: 2dFLenS
- `best.observations.idz.gz`: 2dFGRS
- `merged_cat_LSST_WL_Y1.fits`: DESI Deep Pilot
- `2mrs_1175_done.fits` and `2mrs_extra_done.fits`: 2MRS
- `part0.snappy.parquet`: Euclid Q1 sample source
- `spAll-v6_1_3.fits`: SDSS DR19 spAll
- `spec_dr3.csv.gz`: 6dFGS
- `elaisfbmc_collection`: small HATS/LSDB collection fixture

Versioned fixtures:

- `tests/data/raw/synthetic_redshift_catalog.csv`: small synthetic catalog used
  by `configs/inspect/synthetic.example.yaml`.
- `tests/data/raw/2dflens_sample.dat.gz`: first 1000 data rows from the 2dFLenS
  file.
- `tests/data/raw/2dfgrs_sample.idz.gz`: first 1000 original rows from the
  2dFGRS file.
- `tests/data/raw/desi_deep_pilot_sample.fits`: first 1000 rows from the DESI
  Deep Pilot FITS table HDU, preserving all columns.
- `tests/data/raw/6dfgs_sample.csv.gz`: CSV header plus first 1000 rows from the
  6dFGS file.
- `tests/data/raw/2mrs_1175_done_sample.fits` and
  `tests/data/raw/2mrs_extra_done_sample.fits`: first 1000 rows from each 2MRS
  FITS table HDU, preserving their distinct 29- and 32-column schemas.
- `tests/data/raw/euclid_q1_sample.snappy.parquet`: first 1000 rows from the
  Euclid Q1 Parquet source, preserving all columns.
- `tests/data/raw/sdss_dr19_sample.fits`: first 1000 rows from the SDSS DR19
  spAll FITS table HDU, preserving all columns.
- `tests/data/raw/elaisfbmc_collection`: small HATS/LSDB collection used by
  `configs/inspect/elaisfbmc_collection.example.yaml` and
  `configs/curate/elaisfbmc_collection.example.yaml`.

Catalogs without column names can be read only when explicit column names are
provided, either with `column_names` in YAML or with `--column-name`/
`--column-names` in the CLI. The list must have exactly the same number of
entries as the file has columns. This is required for the 2dFGRS and 2dFLenS
fixtures, and those names should be treated as user-supplied scientific
metadata.

For files that already carry names, such as CSV with a header or FITS binary
tables, `column_names` is ignored.
