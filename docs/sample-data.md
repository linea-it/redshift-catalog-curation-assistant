# Sample Data

The repository keeps scientific naming conservative.

Synthetic data must be named as synthetic data. The versioned example catalog at
`examples/raw/synthetic_redshift_catalog.csv` is not associated with any real
survey.

Real survey names should only be used when the input file is a real catalog or a
sample derived from that real catalog. Small redistributable fixtures live under
`tests/data/raw/`. Larger exploratory source files stay under
`scratch-folder-NOT-TRACKED/`, which is intentionally ignored by git.

Current source files used to generate the versioned fixtures:

- `2dflens_bestredshifts_goodz_withtypesandmags_final.dat.gz`: 2dFLenS
- `best.observations.idz.gz`: 2dFGRS
- `merged_cat_LSST_WL_Y1.fits`: DESI Deep Pilot
- `spec_dr3.csv.gz`: 6dFGS

Versioned fixtures:

- `tests/data/raw/2dflens_sample.dat.gz`: first 1000 data rows from the 2dFLenS
  file.
- `tests/data/raw/2dfgrs_sample.idz.gz`: first 1000 original rows from the
  2dFGRS file.
- `tests/data/raw/desi_deep_pilot_sample.fits`: first 1000 rows from the DESI
  Deep Pilot FITS table HDU, preserving all columns.
- `tests/data/raw/6dfgs_sample.csv.gz`: CSV header plus first 1000 rows from the
  6dFGS file.

Catalogs without column names can be read only when the YAML config provides an
explicit `column_names` list. The list must have exactly the same number of
entries as the file has columns. This is required for the 2dFGRS and 2dFLenS
fixtures, and those names should be treated as user-supplied scientific
metadata.

For files that already carry names, such as CSV with a header or FITS binary
tables, `column_names` is ignored.
