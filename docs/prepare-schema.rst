Prepare Schema
==============

The ``prepare`` command normalizes local catalog inputs into Parquet datasets or
HATS collections. It is the recommended first step for large or repeated
workflows when the input is not already in a supported partitioned format.

Minimal Example
---------------

.. code-block:: yaml

   input_file: tests/data/raw/desi_deep_pilot_sample.fits
   output_dir: outputs/prepared/desi_deep_pilot.parquet
   fits_hdu: 1
   overwrite: true

Required Fields
---------------

``input_file`` or ``input_files``
  Source catalog path, or a list of files that represent one logical catalog.
  Multiple inputs must have the same format. Their schemas must match unless
  ``schema_policy: union`` is selected.

  If the input is already HATS, ``prepare`` stops with a message explaining that
  HATS is supported directly by downstream commands and does not need
  preparation.

``output_dir``
  Directory where the prepared Parquet dataset or HATS collection and manifest
  are written.

Input Options
-------------

``fits_hdu``
  FITS HDU to convert. Defaults to ``1``.

``column_names``
  Required for headerless whitespace inputs such as ``.dat`` and ``.idz``.

``schema_policy``
  Controls multi-file schema handling. ``strict`` is the default and rejects
  mismatches. ``union`` keeps columns in first-seen order and fills columns
  missing from an input with null values.

``large_file_threshold_mb``
  Defaults to ``100``. Small single-file inputs below the threshold are read in
  memory and written as one Parquet part. Large inputs use Dask/chunked paths.

``chunk_size_rows``
  Row chunk size for FITS input. Defaults to ``200000``. The effective chunk
  size is capped by ``target_partition_size_mb``.

``target_partition_size_mb``
  Target Parquet partition size for Dask processing. Defaults to ``100``.

Output Options
--------------

``output_format``
  ``parquet`` by default. Valid values are ``parquet`` and ``hats``.

``output_mode``
  ``auto`` by default. Valid values are:

  - ``auto``: single output for small single-file inputs, partitioned output for
    large or multi-file inputs.
  - ``single``: force one output part.
  - ``partitioned``: force partitioned output.

``allow_large_single_output``
  ``false`` by default. Required when ``output_mode: single`` would concentrate
  a large or multi-file catalog into one partition.

``overwrite``
  ``false`` by default. Set ``true`` to replace an existing output directory.

``part_prefix``
  Prefix for generated Parquet part filenames. Defaults to the input stem.

``dask_cluster``
  Optional Dask executor config. If omitted and a Dask path is used, the default
  local cluster has one worker, one thread per worker, and 6 GB memory.

HATS Output
-----------

When ``output_format: hats`` is set, ``prepare`` writes a HATS collection.
The ``hats`` block must identify the coordinate columns:

.. code-block:: yaml

   output_format: hats

   hats:
     catalog_name: my_catalog
     ra_column: ra
     dec_column: dec
     hats_output_with_margin: true
     margin_threshold: 5.0

``hats.catalog_name``
  Name of the catalog inside the collection. Defaults to the output directory
  name.

``hats.ra_column`` and ``hats.dec_column``
  Required coordinate columns used to build the HATS spatial index. They must
  already be numeric degree columns in the standard ranges required by HATS:
  RA in ``[0, 360)`` and Dec in ``(-90, 90)``.

``hats.margin_threshold``
  Margin cache threshold in arcseconds. Defaults to ``5.0``. When margin
  output is enabled, this must be greater than zero.

``hats.hats_output_with_margin``
  ``true`` by default. When true, HATS output includes a default margin cache
  using ``hats.margin_threshold``. When false, ``hats.margin_threshold`` is
  ignored and no margin cache is generated.

``hats.sort_columns``
  Optional column name passed to ``hats_import`` for large-input sorting.

Small inputs below ``large_file_threshold_mb`` are read into pandas, converted
with ``lsdb.from_dataframe()``, and written with ``Catalog.write_catalog()``.

Large inputs are first normalized to a temporary Parquet dataset using the same
``prepare`` path as Parquet output. Then ``hats_import`` imports that temporary
Parquet dataset with ``file_reader: parquet``, writes the catalog plus default
margin cache when requested, and the temporary Parquet files are removed.

``prepare`` does not transform coordinates before writing HATS. If the source
catalog stores coordinates as HMS/DMS, hourangle strings, radians, or any other
non-standard representation, prepare it as Parquet first, run ``curate`` with
the coordinate transformations and validations, then write HATS from the
curated standard-coordinate catalog.

Large Input Policy
------------------

Large compressed files are rejected before processing because they cannot be
partitioned efficiently. Decompress them first, then run ``prepare`` on the
uncompressed input.

Supported large paths:

- CSV/TXT with Dask ``read_csv``;
- headerless DAT/IDZ with explicit ``column_names``;
- Parquet with Dask ``read_parquet`` and row-group splitting;
- FITS with ``fitsio`` row chunks wrapped as Dask delayed partitions.

For ``output_format: hats``, the same large paths are used to create temporary
Parquet before importing the final HATS collection.

Manifest
--------

``prepare`` writes ``_redshift_curator_manifest.json`` with source paths,
source format, partition format, output mode, partition count, and relevant
size settings.

Examples
--------

Prepare a versioned FITS sample:

.. code-block:: console

   redshift-curator prepare configs/prepare/desi_deep_pilot.example.yaml

Prepare a small CSV sample as HATS:

.. code-block:: console

   redshift-curator prepare configs/prepare/synthetic_hats.example.yaml

Prepare HATS directly from CLI options:

.. code-block:: console

   redshift-curator prepare \
     --path tests/data/raw/synthetic_redshift_catalog.csv \
     --output-dir outputs/prepared/synthetic_hats \
     --output-format hats \
     --hats-ra-column ra \
     --hats-dec-column dec \
     --hats-catalog-name synthetic_hats

Validate a prepare config or CLI invocation without writing output:

.. code-block:: console

   redshift-curator prepare configs/prepare/synthetic.example.yaml --dry-run
   redshift-curator prepare \
     --path tests/data/raw/synthetic_redshift_catalog.csv \
     --output-dir outputs/prepared/synthetic.parquet \
     --dry-run

Prepare several files as one logical catalog:

.. code-block:: yaml

   input_files:
     - path/to/catalog_part0.csv
     - path/to/catalog_part1.csv
   output_dir: outputs/prepared/catalog.parquet
   schema_policy: union
   overwrite: true
   output_mode: auto
