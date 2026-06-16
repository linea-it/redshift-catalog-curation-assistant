Prepare Schema
==============

The ``prepare`` command normalizes local catalog inputs into Parquet datasets.
It is the recommended first step for large or repeated workflows.

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
  Multiple inputs must have the same format and schema.

``output_dir``
  Directory where the prepared Parquet dataset and manifest are written.

Input Options
-------------

``fits_hdu``
  FITS HDU to convert. Defaults to ``1``.

``column_names``
  Required for headerless whitespace inputs such as ``.dat`` and ``.idz``.

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

Prepare several files as one logical catalog:

.. code-block:: yaml

   input_files:
     - path/to/catalog_part0.csv
     - path/to/catalog_part1.csv
   output_dir: outputs/prepared/catalog.parquet
   overwrite: true
   output_mode: auto
