Inspect Schema
==============

The ``inspect`` command summarizes a catalog without making scientific choices.
It writes a machine-readable JSON report and a human-readable Markdown report.

Minimal Example
---------------

.. code-block:: yaml

   input_file: tests/data/raw/synthetic_redshift_catalog.csv
   survey_name: SYNTHETIC_REDSHIFT

Required Fields
---------------

``input_file``
  Catalog file, partitioned Parquet directory, or HATS catalog/collection
  directory to inspect.

Common Options
--------------

``survey_name``
  Name used in the report and default output directory. Defaults to the input
  stem.

``output_dir``
  Optional directory where ``inspect_report.json`` and ``inspect_report.md`` are
  written. If omitted, a default output directory is derived from
  ``survey_name``.

``fits_hdu``
  FITS HDU to inspect. Defaults to ``1``.

``unique_limit``
  Maximum number of unique categorical values reported per column.

``stats_mode``
  ``candidates`` by default. Valid values are:

  - ``candidates``: collect stats only for candidate science columns.
  - ``all``: collect stats for all eligible selected columns.
  - ``none``: skip statistics.

``sample_max_columns``
  Maximum number of columns included in the sample rows.

``sample_seed``
  Random seed used by HATS/LSDB sampling. Defaults to ``42``.

``column_names``
  Required for headerless whitespace files such as ``.dat`` and ``.idz``.

``column_selection``
  Optional list of columns to include in the report. ``n_rows`` and
  ``n_columns`` still describe the original catalog; ``n_columns_selected``
  describes the selected subset.

``column_patterns``
  Mapping from semantic categories to regex lists. Supported categories include
  ``ra``, ``dec``, ``redshift``, ``quality``, ``redshift_error``, ``id``, and
  ``object_type``.

Large Input Options
-------------------

``dask_threshold_mb``
  Size threshold for using Dask on generic tabular inputs. Defaults to ``100``.
  ``0`` or a negative value disables threshold-based Dask reads for this path.

``allow_large_raw_inspect``
  ``false`` by default. Large raw inputs are normally blocked with a suggestion
  to run ``prepare`` first. Set this to ``true`` only when direct raw inspection
  is intentional.

``parallel_stats``
  ``false`` by default. When ``true``, Parquet statistics are computed with
  Dask delayed tasks per Parquet fragment and column batch, and FITS statistics
  are computed with Dask delayed tasks per row chunk and column batch.

``parquet_stats_batch_size``
  Number of Parquet columns read per statistics batch. Defaults to ``50``.

``fits_stats_batch_size``
  Number of FITS columns read per statistics batch. Defaults to ``8``.

``fits_stats_chunk_rows``
  Number of FITS rows read per row chunk when ``parallel_stats: true``.
  Defaults to ``200000``.

``dask_cluster``
  Optional Dask executor config. If omitted and a Dask path is used, the default
  local cluster has one worker, one thread per worker, and 6 GB memory.

Dry Run
-------

Use ``--dry-run`` to validate the inspect config and input schema without
writing ``inspect_report.json`` or ``inspect_report.md``. The dry run opens
schema/metadata for supported inputs and validates selected columns, but it does
not compute statistics or samples.

HATS Input
----------

When ``input_file`` points to a HATS catalog or HATS collection directory,
``inspect`` opens it with ``lsdb.open_catalog()`` and always creates a Dask
client, regardless of catalog size. The report stays on the public LSDB
``Catalog`` API: columns and dtypes come from catalog attributes, sample rows
come from ``random_sample()``, and numeric min/max/null/count statistics come
from ``aggregate_column_statistics()``.

HATS categorical unique values are estimated from an LSDB random sample rather
than a full catalog scan. This keeps inspection lightweight and avoids dropping
below the LSDB catalog abstraction.

Reports
-------

The JSON report includes:

- survey name;
- input path;
- row count;
- total and selected column counts;
- selected columns;
- dtypes;
- candidate columns;
- numeric statistics;
- categorical unique values;
- sample rows;
- warnings.

The Markdown report contains the same core information in review-friendly
tables and sections.

Examples
--------

Inspect a versioned sample:

.. code-block:: console

   redshift-curator inspect configs/inspect/synthetic.example.yaml

Inspect a HATS collection:

.. code-block:: console

   redshift-curator inspect configs/inspect/elaisfbmc_collection.example.yaml

Inspect a FITS file after checking HDUs:

.. code-block:: console

   redshift-curator inspect-fits tests/data/raw/desi_deep_pilot_sample.fits
   redshift-curator inspect configs/inspect/desi_deep_pilot.example.yaml

Inspect a selected subset:

.. code-block:: yaml

   input_file: tests/data/raw/desi_deep_pilot_sample.fits
   survey_name: DESI_DEEP_PILOT
   fits_hdu: 1
   column_selection:
     - TARGETID
     - TARGET_RA
     - TARGET_DEC
     - Z
     - VI_quality
