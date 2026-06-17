Large Data And HPC
==================

The recommended large-data workflow is:

.. code-block:: text

   raw catalog -> prepare -> partitioned Parquet -> inspect/curate

Default Dask Executor
---------------------

When a command uses Dask and the user does not provide ``dask_cluster``, the
default local cluster is:

.. code-block:: yaml

   dask_cluster:
     name: local
     args:
       n_workers: 1
       threads_per_worker: 1
       memory_limit: 6GB
       dashboard_address:

This default favors one larger local worker over several small workers.

Prepare
-------

``prepare`` is the primary entry point for large raw catalogs. It can use Dask
for large CSV/TXT/DAT/IDZ and Parquet inputs, and uses FITS row chunks wrapped
as Dask delayed partitions for large FITS files.

Use ``target_partition_size_mb`` to tune output partition size. For HPC testing,
start conservatively and increase workers only after checking storage behavior.

When ``output_format: hats`` is requested for large non-HATS inputs, ``prepare``
first creates a temporary Parquet dataset, then imports it into a HATS
collection with ``hats_import`` using ``file_reader: parquet``. The temporary
Parquet dataset is removed after the HATS import. Inputs that are already HATS
do not need ``prepare``.

HATS output requires RA/Dec columns that are already numeric degrees in the
standard HATS ranges: RA in ``[0, 360)`` and Dec in ``(-90, 90)``. ``prepare``
validates those columns before invoking LSDB or ``hats_import``. Catalogs that
need coordinate conversion should be prepared as Parquet, curated into standard
coordinates, and only then written as HATS.

Inspect
-------

For Parquet and FITS, ``inspect`` uses selective reads by default. Expensive
statistics can be parallelized explicitly:

.. code-block:: yaml

   parallel_stats: true
   stats_mode: candidates
   parquet_stats_batch_size: 50
   fits_stats_batch_size: 8
   fits_stats_chunk_rows: 200000

For Parquet, parallel statistics are scheduled by fragment and column batch.
For FITS, they are scheduled by row chunk and column batch.

If ``parallel_stats`` is not set, a configured ``dask_cluster`` does not change
the Parquet/FITS inspect path. This keeps simple inspection lightweight.

For HATS catalogs and collections, ``inspect`` always creates a Dask client and
opens the input through ``lsdb.open_catalog()``. The HATS path stays on the
public LSDB ``Catalog`` API for sampling and aggregate statistics.

Curate
------

For large inputs, ``curate`` requires Parquet. It reads only needed input
columns when ``column_selection`` is configured: selected source columns,
transformation dependencies, and source versions of configured RA, Dec, and
redshift columns.

RA, Dec, and redshift validation are aggregated into one Dask computation on
large Parquet inputs. For expensive transformations, set:

.. code-block:: yaml

   persist_after_transformations: true

This persists the transformed Dask dataframe before validation and writing.

When ``output_format: hats`` is requested for large Parquet input, ``curate``
writes the curated Dask dataframe to temporary Parquet and imports it with
``hats_import``. The temporary Parquet dataset is removed after the HATS import.
When ``hats_output_with_margin`` is enabled, the HATS import also writes a
default margin cache using the configured or default margin threshold.

For HATS input, ``curate`` always creates a Dask client, opens the catalog with
``lsdb.open_catalog()``, applies transformations through public LSDB
``Catalog`` operations, and writes HATS with ``Catalog.write_catalog()``.
HATS input currently requires HATS output.

SLURM Example
-------------

.. code-block:: yaml

   dask_cluster:
     name: slurm
     args:
       instance:
         cores: 4
         processes: 1
         memory: 16GB
         queue: cpu
         account: my-account
       scale:
         minimum_jobs: 1
         maximum_jobs: 4

Operational Notes
-----------------

- Prefer ``stats_mode: candidates`` for wide catalogs.
- Avoid ``output_mode: single`` for large catalogs unless intentional.
- Start with moderate partition sizes, then benchmark.
- FITS parallel reads can stress shared storage; scale workers gradually.
