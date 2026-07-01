QA Schema
=========

The ``qa`` command generates a local quality-assurance notebook from a curated
Parquet, CSV, or HATS catalog. Plot sections are optional, so catalogs without
redshift errors or quality flags do not need placeholder columns or settings.

Minimal Example
---------------

.. code-block:: yaml

   title: Example Catalog
   input_file: tests/data/curated/example.parquet
   input_format: parquet
   output_notebook: reports/qa/example.ipynb
   include_absolute_input_path: false

   plots:
     spatial:
       ra_column: ra
       dec_column: dec
     redshift:
       column: redshift

Run or validate the configuration with:

.. code-block:: console

   redshift-curator qa configs/qa/2mrs.example.yaml --dry-run
   redshift-curator qa configs/qa/2mrs.example.yaml

Input And Output
----------------

``input_file``
  Required local path to the curated catalog.

``input_format``
  ``parquet`` by default. Valid values are ``parquet``, ``csv``, and ``hats``.
  HATS inputs are opened with ``lsdb.open_catalog()``.

``output_notebook`` or ``output_dir``
  Optional notebook destination. Configure at most one. ``output_notebook``
  names the file directly; ``output_dir`` writes ``qa_notebook.ipynb`` inside
  that directory. The default is ``reports/qa/qa_notebook.ipynb``.

``include_absolute_input_path``
  ``true`` by default. When true, generated read cells contain absolute paths.
  When false, the catalog and footprint paths are relative to the generated
  notebook directory. Relative paths avoid exposing machine-specific paths
  while keeping the notebook directly executable in its generated location.
  This policy applies to both in-memory and lazy cells.

``large_input_threshold_mb``
  ``100`` by default. Inputs at or below this on-disk size are loaded into a
  pandas dataframe. Larger inputs use lazy partition aggregation. Directory
  inputs such as partitioned Parquet and HATS are measured recursively.

``force_compute``
  ``false`` by default. When true, bypasses the size threshold and loads the
  complete input into memory. The generated notebook identifies the mode as
  ``forced in-memory`` and emits a runtime warning before loading data.

``dask_cluster``
  Optional executor configuration using the same ``local`` or ``slurm`` schema
  as inspect, prepare, and curate. It is used only in lazy mode. If omitted,
  lazy QA creates the standard local cluster with one worker, one thread, and a
  6 GB memory limit. In-memory and forced in-memory notebooks do not start a
  cluster.

``generate_html``
  ``false`` by default. When true, the pipeline executes an in-memory copy of
  the notebook and exports HTML. The versioned ``.ipynb`` remains unexecuted.
  ``output_html`` optionally selects the HTML path;
  ``html_execution_timeout`` and ``html_kernel_name`` control execution.

Header And Metadata
-------------------

``title`` and ``subtitle``
  Notebook heading. The defaults are ``QA Notebook`` and
  ``Basic dataset characterization``.

``last_verified_run``
  Optional displayed date or label. When omitted, the generation date is used.

``header_images``
  Optional list of at most two images. Each item can be a source string or a
  mapping with ``url``, ``path``, or ``src`` plus optional ``width``, ``align``,
  and ``style``. Images are entirely configuration-driven; the pipeline has no
  survey- or organization-specific image constants.

``summary`` and ``acknowledgments``
  Optional lists of Markdown strings rendered before the data overview.

``metadata``
  Optional complete override for the generated Jupyter notebook metadata. Omit
  it to use the maintained Python 3 kernelspec defaults.

Optional Plots
--------------

``plots.spatial``
  Mollweide density map. ``ra_column`` and ``dec_column`` default to ``ra`` and
  ``dec``. ``title`` is optional. Zero or more curves can be supplied through
  ``footprints``.

``plots.redshift`` and ``plots.redshift_error``
  Histogram settings. ``column`` selects the source column, ``range`` provides
  optional inclusive plotting limits, ``bins`` defaults to ``50``, and
  ``title`` optionally overrides the generated plot title.

``plots.quality``
  Count plot for a configured ``column``. ``description`` is Markdown and can
  contain a table explaining flag values and their source. ``title``
  optionally overrides the generated plot title.

Large Input Behavior
--------------------

The generated notebook records the selected data-access mode, measured input
size, and configured threshold. Large Parquet and CSV inputs are opened with
Dask; large HATS inputs remain on the public LSDB ``Catalog`` API. The pipeline
does not use private LSDB dataframe attributes.

Before opening lazy data, the notebook creates the configured cluster and a
``distributed.Client`` through the shared executor implementation. A final
cleanup cell closes both objects, with an ``atexit`` fallback if execution
stops before that cell. For SLURM, ``dask_cluster.logs_dir`` controls worker
logs and defaults to a ``logs`` directory beside the notebook output.

.. code-block:: yaml

   dask_cluster:
     name: slurm
     args:
       instance:
         cores: 4
         processes: 1
         memory: 24GB
         queue: your-queue
         account: your-account
       scale:
         minimum_jobs: 1
         maximum_jobs: 4

Each plot opens only its configured columns and performs an exact aggregation
per partition:

- spatial distribution computes a two-dimensional histogram of RA and Dec;
- redshift and redshift-error distributions compute one-dimensional
  histograms;
- quality flags compute partitioned value counts.

Only histogram bins or category counts are materialized in memory. The lazy
plot object and aggregate arrays are deleted at the end of each plot cell so
the next cell starts without retaining those data. Basic row counts and
statistics are also aggregated lazily. HATS statistics are rendered as a full
HTML table inside a scrollable container in both lazy and in-memory modes.

In-memory histograms retain the seaborn KDE overlay used by the original QA
template. Lazy histograms render exact aggregated bin counts without KDE,
because computing a KDE would require collecting raw values or introducing an
approximation.

Footprint Curves
----------------

``plots.spatial.footprints`` accepts any number of CSV curve mappings. Common
display options are ``label``, ``color``, and ``linewidth``. Two schemas are
supported:

``region_vertices``
  Polygon or ring curves with columns ``region_id``, ``ring_type``,
  ``vertex_id``, ``ra_deg``, and ``dec_deg``. Exterior vertices are ordered by
  ``vertex_id`` and curves are split safely at the RA wrap.

``declination_limit``
  A sampled boundary with columns ``ra_center`` and ``dec_limit``.

The format is detected from the columns. An optional ``format`` field may make
the expectation explicit; it must match the detected schema. Missing files or
unsupported columns fail configuration validation with the expected schemas in
the error message.

2MRS Example
------------

``configs/qa/2mrs.example.yaml`` reads the curated union of both 2MRS FITS
files. It demonstrates two header images, two footprint curves, relative input
paths, automatic run date, spatial and redshift distributions, and the optional
redshift-error distribution. It intentionally omits a quality plot because the
real use case does not define a redshift-quality flag for that section.
