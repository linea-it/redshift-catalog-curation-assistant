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
  HATS inputs are opened with ``lsdb.open_catalog()`` and materialized for the
  notebook with ``compute()``.

``output_notebook`` or ``output_dir``
  Optional notebook destination. Configure at most one. ``output_notebook``
  names the file directly; ``output_dir`` writes ``qa_notebook.ipynb`` inside
  that directory. The default is ``reports/qa/qa_notebook.ipynb``.

``include_absolute_input_path``
  ``true`` by default. When true, generated read cells contain absolute paths.
  When false, the catalog and footprint paths are relative to the generated
  notebook directory. Relative paths avoid exposing machine-specific paths
  while keeping the notebook directly executable in its generated location.

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

Optional Plots
--------------

``plots.spatial``
  Mollweide density map. ``ra_column`` and ``dec_column`` default to ``ra`` and
  ``dec``. ``title`` is optional. Zero or more curves can be supplied through
  ``footprints``.

``plots.redshift`` and ``plots.redshift_error``
  Histogram settings. ``column`` selects the source column, ``range`` provides
  optional inclusive plotting limits, and ``bins`` defaults to ``50``.

``plots.quality``
  Count plot for a configured ``column``. ``description`` is Markdown and can
  contain a table explaining flag values and their source.

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
