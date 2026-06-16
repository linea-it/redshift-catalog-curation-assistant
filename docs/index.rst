
Redshift Catalog Curation Assistant
===================================

Redshift Catalog Curation Assistant is a local-first Python CLI for technical
curation of heterogeneous redshift catalogs.

The current workflow focuses on reproducible file-based inspection and safe
preparation of raw catalogs as Parquet datasets:

.. code-block:: console

   redshift-curator inspect configs/inspect/synthetic.example.yaml
   redshift-curator inspect-fits tests/data/raw/desi_deep_pilot_sample.fits
   redshift-curator prepare configs/prepare/desi_deep_pilot.example.yaml

For local development, create or activate a Python 3.11+ environment and install
the package in editable mode:

.. code-block:: console

   python -m pip install -e '.[dev]'
   pre-commit install


.. toctree::
   :hidden:

   Home page <self>
   Design <design>
   Curate schema <curate-schema>
   Curate transformations <curate-transformations>
   Scientific scope <scientific-scope>
   Sample data <sample-data>
   AWS architecture <aws-architecture>
   Cost control <cost-control>
   API Reference <autoapi/index>
   Notebooks <notebooks>
