
Redshift Catalog Curation Assistant
===================================

Redshift Catalog Curation Assistant is a local-first Python CLI for technical
curation of heterogeneous redshift catalogs.

The current workflow focuses on reproducible file-based inspection and safe
preparation and curation of catalogs as Parquet datasets or HATS collections:

.. code-block:: console

   redshift-curator inspect configs/inspect/synthetic.example.yaml
   redshift-curator inspect-fits tests/data/raw/desi_deep_pilot_sample.fits
   redshift-curator prepare configs/prepare/desi_deep_pilot.example.yaml
   redshift-curator curate configs/curate/synthetic.example.yaml

For local development, create or activate a Python 3.11+ environment and install
the package in editable mode:

.. code-block:: console

   python -m pip install -e '.[dev]'
   pre-commit install

This project was created following the `LINCC Frameworks Python Project
Template <https://lincc-ppt.readthedocs.io/en/latest/>`_.


.. toctree::
   :hidden:

   Home page <self>
   Design <design>
   Inspect schema <inspect-schema>
   Prepare schema <prepare-schema>
   Curate schema <curate-schema>
   Curate transformations <curate-transformations>
   Large data and HPC <large-data>
   Scientific scope <scientific-scope>
   Sample data <sample-data>
   AWS architecture <aws-architecture>
   Cost control <cost-control>
   API Reference <autoapi/index>
   Notebooks <notebooks>
