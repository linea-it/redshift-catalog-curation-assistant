# Project notebooks

Only add notebooks here when they document a maintained Redshift Catalog Curation
Assistant workflow.

Notebook requirements:

- Use small, versioned inputs from `tests/data/raw/`.
- Avoid network access and large local-only datasets.
- Keep execution time suitable for CI and documentation builds.
- Add the notebook to `../notebooks.rst` only when it is intended to be part of
  published documentation.

For notebooks that require large data or expensive computation, store a
pre-executed artifact under `./pre_executed/` instead of executing it during the
documentation build.
