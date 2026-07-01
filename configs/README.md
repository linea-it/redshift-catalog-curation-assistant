# Configuration examples

This directory contains executable examples, grouped by pipeline stage.

- `manual.yaml` files document the complete configuration surface.
- `<survey>.example.yaml` files run against small source fixtures in
  `tests/data/raw/`.
- Technical variants use `<survey>_<variant>.example.yaml`.
- Fixture files use `<survey>_sample.<ext>`; multipart fixtures use
  `<survey>_<part>_sample.<ext>`.
- Generated data belongs under `reports/` and must not be committed.
- Deployment-specific configurations belong under `production_configs/`.

Tests create prepared and curated intermediates in temporary directories. Do
not add generated pipeline outputs to `tests/data/`.
