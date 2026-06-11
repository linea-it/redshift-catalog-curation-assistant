# AWS Architecture

Cloud execution is planned for later phases.

The target architecture is:

- raw and curated catalogs stored in S3;
- containerized execution for repeatable runs;
- structured logs for auditability;
- catalog metadata registered for discovery;
- Parquet outputs available for analytical queries.

The local CLI remains the reference implementation before cloud deployment.
