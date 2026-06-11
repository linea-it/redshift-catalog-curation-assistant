# Cost Control

Cost controls for later AWS phases should be designed before processing large
catalogs.

Initial principles:

- keep raw sample files small and versioned only when redistribution is allowed;
- prefer local runs for development and tests;
- use column selection when supported by file formats;
- store curated outputs in compressed columnar formats;
- make dry-run modes available before uploads or cloud execution;
- keep generated reports and manifests small enough for code review.
