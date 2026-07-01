Curate Transformation Examples
==============================

These examples show complete ``transformations`` blocks for common local
curation tasks. They are declarative: every scientific choice remains explicit
in YAML, and ``curate`` validates the configured final RA, Dec, and redshift
columns after transformations run.

Add A Constant Survey Column
----------------------------

Use this when every output row should carry the survey/source name.

.. code-block:: yaml

   column_selection:
     - object_id
     - ra
     - dec
     - z

   transformations:
     - type: add_constant_column
       name: survey_name
       value: EXAMPLE_SURVEY

Cast A Column
-------------

Use ``cast`` for simple dtype normalization. This is useful for quality columns
that arrive as strings or integers but should be compared numerically later.

.. code-block:: yaml

   column_selection:
     - TARGETID
     - TARGET_RA
     - TARGET_DEC
     - Z
     - VI_quality

   transformations:
     - type: cast
     column: VI_quality
     dtype: float64

Normalize Signed Components And Sentinels
-----------------------------------------

Use ``signed_value_sign`` and ``absolute_value`` when a signed numeric component
must become separate sign and magnitude columns. Use ``replace_values`` for
explicit missing-value sentinels or other small value mappings.

.. code-block:: yaml

   transformations:
     - type: signed_value_sign
       input_column: signed_degrees
       output_column: degree_sign
       negative_label: "-"
       non_negative_label: "+"
     - type: absolute_value
       input_column: signed_degrees
       output_column: degrees
       dtype: int64
     - type: replace_values
       column: magnitude
       replacements:
         - from: -1.0
           to: null

Velocity To Redshift
--------------------

The 2MRS sample config uses this pattern to convert velocity ``V`` and velocity
error ``EV`` in km/s to redshift and redshift error.

.. code-block:: yaml

   input_file: reports/prepared/2mrs.parquet
   output_dir: reports/curated/2mrs.parquet

   coordinates:
     ra_column: RA
     dec_column: DEC

   redshift:
     column: redshift

   transformations:
     - type: velocity_to_redshift
       velocity_column: V
       output_column: redshift
       velocity_error_column: EV
       error_output_column: redshift_err

HMS And DMS To Degrees
----------------------

The 2dFGRS sample config uses separate hour/minute/second and
degree/arcminute/arcsecond columns to generate final coordinates in degrees.
The sign of Dec is taken from the degree component.

.. code-block:: yaml

   input_file: tests/data/raw/2dfgrs_sample.idz.gz
   output_dir: reports/curated/2dfgrs.parquet

   column_selection:
     - serial
     - name
     - ra_j2000_deg
     - dec_j2000_deg
     - z
     - quality

   coordinates:
     ra_column: ra_j2000_deg
     dec_column: dec_j2000_deg

   redshift:
     column: z
     invalid_policy: flag

   transformations:
     - type: ra_hms_to_degrees
       output_column: ra_j2000_deg
       hours: ra_j2000_h
       minutes: ra_j2000_m
       seconds: ra_j2000_s
     - type: dec_dms_to_degrees
       output_column: dec_j2000_deg
       degrees: dec_j2000_d
       arcminutes: dec_j2000_m
       arcseconds: dec_j2000_s

SkyCoord To Degrees
-------------------

Use ``skycoord_to_degrees`` when the input coordinate encoding is best handled
by Astropy. The 6dFGS sample uses RA as hourangle and Dec as degrees.

.. code-block:: yaml

   input_file: tests/data/raw/6dfgs_sample.csv.gz
   output_dir: reports/curated/6dfgs.parquet

   column_selection:
     - SPECID
     - TARGETID
     - RA_deg
     - DEC_deg
     - Z
     - QUALITY

   coordinates:
     ra_column: RA_deg
     dec_column: DEC_deg

   redshift:
     column: Z
     invalid_policy: flag

   transformations:
     - type: skycoord_to_degrees
       ra_column: OBSRA
       dec_column: OBSDEC
       output_ra_column: RA_deg
       output_dec_column: DEC_deg
       ra_unit: hourangle
       dec_unit: deg

Coalesce Redshift Candidates
----------------------------

Use ``coalesce_redshift`` when a catalog has prioritized redshift candidates.
The first value in the configured valid redshift range is selected. By default
that range is ``-0.1 < z < 20`` because ``redshift.allow_blueshifts`` defaults
to ``true``. If ``redshift.allow_blueshifts`` is ``false``, the range is
``0 < z < 20``. If no candidate is valid, ``invalid_value`` is written.

.. code-block:: yaml

   column_selection:
     - object_id
     - ra
     - dec
     - z_final

   coordinates:
     ra_column: ra
     dec_column: dec

   redshift:
     column: z_final
     allow_blueshifts: true
     invalid_policy: flag
     invalid_value: -1

   transformations:
     - type: coalesce_redshift
       output_column: z_final
       columns:
         - z_spec
         - z_phot
       labels:
         - s
         - p
       label_column: z_source
       invalid_label: none
       invalid_value: -1

``label_column`` is optional. When present, ``curate`` writes the label for the
candidate column that supplied the selected redshift. ``labels`` must have the
same length as ``columns``; if omitted, the candidate column names are used.
Rows with no valid redshift get ``invalid_label``, which defaults to
``invalid``.

Filter By Redshift
------------------

Use ``redshift.filters`` to keep only rows matching simple comparisons on the
final redshift column. Filters are applied after transformations and standard
redshift validation. Multiple filters are combined with logical AND.

.. code-block:: yaml

   redshift:
     column: z_final
     filters:
       - op: ">"
         value: 1.6
       - op: "<"
         value: 9

Generated Column Placement
--------------------------

Generated columns are appended by default. Use ``generated_columns_position`` to
place them before selected source columns.

.. code-block:: yaml

   generated_columns_position: first

   column_selection:
     - object_id
     - ra
     - dec
     - z

   transformations:
     - type: add_constant_column
       name: survey_name
       value: SYNTHETIC_REDSHIFT
