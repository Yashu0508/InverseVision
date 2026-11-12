# Phase 8 Canonical Long-Form Schema

## Purpose

`results/metrics/phase8/phase8_canonical_long_form.csv` is the Phase 8
canonical observation layer. It represents one meaningful numeric measurement
per row and retains the raw source row in `source_attributes_json`. It does not
perform configuration compatibility assessment, aggregation, ranking, or
statistical comparison.

## Canonical fields

| Field | Type | Meaning and missing-value policy |
|---|---|---|
| `canonical_record_id` | string | Deterministic `phase:file:source-row:metric` identifier. |
| `source_phase`, `source_file`, `source_row` | string, string, integer | Immutable source provenance. `source_row` is one-based after the CSV header. |
| `record_type`, `analytical_category` | string | Source-specific analytical meaning. They keep spectral, selection, aggregate, metadata, and iterative records distinct. |
| `experiment_id`, `case_id` | string | Populated only from an explicit source `case_id`; otherwise blank. |
| `method`, `solver`, `parameter_selection_method` | string | Original method, Phase 2 solver, and selection criterion where explicitly recorded. |
| `image_shape`, `image_height`, `image_width`, `blur_sigma`, `kernel_size`, `boundary`, `noise_standard_deviation`, `random_seed` | string | Explicit source configuration only. Blank means unavailable; no cross-phase inference is used. |
| `metric_name`, `metric_value`, `metric_value_status`, `metric_unit` | string | One numeric measurement. Values are finite numeric strings, or blank with status `missing` for blank/non-finite source values. Units are source-aware; `unspecified` means the source did not support a justified physical unit. |
| `observation_index`, `iteration` | string | Source index/candidate index and recorded iteration count where available. |
| `parameter_name`, `parameter_value` | string | Explicit regularization, candidate, selected, or damping parameter when represented deterministically by the source row. |
| `source_attributes_json` | JSON object string | Complete original CSV row. It preserves every source field, including fields without a justified canonical destination. |

## Record types and adapter rules

Each source file has a fixed adapter and one of these record types: method
result, regularization result, parameter candidate, selection result,
case-method result, aggregate statistic, spectral component, spectral summary,
blur-condition summary, validation summary/result, iterative solver result,
or case metadata. In particular:

- Phase 3 singular-spectrum and Picard rows remain spectral components.
- Phase 5 candidate rows remain parameter candidates; Phase 5 summary rows
  remain selection results.
- Phase 6 aggregate rows remain aggregate statistics. Their source `metric`
  label is combined with `mean`, `standard_deviation`, `median`, `minimum`, or
  `maximum` in `metric_name`; they are not case-level observations.
- Phase 7 iterative rows remain solver-run observations, and Phase 7
  small-system validation remains a separate validation record type.
- Phase 7 case metadata emits only explicit numeric diagnostics as
  observations. Array-file names and boolean implementation flags remain in
  `source_attributes_json`.

All Phase 2--7 metric CSV sources listed in `phase8_source_inventory.csv` are
processed by a source-specific classification. The mapping audit records every
header field and whether it is mapped to a canonical field or retained solely
as source-specific information.

## Units and numeric policy

Known units include seconds, dB, dimensionless ratios, normalized-intensity
square, data-space L2 norms, normal-equation L2 norms, solution/reference L2
norms, bytes, and operator gain. Where a source metric has no defensible unit,
the unit is `unspecified` or `source_metric_unit_unspecified`; it is never
guessed. Blank and non-finite source numeric entries are represented by an
empty `metric_value` and `metric_value_status=missing`, while the original
spelling remains in `source_attributes_json`.

## Deliberate limits

The schema preserves meaning and provenance. It does not decide whether two
records are comparable, whether repeated Phase 6 configurations are
independent, or which method is preferable. Those are later Phase 8 tasks and
must use explicit compatibility and aggregation rules.
