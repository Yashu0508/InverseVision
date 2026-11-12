# Phase 8 Configuration Compatibility Audit

## Method

The audit assigns one purpose-specific compatibility outcome to every canonical
observation. It uses only canonical configuration fields and the originating
row retained in `source_attributes_json`. An absent setting is `unknown`, never
a match. The audit does not add metadata, compare reconstruction quality, or
aggregate outcomes.

## Status definitions

- `compatible`: every setting required by the stated comparison purpose is
  explicitly present and matches the group's deterministic reference record.
- `incompatible`: every required setting is present, and at least one differs.
- `unknown`: one or more required settings are absent, or the record is
  metadata rather than a numerical performance-comparison candidate.

## Implemented comparison purposes

| Purpose | Required matching settings | Intentional variation |
|---|---|---|
| `controlled_noise_seed_robustness` | resolution, sigma, kernel, boundary, noise level | random seed |
| `controlled_blur_sensitivity` | resolution, kernel, boundary, noise level, seed | sigma |
| `controlled_resolution_sensitivity` | sigma, kernel, boundary, noise level, seed | resolution |
| `within_solver_convergence_trajectory` | case configuration and run type | iteration |
| `equal_iteration_budget_solver_comparison` | case configuration, run type, requested budget | solver |
| `within_solver_tolerance_sensitivity` | case configuration and run type | requested tolerance |
| `descriptive_aggregate_within_sensitivity_axis` | none | aggregate rows are kept descriptive, never individual cases |
| `parameter_selection_within_source` | explicit image/blur/noise settings when present | candidate or selected parameter by source role |
| `spectral_blur_sensitivity` | kernel size | sigma |
| `source_specific_descriptive_only` | core image/blur/noise settings | none; insufficient evidence yields `unknown` |
| `metadata_diagnostic_only` | none | not a performance-comparison candidate; always `unknown` |

## Evidence and safeguards

The audit retains the source phase, file, source row, canonical ID, analytical
category, assigned record type, full explicit configuration signature, and
configuration-source provenance. It uses the P8-07 taxonomy to keep parameter
candidates distinct from selected results, raw Phase 6 cases distinct from
aggregates, and Phase 7 trajectories distinct from iteration-budget and
tolerance runs.

Phase 4/5 source rows without explicit configuration fields are not marked
compatible merely because project reports describe related protocols. Phase 7
case metadata remains a metadata diagnostic rather than solver performance.
The audit does not make Phase 2--7 methods globally comparable. Later work
must choose one stated purpose and use only records eligible for that purpose.

## Generated audit summary

The isolated audit generated 8,741 outcomes, one for every canonical ID:

| Status | Records |
|---|---:|
| compatible | 5,021 |
| incompatible | 288 |
| unknown | 3,432 |

| Comparison purpose | Records |
|---|---:|
| controlled noise-seed robustness | 540 |
| controlled blur sensitivity | 324 |
| controlled resolution sensitivity | 324 |
| descriptive aggregate within sensitivity axis | 3,045 |
| equal iteration-budget solver comparison | 384 |
| metadata diagnostic only | 8 |
| parameter selection within source | 2,517 |
| source-specific descriptive only | 907 |
| spectral blur sensitivity | 20 |
| within-solver convergence trajectory | 384 |
| within-solver tolerance sensitivity | 288 |

## Limits

Compatibility describes configuration evidence for a stated use; it does not
establish statistical independence, equal numerical method formulation, or
scientific superiority. Different noise seeds can be compatible for a controlled
noise-robustness purpose but are not evidence of the same observation. Different
iteration budgets can be compatible in a trajectory analysis but are not
eligible for an equal-budget solver comparison. Missing source image identity,
operator details, and observation identity remain unknown until separately
documented; this audit does not infer them.
