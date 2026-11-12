# Phase 8 Analytical Record-Type Taxonomy

## Purpose

The P8-07 audit assigns every canonical observation a detailed analytical role
using its source file and, where required, explicit source-row fields. It does
not compare configurations, aggregate observations, or rank methods.

## Taxonomy

| Analytical category | Assigned record types | Rule |
|---|---|---|
| Spectral conditioning | `singular_value_observation`, `picard_noise_diagnostic`, `spectral_or_conditioning_summary`, `structured_svd_validation_diagnostic` | Phase 3 source-specific adapters retain singular spectra, Picard/noise diagnostics, blur summaries, and structured-SVD validation separately. |
| Reconstruction baseline | `unregularized_baseline_reconstruction_result` | Phase 2 rows are direct-inverse, QR, or normal-equation reconstruction results. |
| Explicit regularization | `regularization_comparison_result`, `explicit_regularization_reconstruction_result` | Phase 4 comparison rows and TSVD/Tikhonov parameter rows remain reconstruction results. |
| Parameter selection | `parameter_candidate_evaluation`, `selected_parameter_result` | Phase 5 candidate curves are never selected-parameter records; summary rows are selected-parameter results with post-selection metrics. |
| Robustness sensitivity | `noise_seed_individual_case_result`, `blur_condition_individual_case_result`, `resolution_individual_case_result`, `aggregated_sensitivity_summary`, `selected_parameter_by_case` | Phase 6 raw rows use explicit `sensitivity_axis`; aggregate rows remain descriptive summaries and are never independent case results. |
| Iterative solvers | `iteration_history_observation`, `tolerance_stopping_observation`, `iteration_budget_solver_observation` | Phase 7 `run_type` determines trajectory, tolerance stopping, or iteration-budget role. Trajectories are not independent experiments. |
| Iterative validation | `small_system_iterative_validation` | Phase 7 small reference comparisons are separate from scalable solver runs. |
| Experimental metadata | `experimental_metadata_diagnostic` | Phase 7 case metadata and consistency diagnostics are not solver-performance observations. |
| Unresolved | `unclassified`, `unclassified_phase6_case_result`, `unclassified_iterative_solver_observation` | A source-specific rule is unavailable or an explicit Phase 6 axis/Phase 7 run type is unknown. These rows receive `classification_pending`; no guess is made. |

## Audit fields

`phase8_record_type_audit.csv` contains one row per canonical record. It retains
source phase/file/row and canonical ID, then records the prior canonical
record type, assigned detailed type, category, deterministic adapter rule,
status, and an explanatory reason. This audit is classification metadata; it
does not alter the canonical observations or infer missing configurations.

## Boundaries

The taxonomy deliberately keeps candidates separate from selections, raw cases
separate from aggregates, and iteration histories separate from solver-budget
or stopping observations. Classification labels describe record meaning, not
comparability, replication, statistical independence beyond the explicit
summary/trajectory distinction, or method quality.
