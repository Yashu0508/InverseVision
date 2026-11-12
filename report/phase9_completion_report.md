# Phase 9 Completion Report

## Verdict

**PASS — Phase 9 complete.**

## Deliverables

- `report/final_research_report.md`: consolidated research narrative with
  actual methods, evidence, negative results, limitations, and selected
  figures.
- `report/methodology_reference.md`: implementation-aligned equations,
  vectorization, forward/adjoint conventions, selection rules, metrics, and
  storage definitions.
- `report/results_and_figures_index.md`: verified index of source metrics,
  Phase 8 controls, comparative tables, figures, and diagnostic reports.
- `README.md`: updated project status, final documentation links, and
  saved-evidence reproduction command.
- `docs/development.md`: current phase status and frozen-artifact guidance.

## Evidence and mathematical checks

The consolidation inspected forward-model, regularization, selection,
iterative-solver, metric, configuration, experiment, Phase 2--8 report, and
Phase 8 validation sources. The final documents preserve C-order vectorization,
the `B_y kron B_x` ordering, zero boundary without truncated-kernel
renormalization, the exact adjoint action, Tikhonov lambda convention, and the
distinction between theoretical storage and measured runtime.

Numerical claims in the final report were cross-checked against the saved
Phase 8 tables and final validation: catastrophic direct/QR errors, Tikhonov
lambda 0.1, the TSVD-GCV rank-142 failure, Phase 6 normal-equation instability,
and strict-compatible Phase 7 equal-budget evidence.

## Integrity and reproducibility

Phase 2--8 result and report hashes were captured before Phase 9 documentation
work and compared afterwards. Frozen artifacts were unchanged. Phase 9 did not
run Phase 2--7 experiments or regenerate Phase 8 tables, figures, canonical
data, compatibility audit, classification audit, or metadata reconciliation.

The checked documentation commands are `python -m pytest`,
`python experiments/phase8_integrated_analysis.py --list-stages`, and
`python experiments/phase8_integrated_analysis.py --stage comparative-analysis`.
The latter is explicitly a saved-evidence Phase 8 stage, not an earlier
experiment runner.

## Remaining limitations

The frozen experiment set remains limited in images, noise realizations, blur
and resolution grids, and maximum 128x128 scalable cases. Runtime is
machine-specific, storage estimates are not peak-RAM observations, and many
historical Phase 2--5 observations are descriptive because their compatibility
status is unknown. These limitations are documented rather than resolved by
Phase 9.
