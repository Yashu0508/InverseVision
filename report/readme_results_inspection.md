# Results Inspection Audit for README Preparation

## Coverage

Recursive inventory found **285 files** under `results/`: 34 CSV, 2 JSON, 2
logs, 201 NPY arrays, 42 PNG, and 4 `.gitkeep` directory markers. Every file
was opened with a type-appropriate reader. No unreadable file, decode failure,
NaN/Inf array value, malformed CSV header, JSON parse error, or log-read error
was found.

| Type | Count | Inspection method | Status |
|---|---:|---|---|
| CSV | 34 | `csv.DictReader`: header, row count, missing cells, and result fields | All read |
| JSON | 2 | JSON parser and configuration-key inspection | All parsed |
| logs | 2 | UTF-8 text read and completion-status inspection | All read |
| NPY | 201 | NumPy load: shape, dtype, finite count, min/max/mean | All loaded |
| PNG | 42 | Pillow decode, mode and dimensions; representative scientific figures visually reviewed | All decoded |
| markers | 4 | Text read | All read |

## Complete file inventory by exact artifact family

The 34 CSV files comprise every Phase 2--7 metrics CSV and all 17 Phase 8
metrics files under `results/metrics/phase8/`. They include Phase 2 baseline,
five Phase 3 diagnostics, three Phase 4 files, two Phase 5 files, three Phase
6 files, three Phase 7 files, source inventory, canonical, classification,
compatibility, metadata, unified, and seven comparative-table files.

The two JSON files are `phase6_experiment_config.json` and
`phase7_experiment_config.json`; both record the settings and provenance used
by their experiments. The logs are `phase6_experiment.log` and
`phase7_experiment.log`; their final lines report 11 cases/99 records and four
matrix-free cases/88 rows, respectively.

All 201 arrays are Float64 reconstructions or forward-model arrays. Exact
families are: three Phase 2 12x12 baselines; 17 Phase 4 12x12 regularization
arrays; six Phase 5 selected arrays; 99 Phase 6 arrays spanning blur, five
noise seeds, and three resolutions; and 76 Phase 7 arrays covering original,
blurred, noise, observation, trajectory, tolerance, and iteration-budget
arrays for four matrix-free cases. Every loaded value was finite. Raw arrays
were not mistaken for metrics; their range inspection confirms the severe
unregularized excursions and bounded regularized reconstructions documented by
the CSV metrics.

All 42 PNG files decoded successfully. They consist of 12 Phase 8 figures, 28
Phase 1--7 plots, and two earlier Phase 8 preliminary plots. Visual review of
the selected README figures verified: visible blur/noise in Phase 1;
residual-versus-error separation in the Phase 4 tradeoff plot; candidate
diagnostics in Phase 5; CGLS semiconvergence in Phase 7; and explicit
theoretical-not-peak-RAM labeling in Phase 8 storage scaling.

## Extracted findings used in README

- `phase2_baseline_comparison.csv` records tiny direct/QR residuals with
  reconstruction errors near 1.95e4; normal equations are near 2.03e4.
- `phase3_svd_summary.csv` reports an 8x8 condition number about 1.77e6.
- `phase5_selection_summary.csv` records TSVD-GCV rank 142 with catastrophic
  error about 17317.138, while `phase5_selection_curves.csv` records its small
  residual.
- `phase4_regularization_comparison.csv` records Tikhonov lambda 0.1 error
  0.185931..., while the Phase 8 compatibility table correctly labels its
  historical comparison status unknown.
- Phase 6 raw/summary files show controlled sensitivity evidence and unstable
  unregularized outcomes. Phase 7 files distinguish equal-budget, tolerance,
  and trajectory observations.
- Phase 8 canonical, classification, compatibility, metadata, and comparison
  files preserve provenance and prevent descriptive evidence from being
  presented as strict controlled evidence.

## Traceability map

| README claim or figure | Source artifact |
|---|---|
| Baseline instability | `results/metrics/phase8/phase8_table1_baseline_inverse_methods.csv` |
| Conditioning | `results/metrics/phase3_svd_summary.csv` |
| Tikhonov and TSVD-GCV findings | `results/metrics/phase8/phase8_table2_regularization_methods.csv`, `phase8_table3_parameter_selection.csv` |
| Strict iterative evidence | `results/metrics/phase8/phase8_table5_iterative_solver_comparison.csv` |
| Forward degradation image | `results/plots/phase1_degradation.png` |
| Tradeoff, selection, semiconvergence, storage figures | `results/plots/phase4_regularization_comparison_tradeoff.png`, `phase5_parameter_selection.png`, `phase7_cgls_convergence_and_early_stopping.png`, `results/figures/phase8/phase8_figure12_storage_scaling.png` |

## Limitations

No result file was unavailable. PNG semantic interpretation is strongest for
the visually reviewed figures; all remaining figures were decoded and their
scientific roles were reconciled with their saved data and generation labels.
This audit did not alter `results/`.
