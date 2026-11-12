# Phase 8 — Integrated Experimental Analysis

## Scope and evidence

This synthesis reads saved Phase 2–7 CSV metrics. It does not rerun prior experiments. The script `experiments/phase8_integrated_analysis.py` writes the provenance-preserving unified records and figures. Each unified row retains its source phase and original source file. The matched subset is limited to the Phase 2/4 small-system comparison; other phases use different resolutions, seeds, or protocols and are not pooled as if identical.

## Unregularized methods and conditioning

Phase 2 provides direct inversion, QR, and normal-equation measurements in `phase2_baseline_comparison.csv`. These records show that a small residual does not guarantee a useful reconstruction: the direct inverse/QR rows report relative reconstruction error around 1.95×10⁴ despite residual near numerical zero. Phase 3 singular-value and Picard/noise-amplification CSVs explain this instability: inverse weighting magnifies components associated with small singular values. Condition estimates depend on the stated numerical-rank threshold and are not interchangeable across matrix sizes.

## Explicit regularization and selection

Phase 4 compares TSVD and Tikhonov through saved reconstruction errors, residuals, and filter outputs. TSVD truncates small-singular-value components; Tikhonov smoothly attenuates them. Neither method dominates independent of blur and chosen parameter. Phase 5 and Phase 6 selected-parameter tables compare L-curve, GCV, and discrepancy selection. The documented TSVD-GCV rank-142 selection is retained as a valid outcome of its criterion/grid and exhibits catastrophic ground-truth error in that case; ground truth is an evaluation metric after selection, never an input to selection.

## Robustness and iterative methods

Phase 6 raw and aggregated data vary noise seed, blur, and resolution; these are interpreted within their sensitivity groups. Repeated rows reporting one shared failed case are not counted as independent failures. Phase 7 compares LSQR, CGLS, and damped LSQR using common observations within each case. Its 64×64 sigma=1.5 trajectories show semiconvergence: CGLS error improves early and later rises while residual continues to fall. Damping and early stopping trade residual fit against reconstruction error. Phase 7's matrix-free operator stores O(n²) one-dimensional factors but its dense factor products cost O(n³) arithmetic per forward or adjoint application for square images; measured solver timings are machine-specific, not complexity estimates.

## Integrated comparison and limitations

The generated `phase8_small_system_baseline_regularization.csv` retains the compatible Phase 2/4 records. Phase 6 and Phase 7 are separate protocols, so their raw values are not treated as a controlled head-to-head ranking. `phase2_fit_vs_error.png` visualizes the distinction between residual and reconstruction error; `phase7_early_stopping_error.png` visualizes iterative semiconvergence. Source CSVs remain the authority for exact metrics and configuration.

There is no single universal best method in this evidence. Explicit regularization offers a tunable spectral filter; parameter-selection rules can fail on particular spectra; iterative methods scale storage well but still incur dense-factor arithmetic and can semi-converge. The study uses limited image sources, noise realizations, and parameter grids. Not every phase exports identical metrics or compatible conditions, so unavailable controlled comparisons are reported as limitations rather than filled with estimates.
