# Phase 8 Comparative Analysis (P8-04)

## Research question and method

This stage compares inverse, explicitly regularized, and iterative
reconstructions across saved Phase 2--7 deblurring experiments. It reads CSV
evidence only. It never runs a blur model, solver, or previous experiment.

Source-row metrics are joined to the P8-02 compatibility audit. The P8-07
taxonomy prevents final outcomes, parameter candidates, aggregate summaries,
metadata, and solver trajectories from being pooled. P8-08 is a required input.

`strict_compatible` means every metric used by a row is compatible.
`descriptive_unknown_configuration` retains provenance but is not a controlled
comparison. Incompatible rows are excluded from strict tables. Tables 1--3 and
parts of Table 6 are descriptive because their audit records remain unknown.

## Tables

| Table | Scope and contents |
|---|---|
| 1 | Three Phase 2 inverse/QR/normal-equation results. Near-zero residuals occur alongside relative errors around 1.95e4--2.03e4. Residual fit is not reconstruction accuracy. |
| 2 | Seventeen Phase 4 TSVD/Tikhonov outcomes, separated from path candidates. The lowest saved error is Tikhonov lambda=0.1 at about 0.186. Full-rank TSVD returns the unstable unregularized result. |
| 3 | Six Phase 5 selected outcomes, separated by TSVD/Tikhonov and criterion. TSVD-GCV rank 142 with error about 1.73e4 is retained as a noise-dominated failure. |
| 4 | Sixty-three strict-compatible Phase 6 groups. `n_unique_cases` prevents repeated metrics for one case being treated as independent replicates. |
| 5 | Eight compatible equal-budget Phase 7 rows: four LSQR and four damped-LSQR. Trajectory and tolerance rows are excluded. |
| 6 | SVD summary, blur-condition observations, and structured/full-SVD validation. These explain conditioning and are not pooled with outcomes. |
| 7 | Eight compatible equal-budget Phase 7 rows. Storage columns are theoretical Float64 array estimates, not measured peak RAM. |

`phase8_comparative_analysis_audit.csv` records 5,021 compatible, 288
incompatible, and 3,432 unknown canonical metric records. These are audit
records, not independent reconstructions.

## Figures

Figures 1--12 in `results/figures/phase8/` show baseline error,
regularization error/residual, L-curves, GCV/discrepancy diagnostics, noise,
blur, resolution, iterative residual and error trajectories, singular-value
decay, runtime evidence, and theoretical storage scaling. Their titles disclose
whether evidence is strict-compatible, descriptive, parameter-candidate, or
theoretical. Trajectories are checkpoints within runs, not independent outcome
rankings. Runtime is machine-specific, and storage is not peak-RAM measurement.

## Findings and limitations

- Direct inverse and QR have tiny residuals but catastrophic reconstruction
  error; normal equations remain catastrophically inaccurate. Lower residual
  alone is inadequate in this ill-conditioned problem.
- Saved Phase 4 results show stabilization at selected TSVD/Tikhonov settings,
  but unknown audit status precludes a new universal ranking.
- The TSVD-GCV high-rank case fits observations closely yet reconstructs
  catastrophically. Ground truth remains post-selection evaluation only.
- Phase 6 strict evidence contains stable regularized and unstable
  unregularized cases. The normal-equation noise group has mean relative error
  about 7.45e4, five unstable unique cases, and small mean residual.
- Phase 7 does not establish a globally superior iterative method. It supports
  compatible equal-budget observations and iteration-dependent trajectories.

No missing metric was imputed, aggregate Phase 6 counts were not added to
individual-case counts, and no significance or universal-best-method claim is
made. This work does not implement P8-05 or modify prior Phase 8 artifacts.
