# Phase 8 Final Independent Validation (P8-05)

## A. Overall verdict

**PASS — Phase 8 complete.**

This conclusion is based on a fresh read of the saved source CSVs, Phase 8
artifacts, comparative outputs, stage registry, command-line dispatch, and
tests. No Phase 2--7 experiment was executed during this validation. The
limitations below restrict scientific interpretation, not implementation
completeness.

## B. Eight-issue checklist

| Issue | Status | Independent evidence |
|---|---|---|
| P8-01 Canonical schema | Pass | 8,741 unique deterministic canonical IDs; 8,134 observed numeric values and 607 explicitly missing values. Boolean `safe_denominator` is not a canonical metric. |
| P8-02 Compatibility audit | Pass | 8,741 audit IDs match canonical IDs exactly: 5,021 compatible, 288 incompatible, 3,432 unknown. Unknown is not promoted to compatible. |
| P8-03 Source inventory | Pass | All 17 discovered Phase 2--7 CSV sources appear once, exist, load, and have inventory row counts matching source rows. |
| P8-04 Comparative analysis | Pass | Seven tables, one comparison-status audit, and 12 nonempty PNG figures exist; recalculated spot checks agree with saved source records. |
| P8-05 Final validation | Pass | Independent integrity tests, CLI checks, focused tests, full regression, and pre/post SHA-256 comparisons pass. |
| P8-06 Stage separation | Pass | The registry has isolated handlers, declared inputs/outputs, explicit `full-core-pipeline`, no default stage, and clear missing-input errors. |
| P8-07 Record taxonomy | Pass | Every canonical ID occurs exactly once in the classification audit and every row is `classified`. |
| P8-08 Metadata reconciliation | Pass | The field-level audit covers every canonical ID; reconciled entries carry evidence source and location, while missing and not-applicable remain distinct. |

## C. Numerical and data integrity

The independent recalculation found 8,741 canonical records and no duplicate
canonical IDs. The classification audit also contains 8,741 rows. Its analytical
category counts are: experimental metadata 8, explicit regularization 271,
iterative solver validation 28, iterative solvers 1,056, parameter selection
2,451, reconstruction baseline 21, robustness sensitivity 4,299, and spectral
conditioning 607. These categories are mutually exclusive at the audit-row
level and sum to 8,741.

Metadata status counts are: present verified 25,283; reconciled 47,389;
missing 40,938; not applicable 8,764; and unmapped source field 85. The first
four statuses cover the 14 mapped fields for every canonical record. The 85
unmapped-field rows preserve source information rather than discarding it. All
47,389 reconciled rows have nonempty evidence source and evidence location.
There are no conflicting-status rows in the saved audit.

Recalculated comparison checks confirmed the Phase 2 baseline values, the
Phase 4 Tikhonov lambda=0.1 error of 0.18593123130628608, the Phase 5
TSVD-GCV rank 142 error of 17317.138052621427, the Phase 6 normal-equations
noise-group mean error, and every selected Table 7 runtime against its unique
Phase 7 iteration-budget source observation.

## D. Provenance and comparison validity

Source inventory records the relative source path, phase, loading status, row
count, and record unit. Canonical rows retain source file, source row, and
source attributes. Classification and compatibility audits both retain the
canonical ID. Comparative Tables 1--3, 5--7 retain source-file/row or a unique
case-method-iteration key; Table 4 identifies the grouping dimensions and
unique-case count used for aggregation.

Strict controlled comparisons use compatible records only. This includes the
Phase 6 sensitivity table and the eight retained Phase 7 equal-budget rows.
Unknown Phase 2--5 and some Phase 3 evidence remains explicitly descriptive.
The 288 incompatible audit records are excluded from strict comparative tables.
Aggregate summaries are not counted as independent individual-case results.

All 12 figures are valid PNGs with at least 800 by 500 pixels. Their generation
code labels descriptive, strict-compatible, candidate, mixed, and theoretical
uses. The storage figure explicitly says that it is not measured peak RAM.

## E. Scientific conclusion

The saved evidence supports the following bounded answer to the project
question. Small residuals do not guarantee accurate deblurring: the direct
inverse and QR baseline residuals are tiny while reconstruction errors are
catastrophic. Ill-conditioning makes noise amplification consequential.
TSVD and Tikhonov regularization stabilize selected outcomes by accepting a
bias--variance trade-off. Parameter selection is not fail-safe: the retained
TSVD-GCV high-rank selection has a very small residual but catastrophic error.
Iterative results demonstrate convergence and semiconvergence behaviour.
Matrix-free representations reduce stated storage requirements, but saved
timings are machine-specific and dense-factor arithmetic remains relevant.

The experiment set uses limited images, realizations, and configurations.
Therefore it does not establish a universal best method or a general
statistical ranking.

## F. Reproducibility and commands

The validated commands were:

```powershell
python experiments/phase8_integrated_analysis.py --list-stages
python experiments/phase8_integrated_analysis.py --help
python -m pytest tests -k phase8 -q
python -m pytest -q
```

The focused command passed 34 tests with 151 deselected. The complete project
suite passed 185 tests. To regenerate only P8-04 from saved evidence, use:

```powershell
python experiments/phase8_integrated_analysis.py --stage comparative-analysis
```

This command consumes saved upstream artifacts and does not run a Phase 2--7
experiment. The explicit `full-core-pipeline` command is the only command that
recomputes all registered stages.

## G. Artifact integrity and corrections

SHA-256 hashes were captured before and after validation for Phase 8 metrics,
figures, prior plots, and prior Phase 8 reports. They were unchanged. The
validation itself created this report and added independent read-only tests; it
did not regenerate tables, figures, canonical data, audits, or Phase 2--7
outputs.

No production defect was found and no production artifact was corrected. One
new validation-test expectation was refined after inspection: the canonical
schema intentionally records 607 absent numerical values with status `missing`.
That is valid missing-data preservation, not a schema defect.
