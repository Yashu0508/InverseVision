# Phase 6 — Robustness and Sensitivity Analysis

**Project:** Numerical Stability and Regularization in Image Deblurring  
**Status:** Phase 6 experiment implemented and run; Phase 7 has not started.  
**Interpretation:** Findings are empirical results from a small controlled
grid. They do not establish a universally superior method or statistical
significance.

## 1. Phase objective

Phase 5 demonstrated a striking TSVD-GCV outcome for one 12×12 camera image:
GCV selected rank 142 and the reconstruction error was very large. Phase 6
tests how the Phase 5 selectors and Phase 2 unregularized baselines behave
when noise realization, Gaussian blur strength, or image resolution changes.

The experiment assesses parameter variability, raw reconstruction accuracy,
data residuals, solution magnitudes, condition numbers, and solver timings.
It records unstable outcomes instead of removing them from the data.

## 2. Experimental design and rationale

Eleven controlled cases were run, with nine method/criterion records per case
(99 records total):

| Sensitivity group | Varied settings | Held fixed |
|---|---|---|
| Noise realization | Seeds 7, 21, 42, 84, 126 | 12×12 image, sigma 1.5, kernel 9, noise std 0.01 |
| Blur strength | Sigma 0.8, 1.5, 2.2 | 12×12 image, noise std 0.01, seed 42, kernel 9 |
| Image resolution | 6×6, 12×12, 16×16 | Sigma 1.5, noise std 0.01, seed 42, kernel 9 |

The established Phase 5 reference (12×12, sigma 1.5, seed 42) appears once in
each sensitivity group. These repeated rows make the varied factor explicit;
they are the same deterministic configuration, not three independent
realizations.

Images are the same normalized scikit-image camera source resized with Pillow
LANCZOS at each requested resolution. All groups use the existing zero-boundary
Gaussian convention with no boundary-row renormalization and C-order
vectorization. The blur group uses the same seed at each sigma, giving the
same seeded noise array while changing the operator. Noise seeds vary only
within the noise group.

Every image size stays within the existing exact dense SVD guard of 256 rows
or columns: tested square operator dimensions are 36, 144, and 256. Nothing is
silently switched to an approximate or matrix-free SVD method.

## 3. Mathematical methodology

For each case, one shared forward operator A, noisy observation b, and SVD
are prepared. All six existing Phase 5 selectors are run on those same
objects:

- TSVD rank selected by L-curve curvature, GCV, and discrepancy principle.
- Standard L2 Tikhonov lambda selected by the same three criteria.

The selectors receive no reference image. All six selections finish before
the experiment accesses the original image for postselection metrics. The
three Phase 2 baselines (direct inverse, QR least squares, and normal
equations) are also run; their existing dense solvers accept all tested
systems. They share the same observation and operator.

The Phase 5 definitions remain unchanged. TSVD candidates cover all integer
ranks from 1 through numerical rank. Tikhonov uses 201 log-spaced lambdas from
1e-6 times the largest singular value through 10 times that value. The
discrepancy target is 1.05 times noise standard deviation times sqrt(m). Raw,
unclipped arrays are used for all metrics. Image display scaling is applied
only in plots.

An output is flagged unstable when relative reconstruction error is at least
10 or the ratio of reconstruction norm to reference norm is at least 10, or
when required quantities are nonfinite. These are explicit operational
diagnostics for this study, not a universal definition and not a
parameter-selection criterion.

Descriptive tables use mean, sample standard deviation (ddof=1), median,
minimum, and maximum. For a one-record blur or resolution condition, sample
standard deviation is left undefined rather than reported as zero. No
significance tests were performed.

## 4. Configuration and reproducibility

The design is centralized in src/utils/phase6_protocol.py; aggregation and
the instability diagnostic are in src/evaluation/robustness.py. The runner
is experiments/phase6_robustness_sensitivity.py.

Run the complete experiment from the repository root:

    $env:MPLBACKEND = 'Agg'
    python experiments/phase6_robustness_sensitivity.py

Run Phase 6 tests and then the complete regression suite:

    $env:MPLBACKEND = 'Agg'
    python -m pytest tests/test_phase6_robustness.py -q
    python -m pytest

The exact case grid, seeds, tolerance, kernel support, boundary convention,
candidate grids, dense limit, and instability thresholds are written to
results/metrics/phase6_experiment_config.json. Execution events are logged
to results/logs/phase6_experiment.log.

## 5. Results and visualizations

All 11 cases completed. All 99 method/criterion runs returned successfully;
the large or poor reconstructions are retained in the raw data. There were
34 instability flags under the stated diagnostic rule. Thirty are the three
unregularized baselines across the ten cases other than sigma 0.8. The
remaining flags are selected regularized solutions: TSVD-GCV at the shared
12×12 reference condition (repeated in three design groups) and Tikhonov-GCV
at 6×6.

### Noise-realization sensitivity

Statistics below are across five independent seeded realizations, with blur
and resolution held at the Phase 5 reference values:

| Method / criterion | Selected parameter: mean ± sample SD (range) | Relative error: mean ± sample SD (median; range) | Unstable |
|---|---|---|---:|
| TSVD / L-curve | rank 39.4 ± 14.21 (24–56) | 0.2377 ± 0.0889 (0.2020; 0.1625–0.3890) | 0/5 |
| TSVD / GCV | rank 55.4 ± 48.43 (32–142) | 3463.56 ± 7744.38 (0.1722; 0.1608–17317.14) | 1/5 |
| TSVD / discrepancy | rank 23.6 ± 5.50 (18–32) | 0.1963 ± 0.0192 (0.2057; 0.1623–0.2070) | 0/5 |
| Tikhonov / L-curve | lambda 0.01505 ± 0.00263 (0.01111–0.01802) | 0.2555 ± 0.0324 (0.2605; 0.2062–0.2964) | 0/5 |
| Tikhonov / GCV | lambda 0.03093 ± 0.00448 (0.02487–0.03721) | 0.1755 ± 0.0091 (0.1742; 0.1635–0.1858) | 0/5 |
| Tikhonov / discrepancy | lambda 0.07670 ± 0.01367 (0.05568–0.09030) | 0.1714 ± 0.0094 (0.1718; 0.1574–0.1824) | 0/5 |

TSVD-GCV selected rank 142 only for seed 42; the other selections were ranks
32, 33, 34, and 36. That single unstable realization dominates its mean error,
so median and range are also reported. Tikhonov GCV and discrepancy were
comparatively consistent over these five seeds. TSVD discrepancy selected
lower ranks with less variation than TSVD GCV in this grid. Five seeds are a
small sample and do not establish general performance.

### Blur-strength sensitivity

| Sigma | Condition number | TSVD L-curve / GCV / discrepancy error | Tikhonov L-curve / GCV / discrepancy error | Baseline error (direct / QR / normal) |
|---:|---:|---:|---:|---:|
| 0.8 | 109.6 | 0.0773 / 0.0839 / 0.1076 | 0.5553 / 0.1764 / 0.0769 | 0.2378 / 0.2378 / 0.2378 |
| 1.5 | 5.68e7 | 0.2020 / 17317.14 / 0.2070 | 0.2964 / 0.1858 / 0.1824 | 19488.53 / 19488.53 / 20340.13 |
| 2.2 | 4.88e5 | 0.4407 / 0.2258 / 0.2403 | 0.2540 / 0.2138 / 0.2056 | 1612.55 / 1612.55 / 1612.55 |

The selected TSVD GCV rank moved from 70 at sigma 0.8 to 142 at sigma 1.5,
then 22 at sigma 2.2. The condition number is not monotone over these three
points: the finite Gaussian kernel support, zero-boundary rows, and discrete
operator make conditioning depend on the full matrix spectrum, not just
sigma. The three blur levels are single realizations each, so these are
observed sensitivity points, not repeated-sample uncertainty estimates.

### Image-resolution sensitivity

| Resolution | Operator dimension | Numerical rank | Condition number | TSVD L-curve / GCV / discrepancy error | Tikhonov L-curve / GCV / discrepancy error |
|---|---:|---:|---:|---:|---:|
| 6×6 | 36 | 36 | 4.64e5 | 0.1289 / 0.1687 / 0.2323 | 0.2289 / 27.8947 / 0.1838 |
| 12×12 | 144 | 144 | 5.68e7 | 0.2020 / 17317.14 / 0.2070 | 0.2964 / 0.1858 / 0.1824 |
| 16×16 | 256 | 256 | 1.63e7 | 0.3134 / 0.1538 / 0.1759 | 0.2786 / 0.1655 / 0.1419 |

At 6×6, Tikhonov GCV chose lambda approximately 6.93e-05, yielding relative
error 27.89 and reconstruction norm 94.58. At 12×12, TSVD GCV again chose
rank 142 and had relative error 17317. At 16×16, all six regularized selections
remained below relative error 0.314. Since the underlying camera is resampled
at each resolution, cross-resolution error differences include sampling
effects as well as inverse-problem dimension effects.

### Phase 2 baselines and method-wise instability

The unregularized direct, QR, and normal-equations baselines had very large
errors in most cases. They were stable under the diagnostic threshold only at
sigma 0.8; at all other conditions their measured errors exceeded the
threshold. This is consistent with their educational, unregularized role.
Normal equations can be slightly worse due to squaring the condition number.

Across all output rows, 34/99 were flagged unstable. Because the Phase 5
reference condition is intentionally repeated across the three sensitivity
groups, the three TSVD-GCV flags there describe one unique seed/configuration,
not three independent failures. Across the five noise seeds, TSVD GCV had one
unstable result (1/5); Tikhonov GCV had none (0/5). Tikhonov GCV did have one
unstable result at 6×6. This supports a configuration-specific concern about
GCV, not a claim that GCV is always unstable.

### Figures

- results/plots/phase6_noise_sensitivity.png — relative error versus noise
  seed.
- results/plots/phase6_blur_sensitivity.png — relative error versus sigma.
- results/plots/phase6_resolution_sensitivity.png — relative error versus
  pixel count.
- results/plots/phase6_selected_parameters.png — selected rank/lambda by
  criterion across all three experimental axes.
- results/plots/phase6_noise_variability.png — method-wise error distributions
  across the five noise seeds.
- results/plots/phase6_error_heatmap.png — all 11 configurations and nine
  methods, log10 relative error.

## 6. Comparative interpretation

Within the five-noise-seed group, Tikhonov GCV and discrepancy had the lowest
and most consistent average errors among the automatic selectors in this
particular grid. TSVD discrepancy was also comparatively steady. L-curve
outcomes varied more and selected different effective regularization
strengths. TSVD-GCV showed a severe single-realization failure at the Phase 5
reference condition; Tikhonov-GCV also failed at 6×6. The Phase 2 baselines
were generally highly unstable under increased blur.

These observations do not establish one selector as universally best. The
grid is small, blur and resolution conditions each have one realization, the
image is a single source, and discrepancy is given the assumed noise standard
deviation. These results should guide further study, not serve as a
statistical ranking.

## 7. Limitations and unresolved questions

- Only five noise realizations were evaluated; more seeds and multiple source
  images are needed for stronger robustness claims.
- Blur strength and resolution each have one controlled seed/configuration per
  point, so their sample standard deviations are undefined.
- Only three square resolutions up to 16×16 were tested to preserve exact SVD
  use under the 256-dimension cap. Larger image sizes remain unsupported by
  these dense SVD selectors; no approximation was substituted.
- Resolution changes also resample the source image. A later design could use
  a carefully defined continuous image reference if isolating discretization
  effects is needed.
- The instability threshold is explicitly heuristic. Both raw metrics and the
  flag are saved so readers can apply other definitions.
- Execution times are per-call small-system timings and should not be treated
  as reliable performance benchmarks.
- No statistical significance claims or tests are made.

## 8. Tests and validation

New tests in tests/test_phase6_robustness.py cover controlled design
construction, dense-limit and invalid-configuration rejection, deterministic
same-seed noise, distinct-seed noise, forward/operator/observation consistency,
descriptive aggregation, instability thresholds, and ground-truth-free
selector API. Existing Phase 0–5 tests remain part of the full regression
suite.

Final test commands:

    $env:MPLBACKEND = 'Agg'
    python -m pytest tests/test_phase6_robustness.py -q
    python -m pytest

Final results: 6 Phase 6 focused tests passed; 138 tests passed in the full
suite.

## 9. Output inventory and exact reproduction

The experiment writes Phase 6-specific outputs only; Phase 5 artifacts remain
intact:

- results/metrics/phase6_raw_results.csv — 99 method-case records with
  selected parameters, reconstruction/error metrics, residuals, solution
  norms, numerical ranks, condition numbers, timings, diagnostics, and
  instability flags.
- results/metrics/phase6_aggregated_summary.csv — mean, sample standard
  deviation, median, min, max, and sample count by metric and group.
- results/metrics/phase6_selected_parameters.csv — selector status, selected
  parameter/index/score, and diagnostics.
- results/metrics/phase6_experiment_config.json — case list and fixed design
  metadata including every seed.
- results/plots/phase6_*.png — six plots listed above.
- results/images/phase6_*_raw.npy — 99 raw, unclipped reconstruction arrays,
  one per method/case.
- results/logs/phase6_experiment.log — per-case rank, condition number, method
  count, instability count, and completion.

Reproduce the Phase 6 artifacts from the project root:

    $env:MPLBACKEND = 'Agg'
    python experiments/phase6_robustness_sensitivity.py

The process completed in approximately 10 seconds in the recorded Python
3.13.17 environment; exact timing depends on hardware and BLAS/LAPACK.

## 10. Files created or modified

Created:

- src/utils/phase6_protocol.py
- src/evaluation/robustness.py
- experiments/phase6_robustness_sensitivity.py
- tests/test_phase6_robustness.py
- report/phase6_robustness_sensitivity_analysis.md

Updated:

- README.md
- docs/development.md
- report/project_handoff_phase_0_phase_5.md (Phase 6 status and report link)

Generated outputs are enumerated above. No Phase 0–5 numerical solver,
regularization, forward-model, selection, configuration, or dependency files
were changed. Phase 5 result files were not overwritten.

## Final status

The controlled Phase 6 framework is implemented and all planned cases ran
successfully. All methods used their case's same forward operator and
observation. Ground truth was used only after selection, outputs are
reproducible from recorded seeds/configuration, and all cases respected the
exact dense-SVD limit. Findings document a recurrent risk from noise-sensitive
GCV choices without claiming universal superiority for any alternative.
Phase 7 has not started.
