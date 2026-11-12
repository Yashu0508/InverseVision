# Project Handoff Report: Phases 0–3

**Project:** Numerical Stability and Regularization in Image Deblurring  
**Current status:** Phases 0–3 are complete. Phase 4 has not started.  
**Validated runtime:** Python 3.13.17

## Research aim

The project studies the ill-posed inverse problem

\[
b=Ax+\epsilon,
\]

where \(x\) is a sharp grayscale image, \(A\) is a zero-boundary Gaussian blur
operator, \(b\) is a noisy observation, and \(\epsilon\) is additive Gaussian
noise. All image vectors use C-order (row-major) flattening.

## Completed phases

- **Phase 0:** modular Python scaffold, configuration, dependencies,
  documentation, and tests.
- **Phase 1:** image validation, Gaussian kernels, explicit sparse and
  matrix-free blur operators, reproducible noise, degradation pipeline, and
  visualization.
- **Phase 2:** small dense direct-inverse, QR, and normal-equations baselines,
  reconstruction metrics, and a shared-observation comparison. These methods
  produced very small residuals but extremely inaccurate raw reconstructions.
- **Phase 3:** analysis-only SVD, conditioning, numerical rank, Picard, noise
  amplification, blur-strength sensitivity, and separable-structure studies.

No Phase 3 function constructs a reconstructed solution.

## Phase 3 implementation

### Full SVD

`src/analysis/svd_analysis.py` computes the economy decomposition

\[
A=U\Sigma V^T
\]

with SciPy. It returns every computed singular value, both singular-vector
matrices, matrix dimensions, numerical rank, rank threshold, largest and
smallest singular values, condition number, and diagnostics.

Full dense SVD rejects a row or column dimension above **256** before converting
a sparse matrix. This permits small validation systems while preventing dense
decomposition of the default 4096-by-4096 operator.

Numerical rank uses the explicit convention

\[
\tau=\mathrm{rtol}\,\sigma_1,
\qquad
r_{\mathrm{num}}=\#\{i:\sigma_i>\tau\}.
\]

Experiments use `rtol=1e-12`. A spectrum that is not numerically full rank under
this threshold receives infinite numerical condition number, while its actual
smallest computed singular value remains available.

### Structured Kronecker spectrum

For \(A=B_y\otimes B_x\), factor SVDs imply

\[
A=(U_y\otimes U_x)(\Sigma_y\otimes\Sigma_x)
(V_y\otimes V_x)^T.
\]

Therefore the singular values of \(A\) are all products

\[
\sigma_{ij}(A)=\sigma_i(B_y)\sigma_j(B_x).
\]

`analyze_structured_svd(...)` decomposes only the two one-dimensional factors,
forms and sorts these products, and applies the same tolerance convention. It
does not form the full Kronecker matrix and supports rectangular factors.

### Spectrum, Picard, and noise analysis

- `singular_value_decay(...)` returns normalized values and cumulative squared
  singular-value energy without discarding small values.
- `src/analysis/picard_analysis.py` computes \(u_i^Tb\), magnitudes, and safe
  \(|u_i^Tb|/\sigma_i\) ratios. Values at or below the reported denominator
  threshold are never directly divided.
- `src/analysis/noise_amplification.py` generates Phase 1-compatible seeded
  Gaussian noise, computes \(U^T\epsilon\), and analyzes
  \(|u_i^T\epsilon|/\sigma_i\). It returns no reconstruction or filtering rule.
- `src/analysis/blur_sensitivity.py` uses the structured spectrum to compare
  fixed-convention Gaussian operators over multiple sigma values.

## Computed SVD demonstration

Configuration: 8-by-8 image, sigma 1.5, kernel size 9, zero boundary, and
`rtol=1e-12`.

| Quantity | Computed value |
|---|---:|
| Matrix shape | 64-by-64 |
| Largest singular value | 0.7986339725 |
| Smallest singular value | 4.508401552e-07 |
| Numerical rank | 64/64 |
| Rank threshold | 7.986339725e-13 |
| Numerical condition number | 1,771,434.8717 |
| Components containing 90% squared energy | 7 |

The operator is numerically full rank under the chosen tolerance but highly
ill-conditioned. This distinguishes numerical sensitivity from exact or
thresholded rank deficiency.

## Picard and noise findings

The Picard experiment uses the same 8-by-8 camera image, sigma 1.5, kernel size
9, noise standard deviation 0.01, seed 42, and rank tolerance `1e-12`.

- Median singular value falls from `1.8719e-01` in the first 16 directions to
  `4.2841e-05` in the last 16.
- Median observation-coefficient magnitude falls only from `6.2613e-02` to
  `5.7660e-03`; it reaches a noise-dominated floor rather than following the
  singular values downward.
- The median coefficient/singular-value ratio rises from `0.3273` to `87.0860`
  between those quartiles.
- Raw noise norm is `0.06323556258`; the transformed-noise norm is
  `0.06323556258`, agreeing to floating-point precision for this square
  orthogonal basis.
- Median inverse-direction noise contribution rises from `0.01941` in the first
  quartile to `87.15848` in the last quartile.
- Maximum computed finite noise contribution is `4897.2952`.

These are computed observations, not a selected truncation rule. No TSVD index
is proposed.

## Blur-strength sensitivity

Configuration: 16-by-16 image, kernel size 9, zero boundary, and fixed
`rtol=1e-12`. Spectra were calculated structurally.

| Sigma | Largest SV | Smallest SV | Numerical rank | Condition number | Median normalized SV | Components for 90% energy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.9926962 | 3.338394e-01 | 256/256 | 2.9736 | 5.951142e-01 | 197 |
| 1.0 | 0.9677226 | 2.826407e-04 | 256/256 | 3.4239e+03 | 5.226950e-02 | 50 |
| 1.5 | 0.9340134 | 5.729414e-08 | 256/256 | 1.6302e+07 | 1.854322e-03 | 23 |
| 2.0 | 0.9022329 | 3.991573e-08 | 256/256 | 2.2603e+07 | 1.387430e-03 | 14 |
| 3.0 | 0.8652133 | 3.680512e-08 | 256/256 | 2.3508e+07 | 2.574095e-03 | 9 |

The computed condition number rises sharply through sigma 1.5 and then changes
more modestly for sigma 2–3 under the fixed nine-sample kernel. No monotonic
trend was hardcoded; the reported plateau-like behavior is the actual result
for this truncated kernel and finite grid.

## Structured versus full SVD

Configuration: 8-by-8 image, sigma 1.5, kernel size 9, seven measured repeats
after one warm-up call.

| Quantity | Computed value |
|---|---:|
| Maximum absolute spectrum difference | 8.3267e-16 |
| Maximum relative spectrum difference | 3.1980e-11 |
| Full median time | 2.2787e-03 s |
| Structured median time | 4.5400e-05 s |
| Dense operator input storage | 32,768 bytes |
| Structured factor input storage | 1,024 bytes |
| Full SVD output-array storage | 66,048 bytes |
| Structured spectrum output storage | 640 bytes |

The timing values are local repeated measurements, not general benchmarks. The
full workload computes singular vectors, whereas the structured workload only
needs factor spectra and pairwise products. The storage values count explicit
NumPy input/output arrays and do not claim to measure all temporary LAPACK
workspace or interpreter overhead.

## Artifacts

Metrics:

- `results/metrics/phase3_svd_summary.csv`
- `results/metrics/phase3_svd_spectrum.csv`
- `results/metrics/phase3_picard_noise.csv`
- `results/metrics/phase3_blur_sensitivity.csv`
- `results/metrics/phase3_structured_vs_full_svd.csv`

Plots:

- `results/plots/phase3_svd_analysis.png`
- `results/plots/phase3_picard.png`
- `results/plots/phase3_noise_amplification.png`
- `results/plots/phase3_blur_sensitivity.png`
- `results/plots/phase3_structured_vs_full_svd.png`

## Validation

The complete Phase 0–3 regression suite passes under Python 3.13.17:

```text
90 passed in 2.59s
```

Coverage includes SVD reconstruction, ordering, dimensions, invalid inputs,
tolerance-dependent rank, rank deficiency, condition number, rectangular
structured factors, full/structured agreement, Picard alignment and safe
division, reproducible unclipped noise analysis, blur-sensitivity record
structure, Phase 3 plots, and every prior test.

## Strict next-phase boundary

Not implemented:

- TSVD or truncation-parameter selection;
- Tikhonov or total-variation regularization;
- L-curve, GCV, discrepancy principle, or other parameter selection;
- iterative reconstruction; and
- advanced robustness studies.

Preserve C-order vectorization, the zero-boundary convention, raw unclipped
arrays, and explicit tolerance reporting. Await user approval before Phase 4.
