# Project Handoff Report: Phases 0–4

**Project:** Numerical Stability and Regularization in Image Deblurring  
**Status:** Phase 4 complete; Phase 5 has not started.  
**Validated environment:** Python 3.13.17, pytest 8.4.2

## Objective

The project studies the ill-posed model \(b=Ax+\epsilon\), where \(x\) is a
sharp grayscale image, \(A\) is the established zero-boundary Gaussian blur
operator, and \(\epsilon\) is Gaussian noise. Phase 4 added TSVD and standard
L2 Tikhonov to investigate reconstruction stability. All vectors use C-order.

## Phase 4 implementation

### TSVD

`src/regularization/tsvd.py` implements

\[
\hat{x}_k=\sum_{i=1}^k\frac{u_i^Tb}{\sigma_i}v_i.
\]

It reuses the Phase 3 SVD factorization, checks the requested integer `k` and
rejects values beyond numerical rank under the existing `rtol=1e-12` experiment
convention. It preserves raw output and reports `A @ x_hat - b`, residual
norms, timing, rank, threshold, and diagnostics. It does not alter `k`.

TSVD filters distinguish the inverse factor \(f_i=1/\sigma_i\) on retained
directions from the dimensionless retention factor \(\phi_i=1\). Both are zero
for discarded directions.

### Tikhonov

`src/regularization/tikhonov.py` implements

\[
\hat{x}_\lambda=\arg\min_x\bigl(\|Ax-b\|_2^2+\lambda^2\|x\|_2^2\bigr)
\]

with the SVD filter formulation

\[
\hat{x}_\lambda=\sum_i\frac{\sigma_i}{\sigma_i^2+\lambda^2}
(u_i^Tb)v_i.
\]

It also implements an augmented least-squares solve using \([A;\lambda I]\),
without forming \(A^TA\). Lambda must be finite and positive. The inverse
filter is \(f_i=\sigma_i/(\sigma_i^2+\lambda^2)\); the dimensionless retention
filter is \(\phi_i=\sigma_i^2/(\sigma_i^2+\lambda^2)\).

Both solvers retain raw reconstructions. Metrics reuse
`src/evaluation/reconstruction_metrics.py`; residual fit and ground-truth
reconstruction accuracy are recorded independently. SSIM uses data range one
without clipping the reconstruction, so it may be uninformative for extreme
out-of-range raw estimates.

Dense regularization input is guarded at 256 dimensions before sparse matrices
are materialized. The Phase 2 baseline code and Phase 1–3 numerical modules were
not changed.

## Shared experiment setup

All Phase 4 scripts use one shared camera image, operator, and observation:

| Setting | Value |
|---|---:|
| Image dimensions | 12×12 |
| Gaussian sigma | 1.5 |
| Kernel size | 9 |
| Boundary | Zero padding, no boundary renormalization |
| Noise standard deviation | 0.01 |
| Seed | 42 |
| Numerical rank relative tolerance | 1e-12 |

Predetermined TSVD ranks: `5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 144`.
The measured numerical rank was 144, so every requested rank was valid.

Predetermined Tikhonov lambdas: `1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6`.
These values are exploratory and no automatic selection was performed.

## Results

### TSVD sweep

| k | Relative error | MSE | PSNR dB | SSIM | Residual norm |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.3832 | 0.04804 | 13.18 | 0.5765 | 0.9585 |
| 10 | 0.2942 | 0.02831 | 15.48 | 0.7782 | 0.4096 |
| 15 | 0.2574 | 0.02168 | 16.64 | 0.8187 | 0.2591 |
| 20 | 0.2060 | 0.01388 | 18.58 | 0.8012 | 0.1174 |
| 30 | 0.2048 | 0.01373 | 18.62 | 0.8186 | 0.09705 |
| 40 | 0.1894 | 0.01174 | 19.30 | 0.8763 | 0.08215 |
| 60 | 0.4706 | 0.07245 | 11.40 | 0.6115 | 0.07063 |
| 80 | 2.9587 | 2.8643 | -4.57 | 0.04882 | 0.06163 |
| 100 | 13.5443 | 60.0230 | -17.78 | 0.00620 | 0.04983 |
| 120 | 93.1000 | 2835.9757 | -34.53 | approximately 0 | 0.03656 |
| 144 | 19488.5339 | 124268649.3850 | -80.94 | approximately 0 | 1.3840e-11 |

As more components are retained, data residual decreases, while reconstruction
error eventually grows rapidly as noise-dominated directions enter. No rank is
declared universally optimal.

### Tikhonov sweep

| lambda | Relative error | MSE | PSNR dB | SSIM | Residual norm |
|---:|---:|---:|---:|---:|---:|
| 1e-1 | 0.1859 | 0.01131 | 19.46 | 0.8555 | 0.1389 |
| 1e-2 | 0.3196 | 0.03342 | 14.76 | 0.7625 | 0.07116 |
| 1e-3 | 2.9258 | 2.8008 | -4.47 | 0.05885 | 0.05813 |
| 1e-4 | 25.5959 | 214.3610 | -23.31 | 0.00030 | 0.04228 |
| 1e-5 | 196.6749 | 12656.1295 | -41.02 | approximately 0 | 0.02685 |
| 1e-6 | 1317.2852 | 567757.4810 | -57.54 | approximately 0 | 0.01120 |

For each lambda, the SVD and augmented solutions agreed. Maximum absolute
differences ranged from `6.55e-15` at `1e-1` to `5.71e-08` at `1e-6`; the
relative difference is also included in the CSV. The smaller-lambda augmented
solve is more sensitive to finite precision, as expected from the increasingly
ill-conditioned augmented problem.

### Baseline comparison

The comparison includes the three Phase 2 methods, all fixed TSVD ranks, and
all fixed Tikhonov lambdas using the same observation. At full rank, TSVD
approaches the highly unstable unregularized result. The same happens as lambda
shrinks. The reported data show the trade-off directly: smaller residuals can
coexist with much worse reconstruction accuracy.

## Generated artifacts

CSV files:

- `results/metrics/phase4_tsvd_results.csv`
- `results/metrics/phase4_tikhonov_results.csv`
- `results/metrics/phase4_regularization_comparison.csv`

Plots:

- `results/plots/phase4_tsvd_reconstructions.png`
- `results/plots/phase4_tsvd_tradeoffs.png`
- `results/plots/phase4_tsvd_filters.png`
- `results/plots/phase4_tikhonov_reconstructions.png`
- `results/plots/phase4_tikhonov_tradeoffs.png`
- `results/plots/phase4_tikhonov_filters.png`
- `results/plots/phase4_regularization_comparison.png`
- `results/plots/phase4_regularization_comparison_tradeoff.png`

Raw, unclipped reconstructions are saved under `results/images/phase4_*_raw.npy`.
Plots use fixed grayscale display limits; no display operation alters the
numerical arrays or their evaluation.

## Testing and backend

The initial resumed test run exposed an intermittent Tk backend failure while
plot tests ran headlessly. `tests/conftest.py` now selects Matplotlib's `Agg`
backend before test modules import pyplot. Assertions were not weakened.

Final command:

```powershell
$env:MPLBACKEND='Agg'; python -m pytest
```

Result: **119 passed in 1.09s** under Python 3.13.17. This includes the Phase
0–3 regression suite and Phase 4 solver, metric, experiment, and visualization
tests.

## Files created during Phase 4

- `src/regularization/_common.py`
- `src/regularization/tsvd.py`
- `src/regularization/tikhonov.py`
- `src/evaluation/experiment_records.py`
- `src/utils/experiment_setup.py`
- `src/utils/phase4_protocol.py`
- `src/visualization/regularization.py`
- `experiments/phase4_tsvd_experiment.py`
- `experiments/phase4_tikhonov_experiment.py`
- `experiments/phase4_regularization_comparison.py`
- `tests/conftest.py`
- `tests/test_tsvd.py`
- `tests/test_tikhonov.py`
- `tests/test_regularization_comparison.py`
- `tests/test_regularization_visualization.py`

Package exports in `src/regularization/__init__.py`,
`src/evaluation/__init__.py`, `src/utils/__init__.py`, and
`src/visualization/__init__.py` were updated. README and development guidance
were updated for the completed phase.

## Current structure

```text
experiments/
  phase1_degradation_demo.py
  phase2_baseline_reconstruction.py
  phase3_*.py
  phase4_tsvd_experiment.py
  phase4_tikhonov_experiment.py
  phase4_regularization_comparison.py
src/
  analysis/                 Phase 3 SVD and conditioning tools
  evaluation/               Metrics and experiment records
  forward_model/            Phase 1 Gaussian forward model
  numerical_methods/        Phase 2 unregularized baselines
  preprocessing/            Image loading and validation
  regularization/           Phase 4 TSVD and Tikhonov
  utils/                    Shared experiment setup and protocol
  visualization/            Phase 1–4 plots
tests/                      Phase 0–4 regression suite
report/
  project_handoff_phase_0_phase_3.md
  project_handoff_phase_0_phase_4.md
results/
  images/                   Raw Phase 2 and Phase 4 arrays
  metrics/                  Phase 2–4 CSV results
  plots/                    Phase 1–4 figures
```

## Limitations and next phase

Experiments are small dense demonstrations, and timings are single-run solver
measurements rather than benchmarks. SSIM on very large raw estimates can be
near zero or otherwise uninformative because estimates are intentionally not
clipped. Augmented Tikhonov differences grow at the smallest lambda due to
floating-point sensitivity.

No L-curve, GCV, discrepancy principle, or automatic `k`/lambda selection was
implemented. Bayesian, TV, wavelet, nonnegative, deep-learning, and iterative
methods also remain outside this phase. **Phase 4 is complete; Phase 5 has not
started. Await user review and approval before proceeding.**
