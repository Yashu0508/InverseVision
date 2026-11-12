# Phase 7 — Scalable Matrix-Free Iterative Methods

**Project:** Numerical Stability and Regularization in Image Deblurring  
**Status:** Phase 7 implemented and executed; Phase 8 has not started.  
**Environment used:** Python 3.13.17.  
**Scope:** LSQR, CGLS, and damped LSQR. All reported values below come from
the saved experiment outputs; no dense approximation was used for large cases.

## 1. Research objective

The previous phases used exact dense SVD-based methods subject to a 256-row
or column dimension guard. Phase 7 asks whether the existing separable
Gaussian model can support useful larger-image reconstructions through
matrix-free iterative solvers, and how convergence, residual fit, accuracy,
and early-stopping regularization behave as resolution and blur change.

## 2. Mathematical formulations

### LSQR

LSQR applies Golub–Kahan bidiagonalization using only forward and adjoint
products. It solves the unregularized least-squares problem

    minimize_x ||A x - b||_2.

The implementation delegates recurrence and stopping tests to SciPy's LSQR,
then independently recomputes the actual residual and normal-equation
residual from the returned vector. Iteration limit, SciPy termination code,
condition estimate, tolerances, residuals, and solution norm are retained.

### CGLS

CGLS applies conjugate gradients to the normal equations without forming
their matrix. With initial x=0, r=b-Ax, s=A.T r, p=s, each step uses

    q = A p
    alpha = (s.T s) / (q.T q)
    x <- x + alpha p
    r <- r - alpha q
    s_new = A.T r
    beta = (s_new.T s_new) / (s.T s)
    p <- s_new + beta p.

Only LinearOperator matvec and rmatvec are called. Stopping checks the
normal-equation residual norm against the initial norm using the supplied
relative and absolute tolerances. The solver records every iterate and its
residual, normal residual, and solution norm. Final residual diagnostics are
recomputed independently rather than relying only on recurrence updates.

### Damped LSQR

The damped objective is

    minimize_x ||A x - b||_2^2 + lambda^2 ||x||_2^2.

SciPy's LSQR argument damp is set to lambda itself. It is not set to lambda
squared. This is the same convention as the Phase 4 standard-form Tikhonov
objective and solver.

### Early stopping

Iteration limits and residual-based stopping form an implicit regularization
choice: early iterates favor smoother, lower-amplification components, while
later iterations fit progressively smaller singular directions. The solver
does not use the reference image to choose an iteration. Ground-truth error is
computed only after each run, to diagnose semiconvergence.

## 3. Algorithmic implementation and matrix-free architecture

- src/numerical_methods/iterative_solvers.py provides LSQR, CGLS,
  zero-start LSQR iteration-budget runs, termination metadata, and histories.
- src/utils/phase7_protocol.py builds a normalized camera image, a Gaussian
  LinearOperator, the blurred image by applying that exact operator, and a
  seeded noise realization. It does not construct the 2-D sparse/dense blur
  matrix or compute an SVD.
- experiments/phase7_scalable_iterative_methods.py performs small-system
  reference validation and scalable experiments.
- numerical_methods package exports the new iterative APIs.

For square image side n, the existing separable operator stores two dense
one-dimensional factors, with float64 data storage proportional to
2 n squared. It does not store a full (n squared) by (n squared) image
operator. This matrix-free representation avoids full-operator storage, but
it does not by itself imply low arithmetic cost; the actual forward and
adjoint costs are detailed below. No peak RAM measurement was made. The
experiment records theoretical float64 array-storage estimates only, excluding
Python and SciPy object overhead.

For a conditioning diagnostic, the runner computes SVDs of the one-dimensional
Gaussian factors and uses the exact Kronecker identity
κ₂(B_y ⊗ B_x) = κ₂(B_y) κ₂(B_x). This does not form or factor the large image
operator. The diagnostic is recorded separately from LSQR's iterative
condition estimate.

The large-image runner only receives scipy.sparse.linalg.LinearOperator. It
does not pass a large case to the dense baseline, regularization, or SVD
functions. Full dense comparisons occur only in the 8×8 validation case.

### Forward and Adjoint Computational Complexity

The Phase 7 `LinearOperator` is implemented in
`src/forward_model/operators.py`. At construction it creates and retains dense
float64 matrices `B_y` of shape r×r and `B_x` of shape c×c for an image of
shape r×c. Its forward closure reshapes the C-order vector to `X` and executes
`B_y @ X @ B_x.T`; its adjoint closure executes `B_y.T @ X @ B_x`. Each
closure therefore performs two ordinary dense matrix multiplications.

Consequently, one forward application costs O(r²c + rc²) arithmetic and one
adjoint application has the same O(r²c + rc²) cost. For a square n×n image,
each is O(n³) (two dense n×n-by-n×n products). The finite Gaussian support k
is **not** exploited computationally: although `gaussian_blur_matrix_1d`
places nonzero entries only in a band of width at most k, those factors are
stored as dense arrays and NumPy dense `@` operations process their full
dimensions. Thus this implementation is not O(n²k) per application.

Persistent factor storage is O(r² + c²) float64 values (2n² for square
images); no r c by r c image operator is stored. During an application, the
reshaped image is a view when possible and the dense products create r×c
intermediate and result arrays, so temporary numerical-array storage is
O(rc), in addition to the factors and the input/output vectors. These are
asymptotic implementation costs inferred from the executed operations, not
runtime benchmarks. The solver timings reported later remain machine-specific
measurements and should not be interpreted as complexity measurements.

## 4. Experimental design

All cases use the normalized camera image, Pillow LANCZOS resizing, C-order
flattening, Gaussian kernel support 9, zero boundary with no edge
renormalization, noise standard deviation 0.01, and random seed 42. For a
given case, every iterative method uses the same operator and exact same
observation generated by that operator.

| Case | Shape | Operator | Sigma | Purpose |
|---|---:|---:|---:|---|
| resolution_32_sigma_1_5 | 32×32 | 1024×1024 | 1.5 | scalable resolution |
| resolution_64_sigma_1_5 | 64×64 | 4096×4096 | 1.5 | scalable resolution and tolerance study |
| resolution_128_sigma_1_5 | 128×128 | 16384×16384 | 1.5 | largest tested resolution |
| conditioning_64_sigma_0_6 | 64×64 | 4096×4096 | 0.6 | lower-blur convergence control |

Exact operator conditions from the one-dimensional factor spectra were
8.7327 for 64×64 at sigma 0.6, 16,967,992 for 32×32 at sigma 1.5,
56,635,597 for 64×64 at sigma 1.5, and 337,441,634 for 128×128 at sigma
1.5. Thus the 64×64 sigma 0.6 case is well-conditioned relative to the
sigma 1.5 cases, which are severely ill-conditioned. No full image-level
dense SVD was computed to obtain these values.

The 32×32, 64×64, and 128×128 image sizes extend beyond the previous
16×16 exact-SVD experiment. Iteration budgets are 8, 16, 32, and 64.
Tolerance experiments use 0.05 and 0.005, plus a strict 1e-12 small-system
validation. The scalable LSQR and damped LSQR budget runs use 1e-14 tolerances
and a 1e12 condition limit. Damped experiments use lambda=0.03.

The tolerance conventions differ by algorithm: LSQR uses atol and btol in
SciPy's compatible/incompatible residual tests, while CGLS uses a relative
normal-equation-residual stopping test. The tolerance plot labels the
requested value and does not imply these are identical tests.

## 5. Small-system numerical validation

Validation uses an 8×8, sigma 0.8, kernel 7, seeded noisy problem. This gives
an explicit 64×64 reference matrix within the existing dense safety limit.
LSQR and CGLS are compared with the Phase 2 QR least-squares solution.
Damped LSQR with lambda 0.05 is compared with the Phase 4 SVD Tikhonov
solution.

| Method | Requested tolerance | Iterations | Termination | Relative solution difference from reference |
|---|---:|---:|---|---:|
| LSQR | 0.05 | 2 | compatible residual tolerance | QR: 0.2889 |
| LSQR | 0.005 | 7 | compatible residual tolerance | QR: 0.2228 |
| LSQR | 1e-12 | 68 | compatible residual tolerance | QR: 1.68e-11 |
| CGLS | 1e-12 | 68 | normal-residual tolerance | QR: 3.84e-12 |
| damped LSQR | 0.05 | 2 | compatible residual tolerance | Tikhonov: 0.1909 |
| damped LSQR | 0.005 | 13 | least-squares normal-residual tolerance | Tikhonov: 0.02424 |
| damped LSQR | 1e-12 | 60 | least-squares normal-residual tolerance | Tikhonov: 1.32e-12 |

The tight-tolerance comparisons agree closely with the existing reference
implementations. Looser tolerances stop early and have visibly larger
solution differences, as expected. The damped comparison verifies that
lambda=0.05 yields the same lambda-squared penalty as Phase 4. Small
floating-point differences remain because the iterative and dense algorithms
follow different recurrences and stopping tests.

## 6. Scalability results

All four matrix-free cases ran, producing 88 experiment rows. Every operator
and observation consistency check passed. No full image matrix or full
image-level dense SVD was constructed for these cases; only the small
one-dimensional factors were decomposed for the exact structured condition
diagnostic.

For the sigma 1.5 cases at iteration limit 64:

| Image | Method | Relative error | Residual norm | Normal residual | Solver time (s) |
|---|---|---:|---:|---:|---:|
| 32×32 | LSQR | 0.3710 | 0.2444 | 0.00368 | 0.00221 |
| 32×32 | CGLS | 0.3702 | 0.2444 | 0.00434 | 0.00252 |
| 32×32 | damped LSQR | 0.1485 | 0.2605 | 0.000198 | 0.00223 |
| 64×64 | LSQR | 0.2941 | 0.5102 | 0.01019 | 0.00585 |
| 64×64 | CGLS | 0.2941 | 0.5102 | 0.01019 | 0.00699 |
| 64×64 | damped LSQR | 0.1362 | 0.5330 | 0.00124 | 0.00555 |
| 128×128 | LSQR | 0.2831 | 1.0403 | 0.02269 | 0.06796 |
| 128×128 | CGLS | 0.2831 | 1.0403 | 0.02269 | 0.12111 |
| 128×128 | damped LSQR | 0.1307 | 1.0829 | 0.00300 | 0.07878 |

These are timings for individual short solver runs on this machine, not
benchmarks. For the stronger blur sigma=1.5, all three methods reached the
64-iteration cap in these runs. The LSQR condition estimate is a Krylov
estimate over the explored subspace, not a computation of the full operator
condition number.

## 7. Convergence and regularization analysis

### Early stopping and semiconvergence

At 64×64, sigma 1.5, CGLS produced:

| Iterations | Relative error | Residual norm |
|---:|---:|---:|
| 8 | 0.1044 | 0.6267 |
| 16 | 0.1011 | 0.5637 |
| 32 | 0.1472 | 0.5351 |
| 64 | 0.2941 | 0.5102 |

The residual decreases throughout these checkpoints while reconstruction
error reaches its lowest listed value near iteration 16 and then increases.
This is direct evidence of semi-convergence and implicit regularization from
early stopping in this run. The stopping rule itself uses only the normal
residual, not ground truth.

At 128×128, sigma 1.5, CGLS relative errors at 8, 16, 32, and 64 iterations
were 0.09275, 0.09377, 0.14252, and 0.28307; residuals fell from 1.2369 to
1.0403. At iteration 64, damped LSQR had smaller reconstruction error
(0.13072) than unregularized LSQR (0.28307) despite a slightly larger
residual (1.08290 versus 1.04033). Lower residual is not equivalent to better
image reconstruction.

### Blur-conditioned convergence

At 64×64, sigma 0.6, the unregularized LSQR residual fell from 0.1397 at
iteration 8 to 3.64e-7 at iteration 64, and relative reconstruction error
was about 0.05972 at the final checkpoint. With sigma 1.5, the residual at
iteration 64 remained 0.5102 and relative error was 0.2941. This is
consistent with slower convergence and a harder inverse problem under stronger
blur. No exact full 64×64 operator condition number was computed; the
comparison is based on measured residual and reconstruction trajectories.

### Tolerance sensitivity

At 64×64, sigma 1.5:

| Method | Tolerance | Stopping iteration | Relative error | Residual norm |
|---|---:|---:|---:|---:|
| LSQR | 0.05 | 2 | 0.1572 | 1.7561 |
| LSQR | 0.005 | 10 | 0.1009 | 0.5977 |
| damped LSQR | 0.05 | 2 | 0.1573 | 1.7565 |
| damped LSQR | 0.005 | 20 | 0.1043 | 0.5559 |
| CGLS | 0.05 | 2 | 0.1572 | 1.7561 |
| CGLS | 0.005 | 6 | 0.1110 | 0.6912 |

These values show how practical residual-based stopping and normal-residual
stopping produce different iteration counts. Accuracy is evaluated only
afterwards and is not used to pick tolerances.

### Storage scaling

At 128×128, a full float64 image operator would require about 2.15 GB for its
entries alone. The two 128×128 separable factors require about 256 KiB of
float64 entries. These are theoretical data-array estimates, not measured
peak RAM; dense factorization workspace and language-object overhead are not
included.

## 8. Visualizations and generated outputs

All plots come from the recorded experiment data:

- results/plots/phase7_cgls_convergence_and_early_stopping.png
- results/plots/phase7_scalability_comparison.png
- results/plots/phase7_tolerance_sensitivity.png
- results/plots/phase7_representative_128_reconstructions.png

The representative panel contains the 128×128 original, blurred image, noisy
observation, and iterative reconstructions. Fixed display limits affect
visualization only; saved numerical reconstructions and metrics are raw.

## 9. Limitations and unresolved issues

- The maximum tested image was 128×128, not 256×256. The 128 case is already
  substantially beyond the old 16×16 SVD study, and this experiment kept
  runtime small and reproducible. Larger cases can be attempted with the same
  operator interface but were not measured here.
- The 64-iteration study does not establish full convergence for sigma 1.5.
  Increasing iterations can further reduce residual while worsening
  reconstruction accuracy.
- Only one image source, one noise realization, and one noise level are used
  in Phase 7.
- The 0.6 and 1.5 sigma comparison is a controlled lower/stronger blur pair,
  not an exhaustive conditioning study.
- LSQR/CGLS timing is affected by BLAS, CPU, and Python environment and is not
  a benchmark.
- Peak memory was not measured. Only explicitly labeled theoretical storage
  estimates are reported.
- CGLS operates on normal equations implicitly; although it avoids forming
  A.T A, its convergence can be sensitive to ill-conditioning and finite
  precision.

## 10. Tests and results

New tests in tests/test_iterative_solvers.py cover:

- LSQR and CGLS against a small explicit least-squares solution.
- Damped LSQR against both an explicit lambda-squared objective and the
  Phase 4 SVD Tikhonov implementation.
- Actual residual and normal-residual calculations.
- Forward/adjoint inner-product identity for rectangular and square images.
- Exact one-dimensional-factor condition formula against an explicit small
  Kronecker matrix.
- Iteration limits and LSQR checkpoint validation.
- Reproducible matrix-free observations and distinct C-order data flow.
- Invalid input and damping validation.
- Missing-adjoint and complex-observation rejection.
- Requirement for LinearOperator and absence of full-matrix methods in a
  32×32 path, with dense SVD patched to fail if called.

Commands:

    $env:MPLBACKEND = 'Agg'
    python -m pytest tests/test_iterative_solvers.py -q
    python -m pytest

Final test results: 13 focused iterative-solver tests passed and 151 tests
passed in the complete Phase 0–7 regression suite.

## 11. Reproduction commands

From the repository root, regenerate small-system validation, all scalable
cases, machine-readable results, raw arrays, logs, and plots:

    $env:MPLBACKEND = 'Agg'
    python experiments/phase7_scalable_iterative_methods.py

The experiment writes:

- results/metrics/phase7_small_system_validation.csv
- results/metrics/phase7_iterative_results.csv
- results/metrics/phase7_case_metadata.csv
- results/metrics/phase7_experiment_config.json
- results/logs/phase7_experiment.log
- 60 raw iterative reconstruction arrays and 16 raw original/blur/noise/
  observation arrays under results/images/phase7_*_raw.npy
- the four plots listed above.

Phase 5 and Phase 6 outputs are not overwritten.

## 12. Files created or modified

Created:

- src/numerical_methods/iterative_solvers.py
- src/utils/phase7_protocol.py
- experiments/phase7_scalable_iterative_methods.py
- tests/test_iterative_solvers.py
- report/phase7_scalable_iterative_methods.md

Modified:

- src/numerical_methods/__init__.py (exports iterative interfaces)
- README.md
- docs/development.md
- report/project_handoff_phase_0_phase_5.md

No Phase 0–6 numerical implementations, dependencies, configuration, or
generated result artifacts were modified. Phase 8 has not started.
