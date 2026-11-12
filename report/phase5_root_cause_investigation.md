# Phase 5 Root-Cause Investigation: TSVD GCV Rank 142

**Project:** Numerical Stability and Regularization in Image Deblurring  
**Investigation status:** completed for the reproducible Phase 5 protocol  
**Verdict:** rank 142 is a legitimate GCV selection for this realization, not
a defect in the GCV or TSVD implementation. The resulting very poor image
accuracy is caused by noise amplification in an ill-conditioned singular
direction. No Phase 0–4 numerical algorithm was changed.  
**Workspace note:** the project directory has no `.git` metadata. Both
`git status --short` and `git rev-parse --show-toplevel` returned “not a git
repository”, so a Git change list could not be obtained. Files were inspected
and edited in place; none were reset or reverted.

## 1. Original problem statement

The Phase 5 experiment selected TSVD rank `k=142` with generalized
cross-validation (GCV), then reported relative reconstruction error around
17,317 despite a residual norm around 0.00138. The investigation was asked to
distinguish implementation mistakes from a real failure mode of the selection
criterion, trace the complete forward and inverse data path, reproduce the
numbers independently, fix any confirmed defects, strengthen tests, rerun the
experiment, and preserve the evidence.

## 2. Mathematical understanding

For an SVD `A = U diag(sigma) V.T`, the rank-`k` TSVD estimate is

\[
\hat x_k=V_k\,\mathrm{diag}(1/\sigma_1,\ldots,1/\sigma_k)\,U_k^T b.
\]

Its fitted observation is `H_k b`, where `H_k = U_k U_k.T` for the retained
left singular subspace. The effective degrees of freedom are therefore
`trace(H_k)=k`. With `m` observations, the GCV score used here is

\[
G(k)=\frac{\|H_kb-b\|_2^2/m}{(1-k/m)^2}.
\]

Increasing `k` can only reduce the exact nested-subspace residual (apart from
roundoff), but it includes more inverse factors `1/sigma_i`. A residual is a
data-fit measure: it does not bound `||xhat-x||`. In the noise component,
the inverse coefficient is `(u_i.T epsilon)/sigma_i`, so very small singular
values can turn modest observation noise into enormous image-space error.

GCV minimizes a prediction-risk proxy without access to the true image. It
does not minimize the realized ground-truth error. In severely ill-conditioned
problems, a particular noise realization can have a large coefficient along a
small-singular-value vector; fitting that coefficient sharply reduces the
residual while strongly increasing reconstruction error. This is the
mathematically expected risk that the rank-142 result raised.

## 3. Investigated hypotheses

### A. GCV mathematics

| Hypothesis | Investigation and evidence | Conclusion |
|---|---|---|
| Numerator is wrong or missing a square/normalization | Checked the implemented `residual**2 / m` against explicit smoother matrices and direct solver residuals. At `k=142`, numerator is `1.319659139e-08`; direct residual-based numerator is `1.319659140e-08`. | Rejected |
| Denominator is wrong or missing a square | Code computes `1 - df/m`, then divides by its square. With `df=142`, `m=144`, denominator before squaring is `0.01388888889`, denominator squared `0.0001929012346`. | Rejected |
| TSVD degrees of freedom are wrong | The TSVD smoother is the rank-`k` orthogonal projector `U_k U_k.T`; its trace is `k`. Explicit trace is `141.999999999991` for `k=142`. | Rejected |
| Observation dimension `m` is wrong | `b` has 144 entries and `A` is 144×144; selector uses `b.size`, i.e. 144. | Rejected |
| Zero or near-zero denominator is mishandled | `k=144` has denominator zero and is set to infinite score / excluded. `k=143` has denominator `1/144`, a valid score. No near-zero nonzero denominator occurs in this grid. | Rejected for this run; guard behaves as intended |

### B. TSVD mathematics

| Hypothesis | Investigation and evidence | Conclusion |
|---|---|---|
| Truncated-SVD reconstruction formula is wrong | Compared solver output with `V_k @ ((U_k.T @ b)/s_k)` and the explicit rank-k smoothing matrix. The solver uses this formula. | Rejected |
| Singular values or vectors are misordered/misaligned | `scipy.linalg.svd(..., lapack_driver="gesdd")` returns descending singular values; observed spectrum is nonincreasing. U, values, and V-transpose are returned together by that same factorization. | Rejected |
| Rank indexing is off by one | Candidate rank `142` maps to index 141 in the zero-based candidate arrays and retains exactly the first 142 singular triplets. `selected_parameter=142`, `selected_index=141`. | Rejected |
| Rank 142 includes numerically unusable directions | Numerical rank is 144 at threshold `8.926935573e-13`. Singular value #142 is `7.003492003e-08`, about `7.85e4` times the threshold. It is numerically above the configured threshold, though highly noise-sensitive. | Rejected as rank-policy error; instability is real |

### C. Residual calculation

| Hypothesis | Investigation and evidence | Conclusion |
|---|---|---|
| Selection residual is not the residual of the reconstructed estimate | In the SVD basis, `A xhat_k=U_k U_k.T b`; independently solved `xhat` was passed through the actual sparse `A`. | Rejected |
| Selection and solver use different `A` or `b` | Both receive `problem.operator`, `problem.observation`, and `problem.svd_analysis` from one `create_phase4_problem()` result. | Rejected |
| SVD residual disagrees materially with forward residual | At rank 142, analytical/projection residual `0.001378517014907`; actual `||A xhat-b||` `0.001378517015666`; absolute difference `7.60e-13`. | Rejected as cause |
| Floating-point cancellation explains the huge error | At rank 142, forward residual agrees to about `5.5e-10` relative. At full rank, projection and forward residual differ by about `1.39e-11` because a nearly exact fit is evaluated through an ill-conditioned inverse. | Rejected as rank-142 cause; small full-rank cancellation is present but does not cause the selection |

### D. Forward model and SVD

| Hypothesis | Investigation and evidence | Conclusion |
|---|---|---|
| C-order vectorization is inconsistent | The explicit operator documents `(B_y kron B_x) @ vec_C(X)`. The experiment compares `A @ original.ravel(C)` with the separable degraded image; discrepancy is `7.16e-16`. | Rejected |
| Separable blur and sparse forward operator differ | Both use the same one-dimensional zero-boundary matrices and `B_y @ X @ B_x.T`. The generated observation equals `degradation.noisy_observation.ravel(C)` exactly. | Rejected |
| Boundary conditions differ | Both constructions use zero padding with no boundary-row renormalization; the same Gaussian matrix constructor is used. | Rejected |
| SVD spectrum is inconsistent with structured Kronecker spectrum | Structured pairwise-product spectrum has rank 144, maximum absolute singular-value difference `8.88e-16`; maximum relative discrepancy over tiny tail singular values `2.98e-10`. Condition numbers: explicit `56,767,069.9861`, structured `56,767,070.0030`. | Rejected |

### E. Candidate grid

| Hypothesis | Investigation and evidence | Conclusion |
|---|---|---|
| Rank 142 is outside the valid candidate range | TSVD candidates are every integer rank 1 through numerical rank 144. | Rejected |
| Candidates are shifted or evaluated inconsistently | Candidate index 141 stores rank 142, dof 142, its own residual, norm, and score; explicit rank-142 matrix reproduces it. | Rejected |
| Tikhonov grid causes TSVD selection behavior | TSVD uses a separate integer-rank grid; no lambda candidate enters TSVD GCV. | Rejected |

### F. Data and evaluation

| Hypothesis | Investigation and evidence | Conclusion |
|---|---|---|
| Methods use different observations or noise is nonreproducible | `create_phase4_problem()` creates one observation using seed 42; repeated construction is array-identical. All six selectors receive that same observation. | Rejected |
| Ground truth leaks into parameter selection | `select_parameter` has no reference-image parameter. Experiment completes six calls before accessing `problem.original` for metrics. | Rejected |
| Display transformations affect metrics | Solver arrays are saved raw. `evaluate_reconstruction` receives raw arrays; `imshow(vmin=0, vmax=1)` is only used when drawing plots. | Rejected |
| The evaluation reference is inconsistent | Original is normalized after resize, and the same array is used to form the degraded image and later postselection metrics. | Rejected |

### G. L-curve and discrepancy principle

| Hypothesis | Investigation and evidence | Conclusion |
|---|---|---|
| L-curve ordering/parameterization changes GCV | L-curve and GCV are separate branches over the same candidate arrays. Tikhonov uses increasing lambda and log-lambda parameter coordinate; TSVD uses increasing rank. | Rejected as GCV cause |
| Curvature endpoints or degeneracies silently influence the selected point | Endpoints, zero/nonfinite norms, duplicate points/coordinates, and degenerate stencils are excluded as NaN; selection takes maximum finite curvature or errors when none exist. | Rejected as GCV cause; finite-difference curvature remains grid-dependent |
| Discrepancy target is malformed | Target is `tau * noise_std * sqrt(m)`; values are `1.05*0.01*sqrt(144)=0.126`. For both methods it lies within candidate residual ranges in the experiment. | Rejected |
| Out-of-range rule chooses an arbitrary interior value | Rule minimizes absolute distance to target; for these monotone residual curves, outside-range cases select the nearest residual endpoint. Actual experiment is bracketed. | Rejected |

## 4. Files and functions inspected

- `src/regularization/parameter_selection.py`: `tsvd_candidate_ranks`,
  `tikhonov_candidate_lambdas`, `discrete_lcurve_curvature`, `_prepare`,
  `_selection_curves`, `_build_result`, and `select_parameter`.
- `src/regularization/tsvd.py`: `tsvd_filter_factors` and `solve_tsvd`.
- `src/regularization/tikhonov.py`: `solve_tikhonov_svd`, filter factors,
  and the augmented formulation for context.
- `src/analysis/svd_analysis.py`: `_validated_dense_matrix`,
  `numerical_rank_analysis`, `analyze_svd`, and `analyze_structured_svd`.
- `src/forward_model/gaussian.py`: 1-D zero-boundary matrices and 2-D C-order
  Kronecker operator.
- `src/forward_model/operators.py`: separable `B_y @ X @ B_x.T` action.
- `src/forward_model/degradation.py`, `noise.py`: generation path,
  normalization assumptions, seed handling, and un-clipped additive noise.
- `src/utils/experiment_setup.py`, `phase4_protocol.py`: camera resize,
  common observation, SVD, and numerical protocol.
- `src/evaluation/reconstruction_metrics.py`: raw-array evaluation and
  forward-model residual calculation.
- `experiments/phase5_parameter_selection.py`: orchestration, selection before
  ground-truth access, solver calls, metric calls, CSV and plot generation.
- `tests/test_parameter_selection.py` and
  `report/project_handoff_phase_0_phase_5.md`.

## 5. Execution path and data flow

1. `create_phase4_problem()` resizes and normalizes `skimage.data.camera()` to
   12×12, applies the separable Gaussian blur, and adds seeded Gaussian noise.
2. It independently builds the sparse 2-D operator from the same 1-D Gaussian
   matrices, flattens the noisy observation in C order, and analyzes that
   operator with SVD at `rtol=1e-12`.
3. The Phase 5 script builds ranks 1…144 and a 201-point log lambda grid.
4. Each selector gets method, criterion, the same sparse operator, same noisy
   observation, same SVD, candidates, and only discrepancy-specific noise
   metadata. No original image is passed.
5. After all six selections return, the script calls the existing solver for
   each selected value, then computes ground-truth metrics and writes raw
   arrays, CSVs, and plots.

## 6. Reproduction and numerical evidence

Reproduction command from the repository root:

```powershell
$env:MPLBACKEND = 'Agg'
python experiments/phase5_parameter_selection.py
```

Protocol and spectrum:

| Quantity | Reproduced value |
|---|---:|
| Image / operator / observation | 12×12 / 144×144 / 144 values |
| Gaussian blur | sigma 1.5, kernel 9, zero boundary |
| Noise std / seed | 0.01 / 42 |
| SVD tolerance | 1e-12 |
| Numerical rank | 144 |
| Largest singular value | 0.8926935573 |
| Smallest singular value | 1.572555282e-08 |
| Rank threshold | 8.926935573e-13 |
| Condition number | 56,767,069.9861 |
| Realized noise norm / std | 0.1034946040 / 0.0086114973 |

GCV candidate calculations near the selected point:

| k | residual | numerator `residual²/m` | `1-df/m` | GCV score | solution norm |
|---:|---:|---:|---:|---:|---:|
| 140 | 0.008371245559 | 4.866510570e-07 | 0.02777777778 | 6.306997698e-04 | 21,722.06 |
| 141 | 0.008295782541 | 4.779167220e-07 | 0.02083333333 | 1.101120128e-03 | 22,039.54 |
| 142 | 0.001378517015 | 1.319659139e-08 | 0.01388888889 | 6.841112977e-05 | 118,866.34 |
| 143 | 0.000937934112 | 6.109169434e-09 | 0.006944444444 | 1.266797374e-04 | 119,738.39 |
| 144 | 6.652684e-15 (SVD projection) | 3.073486608e-31 | 0 | infinite | 133,771.01 |

For `k=142`, independent checks gave:

- explicit rank-142 smoother trace: `141.999999999991`;
- explicit smoother residual: `0.001378517015747`;
- explicit-matrix GCV: `6.841112985672e-05`;
- selector GCV: `6.841112977397e-05`;
- actual TSVD solver residual `||A xhat-b||`: `0.001378517015666`;
- selector versus actual residual absolute difference: `7.60e-13`.

The explicit smoother was independently assembled as
`H_k = A @ V_k @ diag(1/sigma_k) @ U_k.T`, rather than reading the selector's
residual curve.

The selected rank-142 noise coefficient is particularly explanatory:

| component | value |
|---|---:|
| singular value `sigma_142` | 7.003492003e-08 |
| `u_142.T @ b` | 0.008180446125 |
| signal contribution `sigma_142 * v_142.T @ x_true` | -1.454253925e-09 |
| noise contribution `u_142.T @ epsilon` | 0.008180447580 |
| noise coefficient after inverse filtering | 116,805.27 |

Thus the rank-142 fitted component is almost entirely this realization's noise.
Its inclusion drops residual from `0.0082958` at rank 141 to `0.0013785`,
which beats the GCV degrees-of-freedom penalty. It also produces relative
ground-truth error `17,317.138` and solution norm about `118,866`. The smaller
residual is real; it is not a reconstruction-quality guarantee.

## 7. Rank-142 GCV result

The selected rank is a legitimate outcome of the implemented criterion under
the fixed candidate grid and observed noise realization. It is also a poor
reconstruction for this inverse problem. The basis coefficient at rank 142 is
noise-dominated, and inversion divides it by `7e-8`. This adds roughly
`1.17e5` in that image-space singular coordinate. The residual drops by a
factor of six from rank 141, while the degrees-of-freedom denominator shrinks
from `0.02083` to `0.01389`; the observed residual decrease dominates the
penalty and GCV is minimized at rank 142.

Ground-truth relative error was inspected only after the selection, as a
diagnostic. It did not enter the rank, GCV, or any other selector result.

## 8–9. Root causes and classification

**Confirmed implementation defect:** none found in the rank-142 selection
path.

**Mathematical limitation / numerical stability issue:** GCV selects a linear
smoother by an observation-based predictive-risk proxy. The realized noise in
this run has a large coefficient in a direction with very small singular
value. GCV is not an oracle for ground-truth error and does not reliably
regularize every severely ill-conditioned problem/noise realization. This is
the root cause of the surprising result.

**Harmless boundary behavior:** full rank `k=144` gives `df=m`, zero GCV
denominator, and is deliberately marked ineligible (`inf`). At that rank the
computed forward residual also shows a small cancellation-level difference
from the SVD projection residual; it is not involved in the selected rank.

**Unresolved issues:** none for reproducing this specific observation and
explaining rank 142. General GCV reliability across other blur strengths,
noise levels, image sizes, and noise realizations is not established by this
single controlled experiment.

## 10. Changes made

No solver, SVD, forward-model, metric, or selection formula was changed because
the audit found no defect to fix. Changes made to preserve diagnostic evidence
and strengthen regression coverage:

- `experiments/phase5_parameter_selection.py`: candidate curve CSV now also
  records the GCV numerator and unsquared denominator for every rank/lambda.
- `tests/test_parameter_selection.py`: added explicit TSVD smoother/GCV
  comparison, actual solver residual comparison, numerical-rank rejection,
  discrepancy bracketing, ground-truth API isolation, and fixed-protocol
  reproducibility/rank-142 checks.
- `report/phase5_root_cause_investigation.md`: this investigation record.
- `report/project_handoff_phase_0_phase_5.md`: updated verified status,
  diagnosis, new test coverage, and diagnostic CSV columns.

The existing `phase5_selection_curves.csv`, summary CSV, plots, and raw arrays
were regenerated after adding the diagnostic columns. Parameter selections
and postselection metrics are unchanged from the previously reported run;
there was no numerical algorithm fix, so no changed result is attributed to a
bug correction.

## 11. Tests added and purpose

- Explicitly construct the TSVD smoother for a small system and compare its
  trace, residual, and GCV with the analytical selector.
- Compare TSVD selection residual with `||A @ solver_result - b||`.
- Assert numerical-rank candidate generation excludes a direction below the
  tolerance and the existing solver rejects requests beyond numerical rank.
- Test discrepancy selection with an in-range target and existing tests cover
  an out-of-range target.
- Verify the public selection signature has no `reference_image` argument.
- Recreate the shared seeded problem twice, verify identical observations,
  check descending singular values and rank 144, reproduce rank 142, compare
  its residual with the solver, and verify the sparse operator maps the raw
  original image to the separable blur.
- Existing tests also cover explicit-matrix Tikhonov GCV, zero-denominator
  handling, L-curve endpoints/duplicates/degeneracy, and parameter-grid shape.

## 12. Test commands and results

Focused tests:

```powershell
$env:MPLBACKEND = 'Agg'
python -m pytest tests/test_parameter_selection.py -q
```

Full suite:

```powershell
$env:MPLBACKEND = 'Agg'
python -m pytest
```

Final observed results: **13 focused tests passed** and **132 total tests
passed** (full result recorded after the final run). Matplotlib Agg remains
configured through the existing test setup.

## 13–14. Before/after comparison

There was no algorithmic fix, so selectors and metrics are unchanged. Old and
regenerated results:

| Method / criterion | Prior selected value | Verified selected value | Prior relative error | Verified relative error |
|---|---:|---:|---:|---:|
| TSVD / L-curve | 24 | 24 | 0.2020048 | 0.2020048 |
| TSVD / GCV | 142 | 142 | 17,317.138 | 17,317.138 |
| TSVD / discrepancy | 18 | 18 | 0.2069533 | 0.2069533 |
| Tikhonov / L-curve | 0.01110970 | 0.01110970 | 0.2964065 | 0.2964065 |
| Tikhonov / GCV | 0.02487152 | 0.02487152 | 0.1857572 | 0.1857572 |
| Tikhonov / discrepancy | 0.09030305 | 0.09030305 | 0.1823646 | 0.1823646 |

The changes were diagnostic output and tests/documentation only. Exact
selection values and metrics did not change; regenerated floating-point values
match the recorded values at shown precision.

## 15. Remaining limitations

- This is one 12×12 image, one blur, and one noise realization; no claim of
  general GCV performance is made.
- GCV on TSVD can choose a high rank for a noise-dominated but strongly
  residual-reducing singular direction. Users should inspect the GCV curve,
  solution norm, spectrum, and alternative criteria together.
- GCV's saturated `df=m` endpoint is undefined; the implementation excludes
  it. Very near-saturated values can still be sensitive to the degrees of
  freedom penalty.
- L-curve corner location depends on the finite candidate grid and discrete
  curvature approximation. Discrepancy selection depends on the assumed noise
  standard deviation and factor.
- The environment has no Git metadata, limiting repository-native change
  tracking during this task.

## 16. Final technical verdict

The rank-142 result is **not an implementation bug**. GCV numerator,
denominator, effective degrees of freedom, observation size, SVD residual, and
the actual TSVD solver all agree with independent calculations. The rank is
valid under the stated numerical-rank tolerance. Forward operator, boundary
condition, C-order vectorization, seeded data, and reference-image evaluation
are consistent. The outcome is a genuine limitation of using GCV as a
ground-truth reconstruction selector for this severely ill-conditioned,
noise-dominated singular direction.

Phase 5 is now documented and tested with the anomaly reproduced and
explained. **Phase 6 has not started.**
