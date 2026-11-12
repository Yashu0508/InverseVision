# Project Handoff Report: Phases 0–5

**Project:** Numerical Stability and Regularization in Image Deblurring  
**Status:** Phases 0–7 complete; Phase 8 has not started. The project remains
under development. Phase 5 rank-142 GCV was investigated and reproduced; it is
a legitimate failure mode for that noise realization, not an implementation
defect. Phase 6 robustness findings are documented in
`report/phase6_robustness_sensitivity_analysis.md`.  
**Purpose of this report:** concise context for the next AI or researcher
continuing the repository.

## Research aim

Study numerical stability and regularization for the ill-posed inverse problem

\[
b = A x + \epsilon,
\]

where `x` is the unknown sharp grayscale image, `A` is the Gaussian blur
operator, `b` is the observed image vector, and `epsilon` is measurement noise.
The project compares data fit, reconstruction quality, and sensitivity to
ill-conditioning while preserving reproducible experiments. Vectors use
C-order image flattening.

## Completed phases

- **Phase 0 — Project initialization:** modular Python package layout,
  dependencies, centralized defaults, development configuration, documentation,
  and basic tests.
- **Phase 1 — Degradation and forward model:** validated image preparation,
  Gaussian kernels and zero-boundary blur operators, additive Gaussian noise,
  and reproducible image degradation.
- **Phase 2 — Baseline reconstruction:** small dense direct inverse, QR
  least-squares, and normal-equations methods, metrics, plots, and comparisons.
- **Phase 3 — SVD and conditioning:** rank/spectrum diagnostics, structured
  Kronecker spectrum analysis, Picard/noise-amplification analysis, and blur
  sensitivity. This phase did not add a new reconstruction method.
- **Phase 4 — Regularization methods:** explicit-rank TSVD and standard-form
  L2 Tikhonov (SVD and augmented least-squares formulations), filters,
  diagnostics, experiments, and comparison artifacts.
- **Phase 5 — Automatic parameter selection:** observation-only L-curve,
  GCV, and discrepancy-principle selectors for TSVD and Tikhonov; candidate
  curves, selected reconstructions, evaluation, tests, and documentation.
- **Phase 6 — Robustness and sensitivity:** tested five noise seeds, three
  blur strengths, and 6×6, 12×12, and 16×16 resolutions. Reused the six Phase 5
  selectors and three Phase 2 baselines with shared per-case operators and
  observations. Added descriptive aggregation and reproducible outputs.
- **Phase 7 — Scalable iterative methods:** implemented matrix-free LSQR,
  CGLS, and damped LSQR; validated against small dense references and ran
  scalable experiments through 128×128 without full image matrices or dense
  SVDs. See report/phase7_scalable_iterative_methods.md.

Phase 7 is complete; Phase 8 has not started. The Phase 6 and Phase 7 reports
contain detailed experimental results, tests, and limitations.

## Phase 5 implementation

Reusable implementation: `src/regularization/parameter_selection.py`.
Experiment: `experiments/phase5_parameter_selection.py`.

The public selector accepts the method, criterion, forward matrix, observation,
SVD analysis, candidate vector, and (for discrepancy) noise level and factor.
It has no reference-image argument. Therefore ground truth does not enter any
selection score. The experiment completes all six selections before it reads
the original image for postselection metrics.

### Criteria

- **L-curve:** forms candidate pairs
  `(log(||A xhat - b||_2), log(||xhat||_2))`. Discrete parametric curvature is
  estimated from nonuniform three-point first and second derivatives, using
  `|x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)`. Tikhonov is parameterized by
  `log(lambda)`; TSVD is parameterized by rank. Endpoints and non-finite,
  duplicate-coordinate, zero-norm, or degenerate stencils are undefined
  (`NaN`) and excluded. The maximum finite curvature is selected. A wholly
  degenerate curve raises a clear `ValueError` rather than inventing a corner.
- **GCV:** minimizes
  `(||A xhat - b||_2^2 / m) / (1 - df/m)^2`. For TSVD, `df=k`, the trace of
  the rank-k truncated projection. For standard L2 Tikhonov,
  `df=sum_i sigma_i^2/(sigma_i^2+lambda^2)`, the trace of its smoothing
  operator. Candidates with a numerically zero denominator are infinite / not
  eligible for selection.
- **Discrepancy principle:** minimizes the absolute distance between the
  residual norm and `tau * noise_std * sqrt(m)`. The experiment uses
  `tau=1.05`. If the target falls outside the candidate residual range, the
  nearest candidate endpoint is selected and the diagnostic states this.
  There is no interpolation between candidates.

### Candidate grids

- TSVD: every integer rank from 1 through the Phase 4 numerical rank (144 in
  the shared experiment).
- Tikhonov: 201 logarithmically spaced positive values from
  `1e-6*s_max` through `10*s_max`, inclusive. This provides weak- and
  strong-regularization coverage without using ground truth.

## Shared Phase 4/5 protocol

Both methods and all criteria use exactly the same forward operator and noisy
observation.

| Setting | Value |
|---|---:|
| Image | scikit-image camera, resized to 12×12 and normalized |
| Blur | Gaussian, sigma 1.5, kernel size 9 |
| Boundary | zero padding; no boundary renormalization |
| Noise standard deviation | 0.01 |
| Random seed | 42 |
| Rank relative tolerance | 1e-12 |
| Discrepancy factor | 1.05 |

The experiment uses the Phase 4 setup in `src/utils/phase4_protocol.py` and
`src/utils/experiment_setup.py`. It retains the dense safety limit of 256
dimensions from prior SVD/regularization code.

## Phase 5 outcome

Selected values and postselection metrics are in
`results/metrics/phase5_selection_summary.csv`. These quality metrics are for
interpretation only; they were computed after the parameters had been fixed by
each criterion.

| Method | Criterion | Selected value | Relative error | Residual norm |
|---|---|---:|---:|---:|
| TSVD | L-curve | k = 24 | 0.2020 | 0.10257 |
| TSVD | GCV | k = 142 | 17317.14 | 0.001379 |
| TSVD | discrepancy | k = 18 | 0.2070 | 0.11980 |
| Tikhonov | L-curve | lambda = 0.01111 | 0.2964 | 0.07173 |
| Tikhonov | GCV | lambda = 0.02487 | 0.1858 | 0.07755 |
| Tikhonov | discrepancy | lambda = 0.09030 | 0.1824 | 0.12518 |

TSVD GCV selected a near-full rank and was catastrophically sensitive to the
small singular directions for this small ill-conditioned example. This is a
reproduced, verified outcome: the analytical rank-142 GCV score is
`6.841112977e-05`; an explicit rank-142 smoother gives `6.841112986e-05`, and
the actual solver residual agrees with the selection residual within
`7.60e-13`. The `k=142` noise coefficient is about `0.00818` at singular value
`7.00e-08`, so inverse filtering amplifies it to about `116,805`. The rank is
valid under the `1e-12` numerical-rank tolerance, but reconstruction relative
error rises to about `17,317`. GCV is minimizing its observation-based
prediction-risk proxy, not ground-truth error. This is a demonstrated
limitation for this case, not evidence that GCV always behaves this way.
L-curve selection depends on candidate discretization and finite-difference
curvature; discrepancy selection depends on the assumed noise standard
deviation and factor. Full hypothesis-by-hypothesis evidence is in the root
cause report.

## Phase 5 files and outputs

New/updated source and test files:

- `src/regularization/parameter_selection.py` — candidate grids, curvature,
  GCV and discrepancy selection.
- `src/regularization/__init__.py` — exports the selection APIs.
- `experiments/phase5_parameter_selection.py` — reproducible experiment and
  artifacts.
- `tests/test_parameter_selection.py` — focused mathematical and edge tests.
- `README.md`, `docs/development.md` — status and run instructions.
- `report/project_handoff_phase_0_phase_5.md` — this handoff.

Generated artifacts:

- `results/metrics/phase5_selection_curves.csv` — all candidates, residuals,
  solution norms, degrees of freedom, GCV numerator, unsquared denominator,
  and criterion curves.
- `results/metrics/phase5_selection_summary.csv` — selected parameter and
  postselection evaluation metrics.
- `results/plots/phase5_parameter_selection.png` — L-curves, GCV, and
  discrepancy panels.
- `results/plots/phase5_selected_reconstructions.png` — six selected raw
  estimates rendered for comparison.
- Six `results/images/phase5_*_raw.npy` arrays — raw, unclipped selected
  reconstructions, one per method/criterion.

Regenerate artifacts from the repository root:

```powershell
$env:MPLBACKEND = 'Agg'
python experiments/phase5_parameter_selection.py
```

## Validation

Focused validation (including explicit TSVD smoother and shared-protocol
rank-142 regression checks): **13 passed**.

```powershell
$env:MPLBACKEND = 'Agg'
python -m pytest tests/test_parameter_selection.py -q
```

Full regression validation: **132 passed**.

```powershell
$env:MPLBACKEND = 'Agg'
python -m pytest
```

The existing `tests/conftest.py` selects Matplotlib Agg for headless tests.
Phase 5 was added without changing the Phase 0–4 solver implementations.

## Continuation notes

- Preserve the shared experiment protocol when comparing later methods.
- Selection code must not receive ground truth; compute accuracy metrics only
  after selection.
- Keep generated numerical arrays raw and unclipped; display-only clipping or
  fixed color limits must not modify them.
- Phase 0–5 are the completed scope. Ask the project owner before starting
  subsequent work if no new phase specification is supplied.
