# Development guide

## Environment setup

Use Python 3.11 or later and create an isolated environment from the repository
root. The available development environment uses Python 3.13.

```powershell
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`py -3` selects an installed Python 3 interpreter. Confirm that each version
command reports Python 3.11 or newer before installing dependencies.

## Running tests

Run the test suite with:

```powershell
python -m pytest
```

`pytest.ini` adds `src` to the test import path, so modules can be imported by
their package name during local development and testing.

## Project conventions

- Keep reusable Python code in `src` and tests in `tests`.
- Place notebooks or reproducible experiment scripts in `experiments`.
- Keep generated outputs in the appropriate `results` subdirectory; those
  outputs are excluded from version control except for directory markers.
- Use the centralized settings in `src/config.py` instead of embedding common
  paths or numerical defaults in future modules.
- Follow PEP 8 and add tests alongside new behavior.

## Current scope

Phases 0–9 are complete. Phase 8 added integrated evidence controls and Phase
9 added final research consolidation. Existing Phase 2–8 experiment artifacts
are frozen; do not rerun or overwrite them during documentation work.

Phase 2 dense methods are restricted to at most 256 matrix rows or columns.
Run the shared 12-by-12 experiment with:

```powershell
python experiments/phase2_baseline_reconstruction.py
```

Phase 3 full dense SVD is limited to matrix dimensions no greater than 256.
Larger separable blur spectra should use the structured pairwise-product
analysis. Phase 3 experiments use and report a fixed relative rank tolerance of
`1e-12`.

Phase 4 experiments use a 12-by-12 image, sigma 1.5, kernel size 9, noise
standard deviation 0.01, and seed 42. They use the same observation across
methods. Run them from the repository root:

```powershell
python experiments/phase4_tsvd_experiment.py
python experiments/phase4_tikhonov_experiment.py
python experiments/phase4_regularization_comparison.py
```

Phase 5 selectors use one shared observation and select TSVD rank or
Tikhonov lambda using L-curve curvature, GCV, or the discrepancy principle.
They do not receive the original image; ground truth is accessed only for
postselection evaluation. The default TSVD candidates are every integer rank
from 1 to numerical rank. Tikhonov candidates are 201 log-spaced lambdas from
`1e-6*s_max` to `10*s_max`. To regenerate Phase 5 artifacts:

```powershell
python experiments/phase5_parameter_selection.py
```

The discrepancy target is `factor * noise_std * sqrt(number_of_observations)`;
the experiment uses factor 1.05. If the target lies outside the residual range
of the candidate grid, the closest candidate endpoint is selected and noted in
the selector diagnostics. See `report/project_handoff_phase_0_phase_5.md` for
method definitions and artifact inventory.

Phase 6 studies five noise seeds at the shared 12x12 reference configuration,
three blur strengths, and 6x6, 12x12, and 16x16 resolutions. All configurations
stay within the exact 256-dimension SVD limit. To regenerate its independent
outputs:

```powershell
python experiments/phase6_robustness_sensitivity.py
```

Phase 6 reports sample standard deviations when there are repeated
observations; single blur-strength and resolution cases have an undefined
sample standard deviation. See
report/phase6_robustness_sensitivity_analysis.md.

Phase 7 provides matrix-free LSQR, CGLS, and damped LSQR. Large cases use the
existing SciPy LinearOperator forward and adjoint actions and do not construct
a full image operator or compute its dense SVD. Damped LSQR uses lambda as the
damping value, giving objective norm(Ax-b)^2 + lambda^2 norm(x)^2. To reproduce
small-system reference validation and the scalable experiments:

```powershell
python experiments/phase7_scalable_iterative_methods.py
```

Full derivations, matrix-free limits, results, and remaining questions are in
report/phase7_scalable_iterative_methods.md.

For saved-evidence Phase 8 analysis, list stages with:

```powershell
python experiments/phase8_integrated_analysis.py --list-stages
```
