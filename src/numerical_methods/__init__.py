"""Baseline and matrix-free iterative numerical reconstruction methods."""

from numerical_methods.baseline_solvers import (
    MAX_DENSE_PIXELS,
    SolverResult,
    solve_direct_inverse,
    solve_normal_equations,
    solve_qr_least_squares,
)
from numerical_methods.iterative_solvers import (
    IterationDiagnostic,
    IterativeResult,
    solve_cgls,
    solve_lsqr,
    solve_lsqr_checkpoints,
)

__all__ = [
    "MAX_DENSE_PIXELS",
    "SolverResult",
    "solve_direct_inverse",
    "solve_normal_equations",
    "solve_qr_least_squares",
    "IterationDiagnostic",
    "IterativeResult",
    "solve_cgls",
    "solve_lsqr",
    "solve_lsqr_checkpoints",
]
