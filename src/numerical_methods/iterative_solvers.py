"""Matrix-free LSQR and CGLS solvers for small and large inverse problems."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, lsqr


@dataclass(frozen=True)
class IterationDiagnostic:
    """Data-fit and optimality diagnostics after an iterative step."""

    iteration: int
    residual_norm: float
    normal_equation_residual_norm: float
    reconstruction_norm: float


@dataclass(frozen=True)
class IterativeResult:
    """Raw vector solution and termination information for one iterative run."""

    method: str
    reconstruction_vector: NDArray[np.float64]
    residual_vector: NDArray[np.float64]
    residual_norm: float
    relative_residual: float
    normal_equation_residual_norm: float
    reconstruction_norm: float
    iteration_count: int
    iteration_limit: int
    relative_tolerance: float
    absolute_tolerance: float
    damping_parameter: float
    termination_code: int
    termination_status: str
    execution_time_seconds: float
    condition_estimate: float
    diagnostic: str
    history: tuple[IterationDiagnostic, ...]
    reconstruction_history: tuple[NDArray[np.float64], ...]


_LSQR_TERMINATION = {
    0: "initial solution is exact",
    1: "compatible residual tolerance reached",
    2: "least-squares normal-residual tolerance reached",
    3: "condition limit reached",
    4: "compatible residual tolerance reached at machine precision",
    5: "least-squares normal-residual tolerance reached at machine precision",
    6: "condition limit reached at machine precision",
    7: "iteration limit reached",
}


def _validated_problem(
    operator: LinearOperator,
    observation: np.ndarray,
    iteration_limit: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[NDArray[np.float64], int, int]:
    if not isinstance(operator, LinearOperator):
        raise TypeError("iterative solvers require a scipy LinearOperator")
    if len(operator.shape) != 2 or min(operator.shape) <= 0:
        raise ValueError("operator must have positive two-dimensional shape")
    rows, columns = operator.shape
    raw_observation = np.asarray(observation)
    if raw_observation.ndim != 1 or raw_observation.size != rows:
        raise ValueError("observation must be a vector matching operator rows")
    if not np.issubdtype(raw_observation.dtype, np.number):
        raise TypeError("observation must have a numeric dtype")
    if np.iscomplexobj(raw_observation):
        raise TypeError("complex observations are not supported")
    b = raw_observation.astype(np.float64, copy=False)
    if not np.all(np.isfinite(b)):
        raise ValueError("observation must contain only finite values")
    if (
        not isinstance(iteration_limit, int)
        or isinstance(iteration_limit, bool)
        or iteration_limit < 1
    ):
        raise ValueError("iteration_limit must be a positive integer")
    for name, value in (
        ("relative_tolerance", relative_tolerance),
        ("absolute_tolerance", absolute_tolerance),
    ):
        if not np.isscalar(value) or not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    try:
        adjoint_probe = np.asarray(
            operator.rmatvec(np.zeros(rows, dtype=np.float64)),
            dtype=np.float64,
        )
    except (NotImplementedError, TypeError, ValueError) as error:
        raise ValueError("operator must provide a valid adjoint rmatvec") from error
    if adjoint_probe.shape != (columns,) or not np.all(np.isfinite(adjoint_probe)):
        raise ValueError("operator adjoint must return a finite column-sized vector")
    return b, rows, columns


def solve_cgls(
    operator: LinearOperator,
    observation: np.ndarray,
    *,
    iteration_limit: int = 100,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 0.0,
) -> IterativeResult:
    """Solve least squares by CGLS using only matvec and adjoint rmatvec.

    The solution starts at zero. Stopping is based on the normal-equation
    residual norm relative to its initial value. The stored residual and
    normal-residual histories allow early-stopping regularization to be
    studied without using ground truth in the stopping decision.
    """
    b, rows, columns = _validated_problem(
        operator, observation, iteration_limit,
        relative_tolerance, absolute_tolerance,
    )
    start = perf_counter()
    x = np.zeros(columns, dtype=np.float64)
    residual = b.copy()
    gradient = operator.rmatvec(residual)
    direction = gradient.copy()
    gamma = float(np.dot(gradient, gradient))
    initial_normal_norm = float(np.sqrt(gamma))
    tolerance = absolute_tolerance + relative_tolerance * initial_normal_norm
    history: list[IterationDiagnostic] = [
        IterationDiagnostic(
            0, float(np.linalg.norm(residual)), initial_normal_norm, 0.0
        )
    ]
    reconstruction_history = [x.copy()]
    termination_code = 7
    status = "iteration limit reached"
    iterations = 0

    if initial_normal_norm <= tolerance:
        termination_code, status = 0, "initial solution meets normal-residual tolerance"
    else:
        for iteration in range(1, iteration_limit + 1):
            image_direction = operator.matvec(direction)
            denominator = float(np.dot(image_direction, image_direction))
            if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
                termination_code, status = 6, "CGLS breakdown: forward direction has zero norm"
                break
            alpha = gamma / denominator
            x += alpha * direction
            residual -= alpha * image_direction
            new_gradient = operator.rmatvec(residual)
            new_gamma = float(np.dot(new_gradient, new_gradient))
            normal_norm = float(np.sqrt(max(0.0, new_gamma)))
            iterations = iteration
            history.append(IterationDiagnostic(
                iteration,
                float(np.linalg.norm(residual)),
                normal_norm,
                float(np.linalg.norm(x)),
            ))
            reconstruction_history.append(x.copy())
            if not (
                np.all(np.isfinite(x))
                and np.all(np.isfinite(residual))
                and np.isfinite(normal_norm)
            ):
                termination_code, status = 6, "CGLS breakdown: non-finite iterate"
                break
            if normal_norm <= tolerance:
                termination_code, status = 2, "normal-residual tolerance reached"
                break
            if gamma <= np.finfo(float).tiny:
                termination_code, status = 6, "CGLS breakdown: zero search-gradient norm"
                break
            beta = new_gamma / gamma
            direction = new_gradient + beta * direction
            gradient, gamma = new_gradient, new_gamma

    final_residual = operator.matvec(x) - b
    final_normal = operator.rmatvec(final_residual)
    execution = perf_counter() - start
    residual_norm = float(np.linalg.norm(final_residual))
    b_norm = float(np.linalg.norm(b))
    final_normal_norm = float(np.linalg.norm(final_normal))
    return IterativeResult(
        method="CGLS",
        reconstruction_vector=x.copy(),
        residual_vector=final_residual.copy(),
        residual_norm=residual_norm,
        relative_residual=residual_norm / b_norm if b_norm else residual_norm,
        normal_equation_residual_norm=final_normal_norm,
        reconstruction_norm=float(np.linalg.norm(x)),
        iteration_count=iterations,
        iteration_limit=iteration_limit,
        relative_tolerance=float(relative_tolerance),
        absolute_tolerance=float(absolute_tolerance),
        damping_parameter=0.0,
        termination_code=termination_code,
        termination_status=status,
        execution_time_seconds=execution,
        condition_estimate=float("nan"),
        diagnostic=(
            "CGLS uses only LinearOperator matvec/rmatvec. Stopping checks "
            "the normal-equation residual against the initial residual."
        ),
        history=tuple(history),
        reconstruction_history=tuple(reconstruction_history),
    )


def solve_lsqr(
    operator: LinearOperator,
    observation: np.ndarray,
    *,
    iteration_limit: int = 100,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 1e-8,
    damping_parameter: float = 0.0,
    condition_limit: float = 1e8,
) -> IterativeResult:
    """Solve by SciPy LSQR; damping minimizes ||Ax-b||² + damp²||x||².

    The damping parameter uses the same lambda convention as the project's
    standard L2 Tikhonov solver. Specifically, damp=lambda, not lambda².
    SciPy's atol and btol control its compatible/incompatible residual tests;
    termination code and actual recomputed residuals are returned.
    """
    b, rows, _ = _validated_problem(
        operator, observation, iteration_limit,
        relative_tolerance, absolute_tolerance,
    )
    if (
        not np.isscalar(damping_parameter)
        or not np.isfinite(damping_parameter)
        or damping_parameter < 0.0
    ):
        raise ValueError("damping_parameter must be finite and non-negative")
    if not np.isscalar(condition_limit) or not np.isfinite(condition_limit) or condition_limit <= 0.0:
        raise ValueError("condition_limit must be finite and positive")
    start = perf_counter()
    output = lsqr(
        operator,
        b,
        damp=float(damping_parameter),
        atol=float(absolute_tolerance),
        btol=float(relative_tolerance),
        conlim=float(condition_limit),
        iter_lim=iteration_limit,
        show=False,
        calc_var=False,
    )
    x = np.asarray(output[0], dtype=np.float64)
    termination_code = int(output[1])
    iterations = int(output[2])
    residual = operator.matvec(x) - b
    optimality = operator.rmatvec(residual) + float(damping_parameter) ** 2 * x
    execution = perf_counter() - start
    residual_norm = float(np.linalg.norm(residual))
    b_norm = float(np.linalg.norm(b))
    return IterativeResult(
        method="damped_LSQR" if damping_parameter else "LSQR",
        reconstruction_vector=x.copy(),
        residual_vector=residual.copy(),
        residual_norm=residual_norm,
        relative_residual=residual_norm / b_norm if b_norm else residual_norm,
        normal_equation_residual_norm=float(np.linalg.norm(optimality)),
        reconstruction_norm=float(np.linalg.norm(x)),
        iteration_count=iterations,
        iteration_limit=iteration_limit,
        relative_tolerance=float(relative_tolerance),
        absolute_tolerance=float(absolute_tolerance),
        damping_parameter=float(damping_parameter),
        termination_code=termination_code,
        termination_status=_LSQR_TERMINATION.get(
            termination_code, "unknown LSQR termination code"
        ),
        execution_time_seconds=execution,
        condition_estimate=float(output[6]),
        diagnostic=(
            "SciPy LSQR; damping_parameter is lambda in the objective "
            "||Ax-b||^2 + lambda^2 ||x||^2."
        ),
        history=(IterationDiagnostic(
            iterations, residual_norm, float(np.linalg.norm(optimality)),
            float(np.linalg.norm(x)),
        ),),
        reconstruction_history=(x.copy(),),
    )


def solve_lsqr_checkpoints(
    operator: LinearOperator,
    observation: np.ndarray,
    iteration_limits: tuple[int, ...],
    *,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
    damping_parameter: float = 0.0,
    condition_limit: float = 1e8,
) -> tuple[IterativeResult, ...]:
    """Run independent zero-start LSQR solves at requested iteration budgets."""
    if not iteration_limits or any(
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        for limit in iteration_limits
    ):
        raise ValueError("iteration_limits must contain positive integers")
    if tuple(sorted(set(iteration_limits))) != iteration_limits:
        raise ValueError("iteration_limits must be unique and strictly increasing")
    return tuple(
        solve_lsqr(
            operator,
            observation,
            iteration_limit=limit,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            damping_parameter=damping_parameter,
            condition_limit=condition_limit,
        )
        for limit in iteration_limits
    )
