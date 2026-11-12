"""Observation-only automatic parameter selection for TSVD and Tikhonov.

All criteria in this module use only ``A``, ``b`` and the supplied SVD. A
reference image is deliberately not accepted by the public selection APIs.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis.svd_analysis import SVDAnalysis


@dataclass(frozen=True)
class SelectionResult:
    """Selected candidate and complete criterion curve."""

    method: str
    criterion: str
    selected_parameter: float
    selected_index: int
    parameters: NDArray[np.float64]
    residual_norms: NDArray[np.float64]
    solution_norms: NDArray[np.float64]
    criterion_values: NDArray[np.float64]
    effective_degrees_of_freedom: NDArray[np.float64]
    diagnostic: str


def tsvd_candidate_ranks(svd: SVDAnalysis) -> NDArray[np.float64]:
    """Return every integer rank from one through numerical rank."""
    if svd.numerical_rank < 1:
        raise ValueError("SVD must have positive numerical rank")
    return np.arange(1, svd.numerical_rank + 1, dtype=np.float64)


def tikhonov_candidate_lambdas(
    svd: SVDAnalysis,
    *,
    count: int = 201,
    minimum_ratio: float = 1e-6,
    maximum_ratio: float = 10.0,
) -> NDArray[np.float64]:
    """Log grid spanning ``1e-6*smax`` through ``10*smax`` by default."""
    if count < 3:
        raise ValueError("count must be at least 3")
    if not np.isfinite(minimum_ratio) or minimum_ratio <= 0.0:
        raise ValueError("minimum_ratio must be finite and positive")
    if not np.isfinite(maximum_ratio) or maximum_ratio <= minimum_ratio:
        raise ValueError("maximum_ratio must be finite and exceed minimum_ratio")
    largest = float(svd.singular_values[0])
    if not np.isfinite(largest) or largest <= 0.0:
        raise ValueError("SVD must contain a positive leading singular value")
    return np.geomspace(largest * minimum_ratio, largest * maximum_ratio, count)


def discrete_lcurve_curvature(
    residual_norms: np.ndarray,
    solution_norms: np.ndarray,
    *,
    parameter_coordinates: np.ndarray | None = None,
) -> NDArray[np.float64]:
    """Estimate parametric curvature of the log-log L-curve.

    Coordinates are ``(log ||Ax-b||, log ||x||)`` parameterized by the
    candidate coordinate (log-lambda for Tikhonov, rank for TSVD by default).
    Nonuniform three-point derivatives approximate the parametric curvature
    ``|x'y''-y'x''|/(x'**2+y'**2)**(3/2)``. Endpoints and degenerate stencils
    are NaN because they cannot define a two-sided discrete curvature.
    """
    residuals = np.asarray(residual_norms, dtype=np.float64)
    solutions = np.asarray(solution_norms, dtype=np.float64)
    if residuals.ndim != 1 or solutions.shape != residuals.shape:
        raise ValueError("norm arrays must be one-dimensional and equal length")
    size = residuals.size
    curvature = np.full(size, np.nan, dtype=np.float64)
    if parameter_coordinates is None:
        coordinates = np.arange(size, dtype=np.float64)
    else:
        coordinates = np.asarray(parameter_coordinates, dtype=np.float64)
        if coordinates.shape != residuals.shape:
            raise ValueError("parameter_coordinates must match norm arrays")
    valid = (
        np.isfinite(residuals) & np.isfinite(solutions)
        & (residuals > 0.0) & (solutions > 0.0) & np.isfinite(coordinates)
    )
    if size < 3:
        return curvature
    lx = np.full(size, np.nan, dtype=np.float64)
    ly = np.full(size, np.nan, dtype=np.float64)
    lx[valid] = np.log(residuals[valid])
    ly[valid] = np.log(solutions[valid])
    for i in range(1, size - 1):
        if not (valid[i - 1] and valid[i] and valid[i + 1]):
            continue
        h0 = coordinates[i] - coordinates[i - 1]
        h1 = coordinates[i + 1] - coordinates[i]
        if h0 == 0.0 or h1 == 0.0 or not (np.isfinite(h0) and np.isfinite(h1)):
            continue
        if h0 + h1 == 0.0:
            continue
        if (
            (lx[i] == lx[i - 1] and ly[i] == ly[i - 1])
            or (lx[i + 1] == lx[i] and ly[i + 1] == ly[i])
        ):
            continue
        dx0, dx1 = (lx[i] - lx[i - 1]) / h0, (lx[i + 1] - lx[i]) / h1
        dy0, dy1 = (ly[i] - ly[i - 1]) / h0, (ly[i + 1] - ly[i]) / h1
        first_x, first_y = (h1 * dx0 + h0 * dx1) / (h0 + h1), (h1 * dy0 + h0 * dy1) / (h0 + h1)
        second_x, second_y = 2.0 * (dx1 - dx0) / (h0 + h1), 2.0 * (dy1 - dy0) / (h0 + h1)
        speed_sq = first_x**2 + first_y**2
        if np.isfinite(speed_sq) and speed_sq > np.finfo(float).tiny:
            value = abs(first_x * second_y - first_y * second_x) / speed_sq**1.5
            if np.isfinite(value):
                curvature[i] = value
    return curvature


def _prepare(observation: np.ndarray, svd: SVDAnalysis, parameters: np.ndarray):
    b = np.asarray(observation, dtype=np.float64)
    p = np.asarray(parameters, dtype=np.float64)
    if b.ndim != 1 or b.size != svd.left_singular_vectors.shape[0]:
        raise ValueError("observation must match SVD row dimension")
    if not np.all(np.isfinite(b)):
        raise ValueError("observation must be finite")
    if p.ndim != 1 or p.size < 1 or not np.all(np.isfinite(p)):
        raise ValueError("parameters must be a non-empty finite vector")
    if np.any(np.diff(p) <= 0):
        raise ValueError("parameters must be strictly increasing")
    return b, p


def _build_result(
    method, criterion, parameters, residuals, solutions, values, dof, diagnostic,
    *, maximize=False,
):
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        raise ValueError(f"no finite candidate scores for {criterion}")
    objective_index = np.argmax if maximize else np.argmin
    selected = int(finite_indices[objective_index(values[finite_indices])])
    return SelectionResult(
        method=method, criterion=criterion, selected_parameter=float(parameters[selected]),
        selected_index=selected, parameters=parameters.copy(), residual_norms=residuals,
        solution_norms=solutions, criterion_values=values,
        effective_degrees_of_freedom=dof,
        diagnostic=f"{diagnostic} Selected candidate index {selected}.",
    )


def _selection_curves(method, observation, svd, parameters, ranks=None):
    b, p = _prepare(observation, svd, parameters)
    u, s, vt = svd.left_singular_vectors, svd.singular_values, svd.right_singular_vectors_transpose
    coeff = u.T @ b
    m = b.size
    if method == "TSVD":
        ks = p.astype(int) if ranks is None else ranks
        if np.any(ks < 1) or np.any(ks > svd.numerical_rank) or np.any(ks != p):
            raise ValueError("TSVD parameters must be integer ranks within numerical rank")
        residuals, solutions, dofs = [], [], []
        for k in ks:
            xcoeff = np.zeros_like(coeff)
            xcoeff[:k] = coeff[:k] / s[:k]
            fitted = u[:, :k] @ coeff[:k]
            residuals.append(float(np.linalg.norm(fitted - b)))
            solutions.append(float(np.linalg.norm(vt.T @ xcoeff)))
            dofs.append(float(k))
        return b, p, np.array(residuals), np.array(solutions), np.array(dofs)
    if method != "Tikhonov":
        raise ValueError("method must be TSVD or Tikhonov")
    if np.any(p <= 0.0):
        raise ValueError("Tikhonov lambda candidates must be positive")
    residuals, solutions, dofs = [], [], []
    for lam in p:
        filt = s**2 / (s**2 + lam**2)
        xcoeff = (s / (s**2 + lam**2)) * coeff
        fitted = u @ (filt * coeff)
        residuals.append(float(np.linalg.norm(fitted - b)))
        solutions.append(float(np.linalg.norm(vt.T @ xcoeff)))
        dofs.append(float(np.sum(filt)))
    return b, p, np.array(residuals), np.array(solutions), np.array(dofs)


def select_parameter(
    method: str,
    criterion: str,
    matrix: np.ndarray,
    observation: np.ndarray,
    svd: SVDAnalysis,
    parameters: np.ndarray,
    *,
    noise_standard_deviation: float | None = None,
    discrepancy_factor: float = 1.05,
) -> SelectionResult:
    """Select TSVD rank or Tikhonov lambda without ground-truth information."""
    if getattr(matrix, "shape", None) != svd.matrix_shape:
        raise ValueError("matrix shape does not match supplied SVD")
    if criterion not in {"l_curve", "gcv", "discrepancy"}:
        raise ValueError("criterion must be l_curve, gcv, or discrepancy")
    b, p, residual, solution, dof = _selection_curves(method, observation, svd, parameters)
    if criterion == "l_curve":
        coords = np.log(p) if method == "Tikhonov" else p
        scores = discrete_lcurve_curvature(residual, solution, parameter_coordinates=coords)
        # Max curvature, with undefined endpoints/degenerate neighborhoods excluded.
        return _build_result(
            method, criterion, p, residual, solution, scores, dof,
            "Selected maximum finite discrete log-log L-curve curvature.",
            maximize=True,
        )
    if criterion == "gcv":
        denominator = 1.0 - dof / b.size
        scores = np.full_like(residual, np.inf)
        valid = denominator > np.sqrt(np.finfo(float).eps)
        scores[valid] = (residual[valid] ** 2 / b.size) / denominator[valid] ** 2
        return _build_result(method, criterion, p, residual, solution, scores, dof, "Selected minimum GCV using effective degrees of freedom.")
    if noise_standard_deviation is None or not np.isfinite(noise_standard_deviation) or noise_standard_deviation < 0:
        raise ValueError("discrepancy selection requires finite non-negative noise_standard_deviation")
    if not np.isfinite(discrepancy_factor) or discrepancy_factor <= 0:
        raise ValueError("discrepancy_factor must be finite and positive")
    target = discrepancy_factor * noise_standard_deviation * np.sqrt(b.size)
    scores = np.abs(residual - target)
    outside = target < np.min(residual) or target > np.max(residual)
    note = "Target lies outside candidate residual range; nearest endpoint is selected." if outside else "Selected candidate nearest the discrepancy target."
    return _build_result(method, criterion, p, residual, solution, scores, dof, note)

