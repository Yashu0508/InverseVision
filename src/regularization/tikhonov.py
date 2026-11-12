"""Standard-form L2 Tikhonov reconstruction methods."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse

from analysis.svd_analysis import SVDAnalysis, analyze_svd
from regularization._common import (
    residual_diagnostics,
    validate_regularization_system,
)


@dataclass(frozen=True)
class TikhonovFilterFactors:
    """Inverse and dimensionless Tikhonov filters for one lambda value."""

    inverse_filter: NDArray[np.float64]
    retention_filter: NDArray[np.float64]
    regularization_parameter: float


@dataclass(frozen=True)
class TikhonovResult:
    """Raw Tikhonov reconstruction and data-fit diagnostics."""

    reconstruction_vector: NDArray[np.float64]
    reconstruction_image: NDArray[np.float64]
    regularization_parameter: float
    formulation: str
    residual_vector: NDArray[np.float64]
    residual_norm: float
    relative_residual: float
    execution_time_seconds: float
    status: str
    diagnostic: str


def _validated_lambda(regularization_parameter: float) -> float:
    if (
        isinstance(regularization_parameter, (bool, np.bool_))
        or not np.isscalar(regularization_parameter)
        or not np.isfinite(regularization_parameter)
        or regularization_parameter <= 0.0
    ):
        raise ValueError(
            "regularization_parameter must be finite and strictly positive"
        )
    return float(regularization_parameter)


def tikhonov_filter_factors(
    singular_values: np.ndarray,
    regularization_parameter: float,
) -> TikhonovFilterFactors:
    """Return inverse and dimensionless standard-form Tikhonov filters.

    The inverse filter is ``sigma_i / (sigma_i**2 + lambda**2)``. The
    dimensionless retention filter is
    ``sigma_i**2 / (sigma_i**2 + lambda**2)``.
    """
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("singular_values must be a non-empty vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("singular_values must be finite and non-negative")
    parameter = _validated_lambda(regularization_parameter)
    denominator = values**2 + parameter**2
    return TikhonovFilterFactors(
        inverse_filter=values / denominator,
        retention_filter=values**2 / denominator,
        regularization_parameter=parameter,
    )


def _build_result(
    reconstruction: NDArray[np.float64],
    matrix: NDArray[np.float64],
    observation: NDArray[np.float64],
    image_shape: tuple[int, int],
    parameter: float,
    formulation: str,
    elapsed: float,
) -> TikhonovResult:
    if not np.all(np.isfinite(reconstruction)):
        raise np.linalg.LinAlgError(
            "Tikhonov method produced a non-finite reconstruction"
        )
    residual, residual_norm, relative_residual = residual_diagnostics(
        matrix,
        reconstruction,
        observation,
    )
    return TikhonovResult(
        reconstruction_vector=reconstruction.copy(),
        reconstruction_image=reconstruction.reshape(image_shape, order="C").copy(),
        regularization_parameter=parameter,
        formulation=formulation,
        residual_vector=residual,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        execution_time_seconds=elapsed,
        status="success",
        diagnostic=(
            "Standard-form L2 Tikhonov regularization using the "
            f"{formulation} formulation with lambda={parameter:g}."
        ),
    )


def solve_tikhonov_svd(
    matrix: np.ndarray | sparse.spmatrix,
    observation: np.ndarray,
    image_shape: tuple[int, int],
    regularization_parameter: float,
    *,
    relative_tolerance: float = 1e-12,
    svd_analysis: SVDAnalysis | None = None,
) -> TikhonovResult:
    """Solve standard L2 Tikhonov using SVD filter factors."""
    parameter = _validated_lambda(regularization_parameter)
    system = validate_regularization_system(matrix, observation, image_shape)
    start = perf_counter()
    analysis = (
        analyze_svd(system.matrix, relative_tolerance=relative_tolerance)
        if svd_analysis is None
        else svd_analysis
    )
    if analysis.matrix_shape != system.matrix.shape:
        raise ValueError("supplied SVD shape does not match the matrix")
    filters = tikhonov_filter_factors(
        analysis.singular_values,
        parameter,
    )
    coefficients = analysis.left_singular_vectors.T @ system.observation
    reconstruction = analysis.right_singular_vectors_transpose.T @ (
        filters.inverse_filter * coefficients
    )
    elapsed = perf_counter() - start
    return _build_result(
        reconstruction,
        system.matrix,
        system.observation,
        image_shape,
        parameter,
        "SVD filter-factor",
        elapsed,
    )


def solve_tikhonov_augmented(
    matrix: np.ndarray | sparse.spmatrix,
    observation: np.ndarray,
    image_shape: tuple[int, int],
    regularization_parameter: float,
) -> TikhonovResult:
    """Solve standard L2 Tikhonov as an augmented least-squares problem.

    This avoids explicitly forming ``A.T @ A``. SciPy's least-squares routine
    operates on ``[A; lambda I]`` and ``[b; 0]`` using a robust dense driver.
    """
    parameter = _validated_lambda(regularization_parameter)
    system = validate_regularization_system(matrix, observation, image_shape)
    columns = system.matrix.shape[1]
    start = perf_counter()
    augmented_matrix = np.vstack(
        (system.matrix, parameter * np.eye(columns, dtype=np.float64))
    )
    augmented_observation = np.concatenate(
        (system.observation, np.zeros(columns, dtype=np.float64))
    )
    try:
        reconstruction, _, _, _ = linalg.lstsq(
            augmented_matrix,
            augmented_observation,
            check_finite=False,
            lapack_driver="gelsy",
        )
    except (linalg.LinAlgError, ValueError) as error:
        raise np.linalg.LinAlgError(
            f"augmented Tikhonov least-squares solve failed: {error}"
        ) from error
    elapsed = perf_counter() - start
    return _build_result(
        reconstruction,
        system.matrix,
        system.observation,
        image_shape,
        parameter,
        "augmented least-squares",
        elapsed,
    )


def tikhonov_objective(
    matrix: np.ndarray | sparse.spmatrix,
    observation: np.ndarray,
    reconstruction: np.ndarray,
    regularization_parameter: float,
) -> float:
    """Evaluate ``||A x-b||_2**2 + lambda**2 ||x||_2**2``."""
    parameter = _validated_lambda(regularization_parameter)
    estimate = np.asarray(reconstruction)
    observed = np.asarray(observation)
    if estimate.ndim != 1 or observed.ndim != 1:
        raise ValueError("observation and reconstruction must be vectors")
    if matrix.shape != (observed.size, estimate.size):
        raise ValueError("matrix shape is incompatible with the supplied vectors")
    if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(observed)):
        raise ValueError("vectors must contain only finite values")
    residual = matrix @ estimate - observed
    return float(
        np.vdot(residual, residual).real
        + parameter**2 * np.vdot(estimate, estimate).real
    )
