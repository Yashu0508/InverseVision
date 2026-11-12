"""Truncated singular value decomposition reconstruction."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from analysis.svd_analysis import SVDAnalysis, analyze_svd
from regularization._common import (
    residual_diagnostics,
    validate_regularization_system,
)


@dataclass(frozen=True)
class TSVDFilterFactors:
    """Inverse and dimensionless retention filters for one truncation rank."""

    inverse_filter: NDArray[np.float64]
    retention_filter: NDArray[np.float64]
    retained_rank: int


@dataclass(frozen=True)
class TSVDResult:
    """Raw TSVD reconstruction and data-fit diagnostics."""

    reconstruction_vector: NDArray[np.float64]
    reconstruction_image: NDArray[np.float64]
    retained_rank: int
    numerical_rank: int
    residual_vector: NDArray[np.float64]
    residual_norm: float
    relative_residual: float
    execution_time_seconds: float
    rank_threshold: float
    status: str
    diagnostic: str


def tsvd_filter_factors(
    singular_values: np.ndarray,
    retained_rank: int,
    *,
    denominator_threshold: float = 0.0,
) -> TSVDFilterFactors:
    """Return TSVD inverse and dimensionless retention filters.

    The inverse filter is ``1 / sigma_i`` for retained directions and zero
    otherwise. The dimensionless retention filter is one for retained
    directions and zero otherwise. No automatic truncation selection occurs.
    """
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("singular_values must be a non-empty vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("singular_values must be finite and non-negative")
    if not isinstance(retained_rank, int) or isinstance(retained_rank, bool):
        raise TypeError("retained_rank must be an integer")
    if retained_rank < 1 or retained_rank > values.size:
        raise ValueError(
            f"retained_rank must lie between 1 and {values.size}"
        )
    if not np.isfinite(denominator_threshold) or denominator_threshold < 0.0:
        raise ValueError("denominator_threshold must be finite and non-negative")
    if np.any(values[:retained_rank] <= denominator_threshold):
        raise ValueError(
            "retained_rank includes singular values classified as numerically zero"
        )

    inverse_filter = np.zeros_like(values)
    inverse_filter[:retained_rank] = 1.0 / values[:retained_rank]
    retention_filter = np.zeros_like(values)
    retention_filter[:retained_rank] = 1.0
    return TSVDFilterFactors(
        inverse_filter=inverse_filter,
        retention_filter=retention_filter,
        retained_rank=retained_rank,
    )


def solve_tsvd(
    matrix: np.ndarray | sparse.spmatrix,
    observation: np.ndarray,
    image_shape: tuple[int, int],
    retained_rank: int,
    *,
    relative_tolerance: float = 1e-12,
    svd_analysis: SVDAnalysis | None = None,
) -> TSVDResult:
    """Compute the user-requested TSVD reconstruction without clipping.

    If a compatible SVD is supplied, its factorization cost is not included in
    the reported filtering time. Requests beyond the numerical rank raise an
    error rather than being silently reduced.
    """
    system = validate_regularization_system(matrix, observation, image_shape)
    start = perf_counter()
    analysis = (
        analyze_svd(system.matrix, relative_tolerance=relative_tolerance)
        if svd_analysis is None
        else svd_analysis
    )
    if analysis.matrix_shape != system.matrix.shape:
        raise ValueError("supplied SVD shape does not match the matrix")
    if retained_rank > analysis.numerical_rank:
        raise ValueError(
            f"retained_rank {retained_rank} exceeds numerically usable rank "
            f"{analysis.numerical_rank}"
        )
    filters = tsvd_filter_factors(
        analysis.singular_values,
        retained_rank,
        denominator_threshold=analysis.rank_threshold,
    )
    coefficients = analysis.left_singular_vectors.T @ system.observation
    filtered_coefficients = filters.inverse_filter * coefficients
    reconstruction = (
        analysis.right_singular_vectors_transpose.T @ filtered_coefficients
    )
    elapsed = perf_counter() - start
    if not np.all(np.isfinite(reconstruction)):
        raise np.linalg.LinAlgError("TSVD produced a non-finite reconstruction")
    residual, residual_norm, relative_residual = residual_diagnostics(
        system.matrix,
        reconstruction,
        system.observation,
    )
    return TSVDResult(
        reconstruction_vector=reconstruction.copy(),
        reconstruction_image=reconstruction.reshape(image_shape, order="C").copy(),
        retained_rank=retained_rank,
        numerical_rank=analysis.numerical_rank,
        residual_vector=residual,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        execution_time_seconds=elapsed,
        rank_threshold=analysis.rank_threshold,
        status="success",
        diagnostic=(
            "TSVD retained the explicitly requested leading singular "
            f"components: k={retained_rank}."
        ),
    )
