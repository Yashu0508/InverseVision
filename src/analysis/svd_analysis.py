"""Dense and separable-structure SVD analysis for small blur operators."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse


MAX_FULL_SVD_DIMENSION = 256


@dataclass(frozen=True)
class RankAnalysis:
    """Numerical rank under the threshold ``rtol * largest_singular_value``."""

    numerical_rank: int
    total_singular_values: int
    relative_tolerance: float
    absolute_threshold: float


@dataclass(frozen=True)
class SVDAnalysis:
    """Full economy-size singular value decomposition and diagnostics."""

    left_singular_vectors: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    right_singular_vectors_transpose: NDArray[np.float64]
    matrix_shape: tuple[int, int]
    numerical_rank: int
    condition_number: float
    relative_tolerance: float
    rank_threshold: float
    largest_singular_value: float
    smallest_singular_value: float
    diagnostic: str


@dataclass(frozen=True)
class StructuredSVDAnalysis:
    """Singular-value diagnostics obtained from two Kronecker factors."""

    vertical_singular_values: NDArray[np.float64]
    horizontal_singular_values: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    vertical_shape: tuple[int, int]
    horizontal_shape: tuple[int, int]
    operator_shape: tuple[int, int]
    numerical_rank: int
    condition_number: float
    relative_tolerance: float
    rank_threshold: float
    largest_singular_value: float
    smallest_singular_value: float
    diagnostic: str


@dataclass(frozen=True)
class SingularValueDecay:
    """Normalized spectrum and cumulative squared singular-value energy."""

    normalized_singular_values: NDArray[np.float64]
    cumulative_energy: NDArray[np.float64]


def _validate_relative_tolerance(relative_tolerance: float | None) -> None:
    if relative_tolerance is None:
        return
    if (
        isinstance(relative_tolerance, (bool, np.bool_))
        or not np.isscalar(relative_tolerance)
        or not np.isfinite(relative_tolerance)
        or relative_tolerance < 0.0
    ):
        raise ValueError("relative_tolerance must be finite and non-negative")


def _resolve_relative_tolerance(
    relative_tolerance: float | None,
    matrix_shape: tuple[int, int],
) -> float:
    _validate_relative_tolerance(relative_tolerance)
    if relative_tolerance is None:
        return float(max(matrix_shape) * np.finfo(np.float64).eps)
    return float(relative_tolerance)


def _validated_dense_matrix(
    matrix: np.ndarray | sparse.spmatrix,
    *,
    maximum_dimension: int,
) -> NDArray[np.float64]:
    if not isinstance(maximum_dimension, int) or isinstance(maximum_dimension, bool):
        raise TypeError("maximum_dimension must be an integer")
    if maximum_dimension <= 0:
        raise ValueError("maximum_dimension must be strictly positive")

    if sparse.issparse(matrix):
        if matrix.ndim != 2 or min(matrix.shape) <= 0:
            raise ValueError("matrix must be non-empty and two-dimensional")
        if max(matrix.shape) > maximum_dimension:
            raise ValueError(
                "full SVD dimension limit exceeded before sparse conversion: "
                f"{matrix.shape} exceeds {maximum_dimension}"
            )
        if not np.issubdtype(matrix.dtype, np.number):
            raise TypeError("matrix must have a numerical dtype")
        if np.iscomplexobj(matrix.data):
            raise TypeError("complex matrices are not supported")
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("matrix must contain only finite values")
        dense = matrix.toarray()
    else:
        candidate = np.asarray(matrix)
        if candidate.ndim != 2 or candidate.size == 0:
            raise ValueError("matrix must be non-empty and two-dimensional")
        if max(candidate.shape) > maximum_dimension:
            raise ValueError(
                f"full SVD dimension limit exceeded: {candidate.shape} "
                f"exceeds {maximum_dimension}"
            )
        if not np.issubdtype(candidate.dtype, np.number):
            raise TypeError("matrix must have a numerical dtype")
        if np.iscomplexobj(candidate):
            raise TypeError("complex matrices are not supported")
        if not np.all(np.isfinite(candidate)):
            raise ValueError("matrix must contain only finite values")
        dense = candidate
    return np.asarray(dense, dtype=np.float64)


def numerical_rank_analysis(
    singular_values: np.ndarray,
    relative_tolerance: float,
) -> RankAnalysis:
    """Calculate numerical rank without altering the supplied spectrum."""
    _validate_relative_tolerance(relative_tolerance)
    values = np.asarray(singular_values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("singular_values must be a non-empty vector")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError("singular_values must be real numerical values")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("singular_values must be finite and non-negative")
    largest = float(np.max(values))
    threshold = float(relative_tolerance) * largest
    return RankAnalysis(
        numerical_rank=int(np.count_nonzero(values > threshold)),
        total_singular_values=int(values.size),
        relative_tolerance=float(relative_tolerance),
        absolute_threshold=threshold,
    )


def _condition_number(
    singular_values: NDArray[np.float64],
    rank: RankAnalysis,
) -> float:
    if rank.numerical_rank < rank.total_singular_values:
        return float("inf")
    smallest = float(singular_values[-1])
    if smallest == 0.0:
        return float("inf")
    return float(singular_values[0] / smallest)


def analyze_svd(
    matrix: np.ndarray | sparse.spmatrix,
    *,
    relative_tolerance: float | None = None,
    maximum_dimension: int = MAX_FULL_SVD_DIMENSION,
) -> SVDAnalysis:
    """Compute a full economy SVD and tolerance-aware rank diagnostics.

    No singular values are discarded. If ``relative_tolerance`` is omitted,
    ``max(matrix.shape) * machine_epsilon`` is used and reported.
    """
    dense = _validated_dense_matrix(matrix, maximum_dimension=maximum_dimension)
    resolved_tolerance = _resolve_relative_tolerance(
        relative_tolerance,
        dense.shape,
    )
    try:
        left, singular_values, right_transpose = linalg.svd(
            dense,
            full_matrices=False,
            check_finite=False,
            lapack_driver="gesdd",
        )
    except (linalg.LinAlgError, ValueError) as error:
        raise np.linalg.LinAlgError(f"SVD computation failed: {error}") from error
    if not (
        np.all(np.isfinite(left))
        and np.all(np.isfinite(singular_values))
        and np.all(np.isfinite(right_transpose))
    ):
        raise np.linalg.LinAlgError("SVD produced non-finite values")

    rank = numerical_rank_analysis(singular_values, resolved_tolerance)
    condition_number = _condition_number(singular_values, rank)
    diagnostic = (
        "Numerically rank deficient under the reported threshold."
        if np.isinf(condition_number)
        else "Numerically full rank under the reported threshold."
    )
    return SVDAnalysis(
        left_singular_vectors=left,
        singular_values=singular_values,
        right_singular_vectors_transpose=right_transpose,
        matrix_shape=dense.shape,
        numerical_rank=rank.numerical_rank,
        condition_number=condition_number,
        relative_tolerance=resolved_tolerance,
        rank_threshold=rank.absolute_threshold,
        largest_singular_value=float(singular_values[0]),
        smallest_singular_value=float(singular_values[-1]),
        diagnostic=diagnostic,
    )


def analyze_structured_svd(
    vertical_matrix: np.ndarray,
    horizontal_matrix: np.ndarray,
    *,
    relative_tolerance: float | None = None,
) -> StructuredSVDAnalysis:
    """Analyze ``vertical_matrix kron horizontal_matrix`` without forming it.

    The singular values of a Kronecker product are every pairwise product of
    the factors' singular values. Rectangular factors are supported.
    """
    vertical = _validated_dense_matrix(
        vertical_matrix,
        maximum_dimension=MAX_FULL_SVD_DIMENSION,
    )
    horizontal = _validated_dense_matrix(
        horizontal_matrix,
        maximum_dimension=MAX_FULL_SVD_DIMENSION,
    )
    operator_shape = (
        vertical.shape[0] * horizontal.shape[0],
        vertical.shape[1] * horizontal.shape[1],
    )
    resolved_tolerance = _resolve_relative_tolerance(
        relative_tolerance,
        operator_shape,
    )
    try:
        vertical_values = linalg.svdvals(vertical, check_finite=False)
        horizontal_values = linalg.svdvals(horizontal, check_finite=False)
    except (linalg.LinAlgError, ValueError) as error:
        raise np.linalg.LinAlgError(
            f"structured factor SVD failed: {error}"
        ) from error
    singular_values = np.multiply.outer(
        vertical_values,
        horizontal_values,
    ).ravel()
    singular_values.sort()
    singular_values = singular_values[::-1].copy()
    rank = numerical_rank_analysis(singular_values, resolved_tolerance)
    condition_number = _condition_number(singular_values, rank)
    diagnostic = (
        "Structured spectrum is numerically rank deficient under the "
        "reported threshold."
        if np.isinf(condition_number)
        else "Structured spectrum is numerically full rank under the "
        "reported threshold."
    )
    return StructuredSVDAnalysis(
        vertical_singular_values=vertical_values,
        horizontal_singular_values=horizontal_values,
        singular_values=singular_values,
        vertical_shape=vertical.shape,
        horizontal_shape=horizontal.shape,
        operator_shape=operator_shape,
        numerical_rank=rank.numerical_rank,
        condition_number=condition_number,
        relative_tolerance=resolved_tolerance,
        rank_threshold=rank.absolute_threshold,
        largest_singular_value=float(singular_values[0]),
        smallest_singular_value=float(singular_values[-1]),
        diagnostic=diagnostic,
    )


def singular_value_decay(singular_values: np.ndarray) -> SingularValueDecay:
    """Return normalized singular values and cumulative squared energy."""
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("singular_values must be a non-empty vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("singular_values must be finite and non-negative")
    largest = float(values[0])
    normalized = (
        values / largest if largest > 0.0 else np.zeros_like(values)
    )
    squared = values**2
    total_energy = float(np.sum(squared))
    cumulative = (
        np.cumsum(squared) / total_energy
        if total_energy > 0.0
        else np.zeros_like(values)
    )
    return SingularValueDecay(normalized, cumulative)
