"""Shared validation and residual utilities for dense regularization methods."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from analysis.svd_analysis import MAX_FULL_SVD_DIMENSION


@dataclass(frozen=True)
class ValidatedSystem:
    """Validated dense system and C-order image shape."""

    matrix: NDArray[np.float64]
    observation: NDArray[np.float64]
    image_shape: tuple[int, int]


def validate_regularization_system(
    matrix: np.ndarray | sparse.spmatrix,
    observation: np.ndarray,
    image_shape: tuple[int, int],
    maximum_dimension: int = MAX_FULL_SVD_DIMENSION,
) -> ValidatedSystem:
    """Validate and safely materialize a small real-valued linear system."""
    if not isinstance(maximum_dimension, int) or isinstance(maximum_dimension, bool):
        raise TypeError("maximum_dimension must be an integer")
    if maximum_dimension <= 0:
        raise ValueError("maximum_dimension must be strictly positive")

    if sparse.issparse(matrix):
        if matrix.ndim != 2 or min(matrix.shape) <= 0:
            raise ValueError("matrix must be non-empty and two-dimensional")
        rows, columns = matrix.shape
        if max(rows, columns) > maximum_dimension:
            raise ValueError(
                "regularization dimension limit exceeded before sparse "
                f"conversion: {matrix.shape} exceeds {maximum_dimension}"
            )
        if not np.issubdtype(matrix.dtype, np.number):
            raise TypeError("matrix must have a numerical dtype")
        if np.iscomplexobj(matrix.data):
            raise TypeError("complex-valued systems are not supported")
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("matrix must contain only finite values")
        dense = matrix.toarray()
    else:
        candidate = np.asarray(matrix)
        if candidate.ndim != 2 or candidate.size == 0:
            raise ValueError("matrix must be non-empty and two-dimensional")
        rows, columns = candidate.shape
        if max(rows, columns) > maximum_dimension:
            raise ValueError(
                f"regularization dimension limit exceeded: {candidate.shape} "
                f"exceeds {maximum_dimension}"
            )
        if not np.issubdtype(candidate.dtype, np.number):
            raise TypeError("matrix must have a numerical dtype")
        if np.iscomplexobj(candidate):
            raise TypeError("complex-valued systems are not supported")
        if not np.all(np.isfinite(candidate)):
            raise ValueError("matrix must contain only finite values")
        dense = candidate

    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 2
        or not all(isinstance(value, int) for value in image_shape)
        or any(value <= 0 for value in image_shape)
    ):
        raise ValueError("image_shape must contain two positive integers")
    if int(np.prod(image_shape)) != columns:
        raise ValueError(
            "the product of image_shape must equal the matrix column count"
        )

    observed = np.asarray(observation)
    if observed.ndim != 1:
        raise ValueError("observation must be a one-dimensional vector")
    if observed.size != rows:
        raise ValueError("observation length must equal the matrix row count")
    if not np.issubdtype(observed.dtype, np.number):
        raise TypeError("observation must have a numerical dtype")
    if np.iscomplexobj(observed):
        raise TypeError("complex-valued systems are not supported")
    if not np.all(np.isfinite(observed)):
        raise ValueError("observation must contain only finite values")

    return ValidatedSystem(
        matrix=np.asarray(dense, dtype=np.float64),
        observation=observed.astype(np.float64, copy=False),
        image_shape=image_shape,
    )


def residual_diagnostics(
    matrix: NDArray[np.float64],
    reconstruction: NDArray[np.float64],
    observation: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float]:
    """Return ``A @ x - b``, its norm, and its relative norm."""
    residual = matrix @ reconstruction - observation
    if not np.all(np.isfinite(residual)):
        raise np.linalg.LinAlgError("regularized solution produced a non-finite residual")
    residual_norm = float(np.linalg.norm(residual))
    observation_norm = float(np.linalg.norm(observation))
    relative = (
        residual_norm / observation_norm
        if observation_norm > 0.0
        else (0.0 if residual_norm == 0.0 else float("inf"))
    )
    return residual, residual_norm, relative
