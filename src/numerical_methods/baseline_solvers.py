"""Unregularized dense baseline solvers for small image-deblurring systems."""

from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse


MAX_DENSE_PIXELS = 256
"""Maximum row or column count accepted by the educational dense solvers."""


@dataclass(frozen=True)
class SolverResult:
    """Raw reconstruction and diagnostics returned by a baseline solver."""

    solver_name: str
    reconstruction_vector: NDArray[np.float64]
    reconstruction_image: NDArray[np.float64]
    residual_vector: NDArray[np.float64]
    residual_norm: float
    relative_residual: float
    execution_time_seconds: float
    status: str
    diagnostic: str


def _validate_inputs(
    matrix: np.ndarray | sparse.spmatrix,
    observation: np.ndarray,
    image_shape: tuple[int, int],
    max_dense_pixels: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and materialize a safely sized dense linear system."""
    if not isinstance(max_dense_pixels, int) or isinstance(max_dense_pixels, bool):
        raise TypeError("max_dense_pixels must be an integer")
    if max_dense_pixels <= 0:
        raise ValueError("max_dense_pixels must be strictly positive")

    if sparse.issparse(matrix):
        if matrix.ndim != 2 or min(matrix.shape) <= 0:
            raise ValueError("matrix must be a non-empty two-dimensional array")
        rows, columns = matrix.shape
        if matrix.dtype.kind not in "biufc":
            raise TypeError("matrix must have a numerical dtype")
        if np.iscomplexobj(matrix.data):
            raise TypeError("complex-valued systems are not supported")
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("matrix must contain only finite values")
    else:
        candidate = np.asarray(matrix)
        if candidate.ndim != 2 or min(candidate.shape, default=0) <= 0:
            raise ValueError("matrix must be a non-empty two-dimensional array")
        rows, columns = candidate.shape
        if not np.issubdtype(candidate.dtype, np.number):
            raise TypeError("matrix must have a numerical dtype")
        if np.iscomplexobj(candidate):
            raise TypeError("complex-valued systems are not supported")
        if not np.all(np.isfinite(candidate)):
            raise ValueError("matrix must contain only finite values")

    if max(rows, columns) > max_dense_pixels:
        raise ValueError(
            "dense solver limit exceeded: matrix dimensions "
            f"{rows}x{columns} exceed the configured limit of "
            f"{max_dense_pixels}"
        )

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

    observation_array = np.asarray(observation)
    if observation_array.ndim != 1:
        raise ValueError("observation must be a one-dimensional vector")
    if observation_array.size != rows:
        raise ValueError("observation length must equal the matrix row count")
    if not np.issubdtype(observation_array.dtype, np.number):
        raise TypeError("observation must have a numerical dtype")
    if np.iscomplexobj(observation_array):
        raise TypeError("complex-valued systems are not supported")
    if not np.all(np.isfinite(observation_array)):
        raise ValueError("observation must contain only finite values")

    dense_matrix = (
        matrix.toarray()
        if sparse.issparse(matrix)
        else np.asarray(matrix, dtype=np.float64)
    )
    return dense_matrix.astype(np.float64, copy=False), observation_array.astype(
        np.float64,
        copy=False,
    )


def _build_result(
    solver_name: str,
    matrix: NDArray[np.float64],
    observation: NDArray[np.float64],
    reconstruction: NDArray[np.float64],
    image_shape: tuple[int, int],
    execution_time_seconds: float,
    diagnostic: str,
) -> SolverResult:
    if reconstruction.ndim != 1 or reconstruction.size != matrix.shape[1]:
        raise RuntimeError(f"{solver_name} returned an invalid solution shape")
    if not np.all(np.isfinite(reconstruction)):
        raise np.linalg.LinAlgError(
            f"{solver_name} produced a non-finite reconstruction"
        )

    residual = observation - matrix @ reconstruction
    if not np.all(np.isfinite(residual)):
        raise np.linalg.LinAlgError(f"{solver_name} produced a non-finite residual")
    residual_norm = float(np.linalg.norm(residual))
    observation_norm = float(np.linalg.norm(observation))
    relative_residual = (
        residual_norm / observation_norm
        if observation_norm > 0.0
        else (0.0 if residual_norm == 0.0 else float("inf"))
    )
    return SolverResult(
        solver_name=solver_name,
        reconstruction_vector=reconstruction.copy(),
        reconstruction_image=reconstruction.reshape(image_shape, order="C").copy(),
        residual_vector=residual,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        execution_time_seconds=execution_time_seconds,
        status="success",
        diagnostic=diagnostic,
    )


def solve_direct_inverse(
    matrix: np.ndarray | sparse.spmatrix,
    observation: np.ndarray,
    image_shape: tuple[int, int],
    *,
    max_dense_pixels: int = MAX_DENSE_PIXELS,
) -> SolverResult:
    """Solve ``A x = b`` by explicitly forming ``inv(A) @ b``.

    This is an educational unregularized baseline. Explicit inversion is
    generally slower, uses more memory, and is less numerically preferable than
    solving a factorized system directly.
    """
    dense_matrix, observation_vector = _validate_inputs(
        matrix,
        observation,
        image_shape,
        max_dense_pixels,
    )
    if dense_matrix.shape[0] != dense_matrix.shape[1]:
        raise ValueError("direct inversion requires a square matrix")

    start = perf_counter()
    try:
        inverse = linalg.inv(dense_matrix, check_finite=False)
        reconstruction = inverse @ observation_vector
    except (linalg.LinAlgError, ValueError) as error:
        raise np.linalg.LinAlgError(
            "direct inverse failed because the matrix is singular or unusable"
        ) from error
    elapsed = perf_counter() - start
    return _build_result(
        "direct_inverse",
        dense_matrix,
        observation_vector,
        reconstruction,
        image_shape,
        elapsed,
        "Unregularized explicit inverse: x = inv(A) @ b.",
    )


def solve_qr_least_squares(
    matrix: np.ndarray | sparse.spmatrix,
    observation: np.ndarray,
    image_shape: tuple[int, int],
    *,
    max_dense_pixels: int = MAX_DENSE_PIXELS,
) -> SolverResult:
    """Solve a full-column-rank least-squares problem via economic QR.

    Unlike direct inversion, QR supports overdetermined systems and avoids
    explicitly forming an inverse or the normal-equations matrix.
    """
    dense_matrix, observation_vector = _validate_inputs(
        matrix,
        observation,
        image_shape,
        max_dense_pixels,
    )
    rows, columns = dense_matrix.shape
    if rows < columns:
        raise ValueError("QR baseline requires rows >= columns")

    start = perf_counter()
    try:
        orthogonal, triangular = linalg.qr(
            dense_matrix,
            mode="economic",
            check_finite=False,
        )
        diagonal = np.abs(np.diag(triangular))
        scale = float(np.max(diagonal)) if diagonal.size else 0.0
        rank_tolerance = np.finfo(np.float64).eps * max(rows, columns) * scale
        if scale == 0.0 or np.any(diagonal <= rank_tolerance):
            raise np.linalg.LinAlgError(
                "QR factor is rank deficient or numerically unusable"
            )
        transformed = orthogonal.T @ observation_vector
        reconstruction = linalg.solve_triangular(
            triangular,
            transformed,
            lower=False,
            check_finite=False,
        )
    except (linalg.LinAlgError, ValueError) as error:
        raise np.linalg.LinAlgError(
            f"QR least-squares solve failed: {error}"
        ) from error
    elapsed = perf_counter() - start
    return _build_result(
        "qr_least_squares",
        dense_matrix,
        observation_vector,
        reconstruction,
        image_shape,
        elapsed,
        "Unregularized economic QR followed by triangular solution.",
    )


def solve_normal_equations(
    matrix: np.ndarray | sparse.spmatrix,
    observation: np.ndarray,
    image_shape: tuple[int, int],
    *,
    max_dense_pixels: int = MAX_DENSE_PIXELS,
) -> SolverResult:
    """Solve ``(A.T @ A) x = A.T @ b`` without fallback.

    For nonsingular ``A``, ``kappa_2(A.T @ A) = kappa_2(A) ** 2``. Squaring
    the conditioning can substantially amplify floating-point and data errors.
    No condition-number analysis is performed in this Phase 2 baseline.
    """
    dense_matrix, observation_vector = _validate_inputs(
        matrix,
        observation,
        image_shape,
        max_dense_pixels,
    )

    start = perf_counter()
    gram_matrix = dense_matrix.T @ dense_matrix
    right_hand_side = dense_matrix.T @ observation_vector
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", linalg.LinAlgWarning)
            reconstruction = linalg.solve(
                gram_matrix,
                right_hand_side,
                assume_a="sym",
                check_finite=False,
            )
    except (linalg.LinAlgError, linalg.LinAlgWarning, ValueError) as error:
        raise np.linalg.LinAlgError(
            "normal-equations solve failed because A.T @ A is singular or unusable"
        ) from error
    elapsed = perf_counter() - start
    return _build_result(
        "normal_equations",
        dense_matrix,
        observation_vector,
        reconstruction,
        image_shape,
        elapsed,
        "Unregularized solution of (A.T @ A) x = A.T @ b.",
    )
