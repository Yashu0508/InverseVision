"""Matrix-free separable Gaussian blur operations."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from config import DEFAULT_CONFIG
from forward_model.gaussian import gaussian_blur_matrix_1d


def _validated_image(image: np.ndarray) -> NDArray[np.float64]:
    array = np.asarray(image)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("image must be a non-empty two-dimensional array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image must have a numerical dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError("image must contain only finite values")
    return array.astype(np.float64, copy=False)


def _validated_matrix(
    matrix: np.ndarray,
    expected_shape: tuple[int, int],
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def apply_separable_gaussian_blur(
    image: np.ndarray,
    sigma: float = DEFAULT_CONFIG.gaussian_blur_sigma,
    kernel_size: int = DEFAULT_CONFIG.gaussian_kernel_size,
    *,
    vertical_matrix: np.ndarray | None = None,
    horizontal_matrix: np.ndarray | None = None,
) -> NDArray[np.float64]:
    """Apply zero-boundary Gaussian blur as ``B_y @ X @ B_x.T``.

    The full two-dimensional Kronecker matrix is never constructed. Optional
    one-dimensional matrices permit an already constructed operator to be
    reused; when omitted, both are generated from ``sigma`` and
    ``kernel_size``.
    """
    array = _validated_image(image)
    rows, columns = array.shape

    vertical = (
        gaussian_blur_matrix_1d(rows, sigma, kernel_size)
        if vertical_matrix is None
        else _validated_matrix(vertical_matrix, (rows, rows), "vertical_matrix")
    )
    horizontal = (
        gaussian_blur_matrix_1d(columns, sigma, kernel_size)
        if horizontal_matrix is None
        else _validated_matrix(
            horizontal_matrix,
            (columns, columns),
            "horizontal_matrix",
        )
    )
    return vertical @ array @ horizontal.T


def gaussian_blur_linear_operator(
    image_shape: tuple[int, int],
    sigma: float = DEFAULT_CONFIG.gaussian_blur_sigma,
    kernel_size: int = DEFAULT_CONFIG.gaussian_kernel_size,
) -> LinearOperator:
    """Return a matrix-free C-order Gaussian blur ``LinearOperator``.

    ``matvec`` applies ``B_y @ X @ B_x.T``. ``rmatvec`` implements the exact
    transpose action ``B_y.T @ X @ B_x`` rather than assuming symmetry.
    """
    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 2
        or not all(isinstance(value, int) for value in image_shape)
        or any(value <= 0 for value in image_shape)
    ):
        raise ValueError("image_shape must contain two positive integers")

    rows, columns = image_shape
    vertical = gaussian_blur_matrix_1d(rows, sigma, kernel_size)
    horizontal = gaussian_blur_matrix_1d(columns, sigma, kernel_size)
    vector_size = rows * columns

    def action(
        vector: np.ndarray,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        values = np.asarray(vector, dtype=np.float64)
        if values.size != vector_size:
            raise ValueError(f"vector must contain {vector_size} elements")
        image = values.reshape(image_shape, order="C")
        return (left @ image @ right).ravel(order="C")

    matvec: Callable[[np.ndarray], NDArray[np.float64]] = lambda vector: action(
        vector,
        vertical,
        horizontal.T,
    )
    rmatvec: Callable[[np.ndarray], NDArray[np.float64]] = lambda vector: action(
        vector,
        vertical.T,
        horizontal,
    )
    return LinearOperator(
        shape=(vector_size, vector_size),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
    )
