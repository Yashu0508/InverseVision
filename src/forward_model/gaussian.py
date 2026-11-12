"""Gaussian kernels and explicit zero-boundary blur matrices."""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from config import DEFAULT_CONFIG


def gaussian_kernel_1d(
    sigma: float = DEFAULT_CONFIG.gaussian_blur_sigma,
    kernel_size: int = DEFAULT_CONFIG.gaussian_kernel_size,
) -> NDArray[np.float64]:
    """Construct a normalized, centered, one-dimensional Gaussian kernel."""
    if not isinstance(kernel_size, int) or isinstance(kernel_size, bool):
        raise TypeError("kernel_size must be an integer")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be positive and odd")
    if (
        isinstance(sigma, (bool, np.bool_))
        or not np.isscalar(sigma)
        or not np.isfinite(sigma)
        or sigma <= 0.0
    ):
        raise ValueError("sigma must be a finite, strictly positive number")

    radius = kernel_size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * float(sigma) ** 2))
    kernel /= np.sum(kernel)
    return kernel


def gaussian_blur_matrix_1d(
    dimension: int,
    sigma: float = DEFAULT_CONFIG.gaussian_blur_sigma,
    kernel_size: int = DEFAULT_CONFIG.gaussian_kernel_size,
) -> NDArray[np.float64]:
    """Construct a dense 1-D zero-boundary Gaussian convolution matrix.

    For ``y = B @ x``, samples of ``x`` beyond ``[0, dimension)`` are treated
    as zero. Truncated kernels at boundaries are deliberately not renormalized,
    so boundary-row sums can be smaller than one. The implementation also
    supports dimensions smaller than the kernel.
    """
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError("dimension must be an integer")
    if dimension <= 0:
        raise ValueError("dimension must be strictly positive")

    kernel = gaussian_kernel_1d(sigma=sigma, kernel_size=kernel_size)
    radius = kernel_size // 2
    matrix = np.zeros((dimension, dimension), dtype=np.float64)

    for row in range(dimension):
        first_column = max(0, row - radius)
        last_column = min(dimension, row + radius + 1)
        columns = np.arange(first_column, last_column)
        matrix[row, columns] = kernel[radius + columns - row]

    return matrix


def gaussian_blur_operator_2d(
    image_shape: tuple[int, int],
    sigma: float = DEFAULT_CONFIG.gaussian_blur_sigma,
    kernel_size: int = DEFAULT_CONFIG.gaussian_kernel_size,
) -> sparse.csr_matrix:
    """Construct the explicit sparse 2-D Gaussian blur operator.

    Images use shape ``(rows, columns)`` and C-order (row-major)
    vectorization. If ``Y = B_y @ X @ B_x.T``, then
    ``Y.ravel(order="C") = (B_y kron B_x) @ X.ravel(order="C")``.
    A sparse Kronecker product avoids allocating a dense
    ``(rows * columns)`` squared matrix.
    """
    rows, columns = _validate_image_shape(image_shape)
    vertical = gaussian_blur_matrix_1d(rows, sigma, kernel_size)
    horizontal = gaussian_blur_matrix_1d(columns, sigma, kernel_size)
    return sparse.kron(vertical, horizontal, format="csr")


def _validate_image_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 2
        or not all(isinstance(value, int) for value in image_shape)
        or any(value <= 0 for value in image_shape)
    ):
        raise ValueError("image_shape must contain two positive integers")
    return image_shape
