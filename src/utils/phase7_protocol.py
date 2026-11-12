"""Matrix-free problem construction for Phase 7 iterative experiments."""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.linalg import svdvals
from scipy.sparse.linalg import LinearOperator
from skimage import data

from forward_model.gaussian import gaussian_blur_matrix_1d
from forward_model.noise import add_gaussian_noise
from forward_model.operators import gaussian_blur_linear_operator
from preprocessing.images import normalize_image


@dataclass(frozen=True)
class MatrixFreeDeblurringProblem:
    """Image, one LinearOperator, and its reproducible noisy observation."""

    original: np.ndarray
    operator: LinearOperator
    blurred: np.ndarray
    noise: np.ndarray
    noisy_observation: np.ndarray
    observation: np.ndarray
    image_shape: tuple[int, int]
    sigma: float
    kernel_size: int
    noise_standard_deviation: float
    random_seed: int
    boundary: str = "zero"


def structured_operator_condition_number(
    image_shape: tuple[int, int],
    *,
    sigma: float,
    kernel_size: int,
) -> float:
    """Compute exact 2-norm condition from 1-D factors, without forming A.

    For the project's Kronecker operator A = B_y kron B_x, singular values are
    pairwise products and kappa_2(A) = kappa_2(B_y) * kappa_2(B_x). Only the
    one-dimensional factors are formed and decomposed; no image-sized matrix
    or full image-level SVD is constructed.
    """
    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 2
        or any(not isinstance(size, int) or size <= 0 for size in image_shape)
    ):
        raise ValueError("image_shape must contain two positive integers")
    vertical = gaussian_blur_matrix_1d(
        image_shape[0], sigma=sigma, kernel_size=kernel_size
    )
    horizontal = gaussian_blur_matrix_1d(
        image_shape[1], sigma=sigma, kernel_size=kernel_size
    )
    vertical_values = svdvals(vertical, check_finite=False)
    horizontal_values = svdvals(horizontal, check_finite=False)
    if vertical_values[-1] == 0.0 or horizontal_values[-1] == 0.0:
        return float("inf")
    return float(
        (vertical_values[0] / vertical_values[-1])
        * (horizontal_values[0] / horizontal_values[-1])
    )


def create_matrix_free_problem(
    image_shape: tuple[int, int],
    *,
    sigma: float,
    kernel_size: int,
    noise_standard_deviation: float,
    random_seed: int,
) -> MatrixFreeDeblurringProblem:
    """Generate b by applying the exact matrix-free operator and seeded noise.

    No full two-dimensional blur matrix or dense SVD is constructed. The
    existing separable Gaussian LinearOperator applies the project's C-order,
    zero-boundary convention.
    """
    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 2
        or any(not isinstance(size, int) or size <= 0 for size in image_shape)
    ):
        raise ValueError("image_shape must contain two positive integers")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if (
        not isinstance(kernel_size, int)
        or isinstance(kernel_size, bool)
        or kernel_size <= 0
        or kernel_size % 2 == 0
    ):
        raise ValueError("kernel_size must be a positive odd integer")
    if (
        not np.isfinite(noise_standard_deviation)
        or noise_standard_deviation < 0.0
    ):
        raise ValueError("noise_standard_deviation must be finite and non-negative")
    if (
        not isinstance(random_seed, int)
        or isinstance(random_seed, bool)
        or random_seed < 0
    ):
        raise ValueError("random_seed must be a non-negative integer")

    source = Image.fromarray(data.camera())
    resized = source.resize(
        (image_shape[1], image_shape[0]),
        resample=Image.Resampling.LANCZOS,
    )
    original = normalize_image(np.asarray(resized))
    operator = gaussian_blur_linear_operator(
        image_shape,
        sigma=sigma,
        kernel_size=kernel_size,
    )
    blurred_vector = operator.matvec(original.ravel(order="C"))
    blurred = blurred_vector.reshape(image_shape, order="C")
    noisy_observation, noise = add_gaussian_noise(
        blurred,
        noise_standard_deviation=noise_standard_deviation,
        seed=random_seed,
    )
    return MatrixFreeDeblurringProblem(
        original=original,
        operator=operator,
        blurred=blurred,
        noise=noise,
        noisy_observation=noisy_observation,
        observation=noisy_observation.ravel(order="C"),
        image_shape=image_shape,
        sigma=float(sigma),
        kernel_size=kernel_size,
        noise_standard_deviation=float(noise_standard_deviation),
        random_seed=random_seed,
    )
