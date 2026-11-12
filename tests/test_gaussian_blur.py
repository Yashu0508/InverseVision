"""Tests for explicit and matrix-free Gaussian blur representations."""

import numpy as np
import pytest

from forward_model.gaussian import (
    gaussian_blur_matrix_1d,
    gaussian_blur_operator_2d,
    gaussian_kernel_1d,
)
from forward_model.operators import (
    apply_separable_gaussian_blur,
    gaussian_blur_linear_operator,
)


def test_gaussian_kernel_length_symmetry_and_normalization() -> None:
    kernel = gaussian_kernel_1d(sigma=1.2, kernel_size=7)

    assert kernel.shape == (7,)
    np.testing.assert_allclose(kernel, kernel[::-1])
    assert np.sum(kernel) == pytest.approx(1.0)


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.inf, np.nan])
def test_gaussian_kernel_rejects_invalid_sigma(sigma: float) -> None:
    with pytest.raises(ValueError):
        gaussian_kernel_1d(sigma=sigma, kernel_size=3)


@pytest.mark.parametrize("kernel_size", [0, -3, 2, 4])
def test_gaussian_kernel_rejects_invalid_size(kernel_size: int) -> None:
    with pytest.raises(ValueError):
        gaussian_kernel_1d(sigma=1.0, kernel_size=kernel_size)


def test_blur_matrix_dimensions_placement_and_boundaries() -> None:
    kernel = gaussian_kernel_1d(sigma=1.0, kernel_size=3)
    matrix = gaussian_blur_matrix_1d(5, sigma=1.0, kernel_size=3)

    assert matrix.shape == (5, 5)
    np.testing.assert_allclose(matrix[2, 1:4], kernel)
    np.testing.assert_allclose(matrix[0, :2], kernel[1:])
    assert matrix[0].sum() < 1.0
    assert matrix[2].sum() == pytest.approx(1.0)


def test_blur_matrix_supports_dimension_smaller_than_kernel() -> None:
    kernel = gaussian_kernel_1d(sigma=1.0, kernel_size=5)
    matrix = gaussian_blur_matrix_1d(2, sigma=1.0, kernel_size=5)

    expected = np.array(
        [
            [kernel[2], kernel[3]],
            [kernel[1], kernel[2]],
        ]
    )
    np.testing.assert_allclose(matrix, expected)


def test_explicit_operator_matches_matrix_free_blur() -> None:
    image = np.arange(20, dtype=float).reshape(4, 5) / 19.0
    vertical = gaussian_blur_matrix_1d(4, sigma=0.9, kernel_size=5)
    horizontal = gaussian_blur_matrix_1d(5, sigma=0.9, kernel_size=5)
    operator = gaussian_blur_operator_2d(
        image.shape,
        sigma=0.9,
        kernel_size=5,
    )

    explicit = operator @ image.ravel(order="C")
    matrix_free = apply_separable_gaussian_blur(
        image,
        vertical_matrix=vertical,
        horizontal_matrix=horizontal,
    ).ravel(order="C")

    assert np.max(np.abs(explicit - matrix_free)) < 1e-10


def test_linear_operator_forward_shape_and_result() -> None:
    image = np.arange(12, dtype=float).reshape(3, 4)
    operator = gaussian_blur_linear_operator(
        image.shape,
        sigma=1.1,
        kernel_size=3,
    )
    expected = apply_separable_gaussian_blur(
        image,
        sigma=1.1,
        kernel_size=3,
    ).ravel(order="C")

    actual = operator.matvec(image.ravel(order="C"))

    assert operator.shape == (12, 12)
    assert actual.shape == (12,)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_linear_operator_adjoint_identity() -> None:
    rng = np.random.default_rng(7)
    shape = (3, 5)
    left = rng.normal(size=np.prod(shape))
    right = rng.normal(size=np.prod(shape))
    operator = gaussian_blur_linear_operator(shape, sigma=0.8, kernel_size=3)

    forward_inner_product = np.vdot(operator.matvec(left), right)
    adjoint_inner_product = np.vdot(left, operator.rmatvec(right))

    assert forward_inner_product == pytest.approx(
        adjoint_inner_product,
        abs=1e-12,
    )
