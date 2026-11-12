"""Integration and regression tests for Phase 4 regularization."""

import numpy as np

from evaluation.reconstruction_metrics import evaluate_reconstruction
from forward_model.degradation import degrade_image
from forward_model.gaussian import gaussian_blur_operator_2d
from numerical_methods.baseline_solvers import solve_qr_least_squares
from regularization.tikhonov import solve_tikhonov_svd
from regularization.tsvd import solve_tsvd


def _small_problem():
    image = np.linspace(0.0, 1.0, 16).reshape(4, 4)
    degradation = degrade_image(
        image,
        sigma=0.8,
        kernel_size=3,
        noise_standard_deviation=0.01,
        seed=42,
    )
    operator = gaussian_blur_operator_2d((4, 4), sigma=0.8, kernel_size=3)
    observation = degradation.noisy_observation.ravel(order="C")
    return image, operator, observation


def test_regularization_integrates_with_gaussian_operator_and_metrics() -> None:
    image, operator, observation = _small_problem()
    tsvd = solve_tsvd(operator, observation, image.shape, 8)
    tikhonov = solve_tikhonov_svd(operator, observation, image.shape, 0.01)

    for result in (tsvd, tikhonov):
        assert result.reconstruction_image.shape == image.shape
        metrics = evaluate_reconstruction(
            image,
            result.reconstruction_image,
            operator,
            observation,
        )
        assert np.isfinite(metrics.mean_squared_error)
        assert metrics.residual_norm >= 0.0


def test_regularization_is_reproducible_for_identical_observation() -> None:
    image, operator, observation = _small_problem()
    first_tsvd = solve_tsvd(operator, observation, image.shape, 8)
    second_tsvd = solve_tsvd(operator, observation, image.shape, 8)
    first_tikhonov = solve_tikhonov_svd(
        operator,
        observation,
        image.shape,
        0.01,
    )
    second_tikhonov = solve_tikhonov_svd(
        operator,
        observation,
        image.shape,
        0.01,
    )
    np.testing.assert_array_equal(
        first_tsvd.reconstruction_vector,
        second_tsvd.reconstruction_vector,
    )
    np.testing.assert_array_equal(
        first_tikhonov.reconstruction_vector,
        second_tikhonov.reconstruction_vector,
    )


def test_existing_qr_baseline_behavior_is_unchanged() -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    expected = np.array([0.25, 0.75])
    result = solve_qr_least_squares(matrix, matrix @ expected, (1, 2))
    np.testing.assert_allclose(result.reconstruction_vector, expected, atol=1e-12)
