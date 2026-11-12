"""Tests for standard-form L2 Tikhonov reconstruction."""

import numpy as np
import pytest

from regularization.tikhonov import (
    solve_tikhonov_augmented,
    solve_tikhonov_svd,
    tikhonov_filter_factors,
    tikhonov_objective,
)


def test_tikhonov_svd_matches_analytic_diagonal_solution() -> None:
    matrix = np.diag([3.0, 2.0])
    observation = np.array([6.0, 4.0])
    parameter = 0.5
    result = solve_tikhonov_svd(
        matrix,
        observation,
        (1, 2),
        parameter,
    )
    expected = np.array(
        [3.0 * 6.0 / (9.0 + parameter**2), 2.0 * 4.0 / (4.0 + parameter**2)]
    )
    np.testing.assert_allclose(result.reconstruction_vector, expected, atol=1e-12)


@pytest.mark.parametrize("parameter", [0.1, 0.01, 1e-4])
def test_svd_and_augmented_tikhonov_agree(parameter: float) -> None:
    matrix = np.array([[2.0, 1.0], [1.0, 3.0], [0.5, -1.0]])
    observation = np.array([1.0, 2.0, -0.5])
    svd = solve_tikhonov_svd(matrix, observation, (1, 2), parameter)
    augmented = solve_tikhonov_augmented(matrix, observation, (1, 2), parameter)
    np.testing.assert_allclose(
        svd.reconstruction_vector,
        augmented.reconstruction_vector,
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("parameter", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_tikhonov_rejects_invalid_lambda(parameter: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        solve_tikhonov_svd(np.eye(2), np.ones(2), (1, 2), parameter)


def test_tikhonov_filter_factors_match_definitions() -> None:
    singular_values = np.array([2.0, 1.0])
    factors = tikhonov_filter_factors(singular_values, 0.5)
    np.testing.assert_allclose(
        factors.inverse_filter,
        singular_values / (singular_values**2 + 0.25),
    )
    np.testing.assert_allclose(
        factors.retention_filter,
        singular_values**2 / (singular_values**2 + 0.25),
    )


def test_tikhonov_objective_matches_definition() -> None:
    matrix = np.array([[2.0, 0.0], [0.0, 1.0]])
    observation = np.array([1.0, 2.0])
    reconstruction = np.array([0.5, 1.5])
    parameter = 0.2
    residual = matrix @ reconstruction - observation
    expected = residual @ residual + parameter**2 * (
        reconstruction @ reconstruction
    )
    assert tikhonov_objective(
        matrix,
        observation,
        reconstruction,
        parameter,
    ) == pytest.approx(expected)


def test_tikhonov_is_finite_for_ill_conditioned_system() -> None:
    matrix = np.diag([1.0, 1e-12, 0.0])
    result = solve_tikhonov_svd(matrix, np.ones(3), (1, 3), 1e-3)
    assert np.all(np.isfinite(result.reconstruction_vector))


def test_tikhonov_residual_and_inputs_are_preserved() -> None:
    matrix = np.array([[2.0, 0.0], [0.0, 1.0]])
    observation = np.array([2.0, 1.0])
    matrix_copy = matrix.copy()
    observation_copy = observation.copy()
    result = solve_tikhonov_augmented(matrix, observation, (1, 2), 0.1)
    np.testing.assert_allclose(
        result.residual_vector,
        matrix @ result.reconstruction_vector - observation,
    )
    np.testing.assert_array_equal(matrix, matrix_copy)
    np.testing.assert_array_equal(observation, observation_copy)
