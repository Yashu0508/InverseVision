"""Tests for Phase 2 unregularized dense baseline solvers."""

import numpy as np
import pytest
from scipy import sparse

from numerical_methods.baseline_solvers import (
    MAX_DENSE_PIXELS,
    solve_direct_inverse,
    solve_normal_equations,
    solve_qr_least_squares,
)


SOLVERS = (
    solve_direct_inverse,
    solve_qr_least_squares,
    solve_normal_equations,
)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_recovers_well_conditioned_system(solver) -> None:
    matrix = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 3.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )
    expected = np.array([0.2, 0.4, 0.6, 0.8])
    observation = matrix @ expected

    result = solver(matrix, observation, (2, 2))

    assert result.reconstruction_vector.shape == (4,)
    assert result.reconstruction_image.shape == (2, 2)
    assert result.residual_vector.shape == (4,)
    assert result.status == "success"
    assert result.execution_time_seconds >= 0.0
    np.testing.assert_allclose(result.reconstruction_vector, expected, atol=1e-12)
    np.testing.assert_allclose(
        result.residual_vector,
        observation - matrix @ result.reconstruction_vector,
        atol=1e-14,
    )
    assert result.residual_norm < 1e-12
    assert result.relative_residual < 1e-12


def test_qr_handles_overdetermined_full_rank_system() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expected = np.array([0.25, 0.75])
    result = solve_qr_least_squares(matrix, matrix @ expected, (1, 2))
    np.testing.assert_allclose(result.reconstruction_vector, expected, atol=1e-12)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers_do_not_clip_raw_reconstruction(solver) -> None:
    matrix = np.eye(4)
    observation = np.array([-1.0, 0.5, 1.5, 3.0])
    result = solver(matrix, observation, (2, 2))
    np.testing.assert_array_equal(result.reconstruction_vector, observation)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_rejects_invalid_observation_dimensions(solver) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        solver(np.eye(4), np.ones((2, 2)), (2, 2))
    with pytest.raises(ValueError, match="length"):
        solver(np.eye(4), np.ones(3), (2, 2))


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_rejects_nonfinite_inputs(solver) -> None:
    matrix = np.eye(4)
    matrix[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solver(matrix, np.ones(4), (2, 2))

    observation = np.ones(4)
    observation[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        solver(np.eye(4), observation, (2, 2))


def test_direct_inverse_requires_square_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        solve_direct_inverse(np.ones((3, 2)), np.ones(3), (1, 2))


def test_direct_inverse_rejects_singular_matrix() -> None:
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        solve_direct_inverse(singular, np.ones(2), (1, 2))


def test_qr_rejects_rank_deficient_matrix() -> None:
    rank_deficient = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        solve_qr_least_squares(rank_deficient, np.ones(3), (1, 2))


def test_normal_equations_reject_singular_gram_matrix() -> None:
    rank_deficient = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    with pytest.raises(np.linalg.LinAlgError, match="singular or unusable"):
        solve_normal_equations(rank_deficient, np.ones(3), (1, 2))


@pytest.mark.parametrize("solver", SOLVERS)
def test_dense_solver_limit_is_enforced_before_sparse_conversion(solver) -> None:
    dimension = MAX_DENSE_PIXELS + 1
    large_sparse_matrix = sparse.eye(dimension, format="csr")
    with pytest.raises(ValueError, match="dense solver limit exceeded"):
        solver(
            large_sparse_matrix,
            np.ones(dimension),
            (dimension, 1),
        )


def test_custom_dense_solver_limit_is_enforced() -> None:
    with pytest.raises(ValueError, match="configured limit of 3"):
        solve_qr_least_squares(
            np.eye(4),
            np.ones(4),
            (2, 2),
            max_dense_pixels=3,
        )
