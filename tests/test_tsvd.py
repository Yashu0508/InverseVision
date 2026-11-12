"""Tests for truncated singular value decomposition reconstruction."""

import numpy as np
import pytest

from analysis.svd_analysis import analyze_svd
from regularization.tsvd import solve_tsvd, tsvd_filter_factors


def test_tsvd_full_rank_recovers_well_conditioned_solution() -> None:
    matrix = np.diag([4.0, 3.0, 2.0, 1.0])
    expected = np.array([0.2, 0.4, 0.6, 0.8])
    result = solve_tsvd(matrix, matrix @ expected, (2, 2), 4)
    np.testing.assert_allclose(result.reconstruction_vector, expected, atol=1e-12)
    assert result.reconstruction_image.shape == (2, 2)
    assert result.residual_norm < 1e-12


def test_tsvd_truncation_preserves_singular_vector_alignment() -> None:
    matrix = np.diag([1.0, 3.0, 2.0])
    expected = np.array([4.0, 5.0, 6.0])
    result = solve_tsvd(matrix, matrix @ expected, (1, 3), 1)
    np.testing.assert_allclose(result.reconstruction_vector, [0.0, 5.0, 0.0])


def test_tsvd_different_ranks_give_different_reconstructions() -> None:
    matrix = np.diag([3.0, 2.0, 1.0])
    observation = np.array([3.0, 2.0, 1.0])
    first = solve_tsvd(matrix, observation, (1, 3), 1)
    second = solve_tsvd(matrix, observation, (1, 3), 2)
    assert not np.array_equal(first.reconstruction_vector, second.reconstruction_vector)
    assert second.residual_norm < first.residual_norm


@pytest.mark.parametrize("retained_rank", [0, -1, 4, 1.5, True])
def test_tsvd_rejects_invalid_rank(retained_rank) -> None:
    expected_error = TypeError if retained_rank in (1.5, True) else ValueError
    with pytest.raises(expected_error):
        solve_tsvd(np.eye(3), np.ones(3), (1, 3), retained_rank)


def test_tsvd_rejects_rank_beyond_numerically_usable_rank() -> None:
    matrix = np.diag([1.0, 1e-14, 0.0])
    with pytest.raises(ValueError, match="numerically usable rank"):
        solve_tsvd(
            matrix,
            np.ones(3),
            (1, 3),
            2,
            relative_tolerance=1e-12,
        )


def test_tsvd_filter_factors_never_divide_by_zero() -> None:
    with pytest.raises(ValueError, match="numerically zero"):
        tsvd_filter_factors(
            np.array([2.0, 0.0]),
            2,
            denominator_threshold=0.0,
        )
    factors = tsvd_filter_factors(np.array([2.0, 0.0]), 1)
    np.testing.assert_array_equal(factors.inverse_filter, [0.5, 0.0])
    np.testing.assert_array_equal(factors.retention_filter, [1.0, 0.0])


def test_tsvd_residual_and_input_preservation() -> None:
    matrix = np.array([[2.0, 0.0], [0.0, 1.0]])
    observation = np.array([2.0, 1.0])
    matrix_copy = matrix.copy()
    observation_copy = observation.copy()
    svd = analyze_svd(matrix, relative_tolerance=1e-12)
    result = solve_tsvd(
        matrix,
        observation,
        (1, 2),
        1,
        svd_analysis=svd,
    )
    np.testing.assert_allclose(
        result.residual_vector,
        matrix @ result.reconstruction_vector - observation,
    )
    assert np.all(np.isfinite(result.reconstruction_vector))
    np.testing.assert_array_equal(matrix, matrix_copy)
    np.testing.assert_array_equal(observation, observation_copy)
