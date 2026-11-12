"""Tests for full and separable-structure singular-value analysis."""

import numpy as np
import pytest
from scipy import sparse

from analysis.svd_analysis import (
    analyze_structured_svd,
    analyze_svd,
    numerical_rank_analysis,
    singular_value_decay,
)


def test_full_svd_shapes_ordering_and_reconstruction() -> None:
    matrix = np.array(
        [[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 1.0], [2.0, 1.0, 1.0]]
    )
    result = analyze_svd(matrix, relative_tolerance=1e-12)
    reconstructed = (
        result.left_singular_vectors
        @ np.diag(result.singular_values)
        @ result.right_singular_vectors_transpose
    )

    assert result.left_singular_vectors.shape == (4, 3)
    assert result.singular_values.shape == (3,)
    assert result.right_singular_vectors_transpose.shape == (3, 3)
    assert np.all(np.diff(result.singular_values) <= 0.0)
    np.testing.assert_allclose(reconstructed, matrix, atol=1e-12)


@pytest.mark.parametrize(
    "matrix",
    [
        np.empty((0, 2)),
        np.array([[1.0, np.nan]]),
        np.array([[1.0, np.inf]]),
        np.array([1.0, 2.0]),
    ],
)
def test_full_svd_rejects_invalid_input(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError):
        analyze_svd(matrix)


def test_full_svd_rejects_large_sparse_before_conversion() -> None:
    with pytest.raises(ValueError, match="before sparse conversion"):
        analyze_svd(sparse.eye(257, format="csr"))


def test_rank_deficiency_preserves_smallest_value_and_reports_infinity() -> None:
    matrix = np.diag([3.0, 1e-14])
    result = analyze_svd(matrix, relative_tolerance=1e-12)

    assert result.numerical_rank == 1
    assert np.isinf(result.condition_number)
    assert result.smallest_singular_value == pytest.approx(1e-14)
    assert result.rank_threshold == pytest.approx(3e-12)


def test_full_rank_condition_number() -> None:
    result = analyze_svd(np.diag([4.0, 2.0]), relative_tolerance=1e-12)
    assert result.numerical_rank == 2
    assert result.condition_number == pytest.approx(2.0)


def test_numerical_rank_changes_with_reported_tolerance() -> None:
    singular_values = np.array([1.0, 1e-4, 1e-8])
    loose = numerical_rank_analysis(singular_values, 1e-3)
    strict = numerical_rank_analysis(singular_values, 1e-9)

    assert loose.numerical_rank == 1
    assert loose.absolute_threshold == pytest.approx(1e-3)
    assert strict.numerical_rank == 3
    assert strict.absolute_threshold == pytest.approx(1e-9)


def test_structured_pairwise_products_support_rectangular_factors() -> None:
    vertical = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    horizontal = np.array(
        [[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]
    )
    result = analyze_structured_svd(
        vertical,
        horizontal,
        relative_tolerance=1e-12,
    )
    expected = np.sort(
        np.multiply.outer(
            np.linalg.svd(vertical, compute_uv=False),
            np.linalg.svd(horizontal, compute_uv=False),
        ).ravel()
    )[::-1]

    assert result.operator_shape == (12, 6)
    assert result.singular_values.shape == (6,)
    assert np.all(np.diff(result.singular_values) <= 0.0)
    np.testing.assert_allclose(result.singular_values, expected)


def test_structured_spectrum_agrees_with_full_kronecker_svd() -> None:
    rng = np.random.default_rng(9)
    vertical = rng.normal(size=(4, 3))
    horizontal = rng.normal(size=(3, 2))
    structured = analyze_structured_svd(
        vertical,
        horizontal,
        relative_tolerance=1e-12,
    )
    full = analyze_svd(
        np.kron(vertical, horizontal),
        relative_tolerance=1e-12,
    )
    np.testing.assert_allclose(
        structured.singular_values,
        full.singular_values,
        rtol=1e-12,
        atol=1e-13,
    )


def test_singular_value_decay_quantities() -> None:
    decay = singular_value_decay(np.array([3.0, 1.0]))
    np.testing.assert_allclose(decay.normalized_singular_values, [1.0, 1.0 / 3.0])
    np.testing.assert_allclose(decay.cumulative_energy, [0.9, 1.0])
