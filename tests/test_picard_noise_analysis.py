"""Tests for Picard coefficients and SVD-coordinate noise amplification."""

import numpy as np
import pytest

from analysis.noise_amplification import analyze_noise_amplification
from analysis.picard_analysis import analyze_picard
from analysis.svd_analysis import analyze_svd


def test_picard_coefficients_align_with_left_singular_vectors() -> None:
    matrix = np.array([[2.0, 0.0], [0.0, 1.0]])
    observation = np.array([4.0, -3.0])
    svd = analyze_svd(matrix, relative_tolerance=1e-12)
    result = analyze_picard(svd, observation)
    expected = svd.left_singular_vectors.T @ observation

    np.testing.assert_allclose(result.expansion_coefficients, expected)
    np.testing.assert_allclose(result.coefficient_magnitudes, np.abs(expected))
    np.testing.assert_allclose(
        result.coefficient_to_singular_value_ratios,
        np.abs(expected) / svd.singular_values,
    )


def test_picard_small_singular_value_is_handled_without_division() -> None:
    svd = analyze_svd(np.diag([1.0, 0.0]), relative_tolerance=1e-12)
    nonzero = analyze_picard(svd, np.array([1.0, 2.0]))
    zero = analyze_picard(svd, np.array([1.0, 0.0]))

    assert not nonzero.safe_denominator_mask[-1]
    assert np.isinf(nonzero.coefficient_to_singular_value_ratios[-1])
    assert zero.coefficient_to_singular_value_ratios[-1] == 0.0


def test_picard_rejects_misaligned_observation() -> None:
    svd = analyze_svd(np.eye(3))
    with pytest.raises(ValueError, match="row count"):
        analyze_picard(svd, np.ones(2))


def test_noise_analysis_is_reproducible_and_correctly_transformed() -> None:
    svd = analyze_svd(np.eye(4), relative_tolerance=1e-12)
    first = analyze_noise_amplification(svd, (2, 2), 0.2, 17)
    second = analyze_noise_amplification(svd, (2, 2), 0.2, 17)

    np.testing.assert_array_equal(first.noise, second.noise)
    np.testing.assert_array_equal(
        first.transformed_noise_coefficients,
        second.transformed_noise_coefficients,
    )
    np.testing.assert_allclose(
        first.transformed_noise_coefficients,
        svd.left_singular_vectors.T @ first.noise,
    )
    assert first.noise.shape == (4,)
    assert first.contribution_magnitudes.shape == (4,)


def test_noise_analysis_does_not_clip_and_handles_zero_singular_value() -> None:
    svd = analyze_svd(np.diag([1.0, 0.0]), relative_tolerance=1e-12)
    result = analyze_noise_amplification(svd, (1, 2), 5.0, 2)

    assert np.any(np.abs(result.noise) > 1.0)
    assert not result.safe_denominator_mask[-1]
    assert np.isinf(result.contribution_magnitudes[-1])


def test_noise_analysis_rejects_incompatible_shape() -> None:
    svd = analyze_svd(np.eye(4))
    with pytest.raises(ValueError, match="row count"):
        analyze_noise_amplification(svd, (3, 2), 0.1, 1)
