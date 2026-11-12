"""Tests for reproducible Gaussian blur-strength sensitivity analysis."""

import numpy as np
import pytest

from analysis.blur_sensitivity import (
    BLUR_SENSITIVITY_FIELDS,
    analyze_blur_strengths,
    blur_sensitivity_records,
)


def test_blur_sensitivity_has_consistent_dimensions_and_fields() -> None:
    results = analyze_blur_strengths(
        [0.5, 1.0],
        image_shape=(4, 5),
        kernel_size=3,
        relative_tolerance=1e-12,
    )

    assert [result.sigma for result in results] == [0.5, 1.0]
    assert all(result.image_shape == (4, 5) for result in results)
    assert all(result.total_singular_values == 20 for result in results)
    assert all(result.singular_values.shape == (20,) for result in results)
    assert all(result.relative_tolerance == 1e-12 for result in results)


def test_blur_sensitivity_is_reproducible() -> None:
    first = analyze_blur_strengths([0.8], (4, 4), 3, 1e-10)[0]
    second = analyze_blur_strengths([0.8], (4, 4), 3, 1e-10)[0]
    np.testing.assert_array_equal(first.singular_values, second.singular_values)
    assert first.condition_number == second.condition_number
    assert first.numerical_rank == second.numerical_rank


def test_blur_sensitivity_csv_record_structure() -> None:
    result = analyze_blur_strengths([0.8], (4, 4), 3, 1e-10)
    records = blur_sensitivity_records(result)
    assert len(records) == 1
    assert tuple(records[0]) == BLUR_SENSITIVITY_FIELDS


def test_blur_sensitivity_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="at least one"):
        analyze_blur_strengths([], (4, 4), 3, 1e-12)
    with pytest.raises(ValueError, match="strictly positive"):
        analyze_blur_strengths([-1.0], (4, 4), 3, 1e-12)
    with pytest.raises(ValueError, match="positive and odd"):
        analyze_blur_strengths([1.0], (4, 4), 4, 1e-12)
