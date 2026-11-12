"""Tests for the complete image degradation pipeline."""

import numpy as np

from forward_model.degradation import degrade_image


def test_pipeline_outputs_and_preserves_original() -> None:
    image = np.linspace(0.0, 1.0, 30).reshape(5, 6)
    input_copy = image.copy()

    result = degrade_image(
        image,
        sigma=0.8,
        kernel_size=3,
        noise_standard_deviation=0.1,
        seed=11,
    )

    arrays = [
        result.original,
        result.blurred,
        result.noise,
        result.noisy_observation,
    ]
    assert all(array.shape == image.shape for array in arrays)
    assert all(np.all(np.isfinite(array)) for array in arrays)
    np.testing.assert_array_equal(image, input_copy)
    assert not np.shares_memory(result.original, image)
    assert not np.array_equal(result.original, result.blurred)
    assert not np.array_equal(result.blurred, result.noisy_observation)
    np.testing.assert_allclose(
        result.noisy_observation,
        result.blurred + result.noise,
    )
    assert result.blur_parameters.boundary == "zero"
    assert result.noise_parameters.seed == 11


def test_pipeline_zero_noise_matches_blurred_image() -> None:
    image = np.eye(4)
    result = degrade_image(
        image,
        noise_standard_deviation=0.0,
    )

    np.testing.assert_array_equal(result.noisy_observation, result.blurred)
