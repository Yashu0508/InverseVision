"""Tests for additive Gaussian noise generation."""

import numpy as np
import pytest

from forward_model.noise import add_gaussian_noise


def test_zero_noise_leaves_image_unchanged() -> None:
    image = np.full((3, 4), 0.5)
    noisy, noise = add_gaussian_noise(image, 0.0, seed=3)

    np.testing.assert_array_equal(noisy, image)
    np.testing.assert_array_equal(noise, np.zeros_like(image))


def test_fixed_seed_is_reproducible() -> None:
    image = np.zeros((4, 4))
    first_noisy, first_noise = add_gaussian_noise(image, 0.2, seed=19)
    second_noisy, second_noise = add_gaussian_noise(image, 0.2, seed=19)

    np.testing.assert_array_equal(first_noise, second_noise)
    np.testing.assert_array_equal(first_noisy, second_noisy)


def test_generator_is_supported() -> None:
    image = np.zeros((2, 2))
    first, _ = add_gaussian_noise(
        image,
        0.1,
        rng=np.random.default_rng(5),
    )
    second, _ = add_gaussian_noise(
        image,
        0.1,
        rng=np.random.default_rng(5),
    )
    np.testing.assert_array_equal(first, second)


def test_negative_noise_standard_deviation_is_rejected() -> None:
    with pytest.raises(ValueError):
        add_gaussian_noise(np.zeros((2, 2)), -0.1)


def test_noise_is_not_clipped() -> None:
    image = np.ones((10, 10))
    noisy, _ = add_gaussian_noise(image, 1.0, seed=2)

    assert np.any(noisy > 1.0) or np.any(noisy < 0.0)
