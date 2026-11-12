"""Tests for image loading, normalization, and validation."""

import numpy as np
import pytest
from PIL import Image

from preprocessing.images import load_image, normalize_image, validate_grayscale_image


def test_normalize_uint8_and_preserve_normalized_float() -> None:
    integer_image = np.array([[0, 127, 255]], dtype=np.uint8)
    normalized = normalize_image(integer_image)
    already_normalized = np.array([[0.0, 0.25, 1.0]])

    np.testing.assert_allclose(normalized, [[0.0, 127.0 / 255.0, 1.0]])
    np.testing.assert_array_equal(
        normalize_image(already_normalized),
        already_normalized,
    )


@pytest.mark.parametrize(
    "invalid_image",
    [
        np.zeros((2, 2, 3)),
        np.array([], dtype=float).reshape(0, 0),
        np.array([[np.nan]]),
        np.array([[-1.0, 0.5]]),
        np.array([[0.0, 256.0]]),
    ],
)
def test_normalization_rejects_invalid_images(invalid_image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalize_image(invalid_image)


def test_validation_requires_normalized_intensity_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_grayscale_image(np.array([[1.01]]))


def test_load_rgb_image_converts_resizes_and_normalizes(tmp_path) -> None:
    source = np.zeros((3, 5, 3), dtype=np.uint8)
    source[..., 0] = 255
    path = tmp_path / "rgb.png"
    Image.fromarray(source, mode="RGB").save(path)

    loaded = load_image(path, target_shape=(4, 6))

    assert loaded.shape == (4, 6)
    assert loaded.dtype == np.float64
    assert np.all((0.0 <= loaded) & (loaded <= 1.0))


def test_load_image_reports_missing_and_unsupported_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")

    invalid_path = tmp_path / "invalid.txt"
    invalid_path.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported or invalid"):
        load_image(invalid_path)
