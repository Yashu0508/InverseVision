"""Image loading, normalization, and validation utilities."""

from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from config import DEFAULT_CONFIG


FloatImage: TypeAlias = NDArray[np.float64]


def validate_grayscale_image(
    image: np.ndarray,
    *,
    require_normalized: bool = True,
) -> None:
    """Validate a two-dimensional numerical grayscale image.

    Args:
        image: Image array to validate.
        require_normalized: Require every intensity to lie in ``[0, 1]``.

    Raises:
        TypeError: If ``image`` is not a NumPy array with a numerical dtype.
        ValueError: If the image is empty, not two-dimensional, contains
            non-finite values, or violates the required intensity range.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("image must have a numerical dtype")
    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array")
    if image.size == 0 or any(dimension <= 0 for dimension in image.shape):
        raise ValueError("image must be non-empty with positive dimensions")
    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain only finite numerical values")
    if require_normalized and (np.min(image) < 0.0 or np.max(image) > 1.0):
        raise ValueError("normalized image intensities must lie in [0, 1]")


def normalize_image(image: np.ndarray) -> FloatImage:
    """Return a validated grayscale image normalized to ``[0, 1]``.

    Integer and floating-point arrays in ``[0, 255]`` are supported. Arrays
    already in ``[0, 1]`` are copied without a second scaling operation.
    Values outside these documented ranges are rejected rather than rescaled
    from their observed minimum and maximum.
    """
    validate_grayscale_image(image, require_normalized=False)

    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError("image intensities must lie in [0, 255]")

    normalized = image.astype(np.float64, copy=True)
    if maximum > 1.0:
        normalized /= 255.0

    validate_grayscale_image(normalized, require_normalized=True)
    return normalized


def _validate_target_shape(target_shape: tuple[int, int]) -> None:
    if (
        len(target_shape) != 2
        or not all(isinstance(value, int) for value in target_shape)
        or any(value <= 0 for value in target_shape)
    ):
        raise ValueError("target_shape must contain two positive integers")


def load_image(
    path: str | Path,
    target_shape: tuple[int, int] = DEFAULT_CONFIG.image_shape,
) -> FloatImage:
    """Load an image as normalized grayscale, resizing when necessary.

    ``target_shape`` follows NumPy's ``(height, width)`` convention. Pillow is
    used for decoding and converts color or palette images to luminance.

    Raises:
        FileNotFoundError: If the path does not identify an existing file.
        ValueError: If the target shape is invalid or the image cannot be
            decoded as a supported image format.
        OSError: If the image file cannot be read for another I/O reason.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"image file does not exist: {image_path}")
    _validate_target_shape(target_shape)

    try:
        with Image.open(image_path) as source:
            source.load()
            grayscale = source.convert("L")
            expected_size = (target_shape[1], target_shape[0])
            if grayscale.size != expected_size:
                grayscale = grayscale.resize(
                    expected_size,
                    resample=Image.Resampling.LANCZOS,
                )
            array = np.asarray(grayscale)
    except UnidentifiedImageError as error:
        raise ValueError(
            f"unsupported or invalid image format: {image_path}"
        ) from error

    return normalize_image(array)
