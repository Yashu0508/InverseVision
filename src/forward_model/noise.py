"""Additive Gaussian noise generation."""

import numpy as np
from numpy.typing import NDArray

from config import DEFAULT_CONFIG


def add_gaussian_noise(
    image: np.ndarray,
    noise_standard_deviation: float = DEFAULT_CONFIG.noise_standard_deviation,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(image + noise, noise)`` without clipping either array.

    Supply either ``seed`` or an existing NumPy generator. When neither is
    supplied, the centralized project seed is used. A zero standard deviation
    returns exact zero noise.
    """
    array = np.asarray(image)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("image must be a non-empty two-dimensional array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image must have a numerical dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError("image must contain only finite values")
    if (
        not np.isscalar(noise_standard_deviation)
        or not np.isfinite(noise_standard_deviation)
        or noise_standard_deviation < 0.0
    ):
        raise ValueError(
            "noise_standard_deviation must be finite and non-negative"
        )
    if rng is not None and seed is not None:
        raise ValueError("provide either seed or rng, not both")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    values = array.astype(np.float64, copy=False)
    if noise_standard_deviation == 0.0:
        noise = np.zeros_like(values)
    else:
        effective_seed = DEFAULT_CONFIG.random_seed if seed is None else seed
        generator = rng if rng is not None else np.random.default_rng(effective_seed)
        noise = generator.normal(
            loc=0.0,
            scale=float(noise_standard_deviation),
            size=values.shape,
        )
    return values + noise, noise
