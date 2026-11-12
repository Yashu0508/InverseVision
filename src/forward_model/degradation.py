"""High-level image degradation pipeline for the forward problem."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from config import DEFAULT_CONFIG
from forward_model.noise import add_gaussian_noise
from forward_model.operators import apply_separable_gaussian_blur
from preprocessing.images import validate_grayscale_image


@dataclass(frozen=True)
class BlurParameters:
    """Parameters defining the separable Gaussian blur."""

    sigma: float
    kernel_size: int
    boundary: str = "zero"


@dataclass(frozen=True)
class NoiseParameters:
    """Parameters defining additive Gaussian noise."""

    standard_deviation: float
    seed: int | None


@dataclass(frozen=True)
class DegradationResult:
    """Numerical outputs and metadata from one degradation experiment."""

    original: NDArray[np.float64]
    blurred: NDArray[np.float64]
    noise: NDArray[np.float64]
    noisy_observation: NDArray[np.float64]
    blur_parameters: BlurParameters
    noise_parameters: NoiseParameters


def degrade_image(
    image: np.ndarray,
    sigma: float = DEFAULT_CONFIG.gaussian_blur_sigma,
    kernel_size: int = DEFAULT_CONFIG.gaussian_kernel_size,
    noise_standard_deviation: float = DEFAULT_CONFIG.noise_standard_deviation,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> DegradationResult:
    """Validate, blur, and add Gaussian noise to a normalized image.

    The original is copied and all returned numerical arrays remain unclipped.
    Explicit arguments override the centralized configuration defaults.
    """
    validate_grayscale_image(image, require_normalized=True)
    original = image.astype(np.float64, copy=True)
    blurred = apply_separable_gaussian_blur(
        original,
        sigma=sigma,
        kernel_size=kernel_size,
    )
    noisy_observation, noise = add_gaussian_noise(
        blurred,
        noise_standard_deviation=noise_standard_deviation,
        seed=seed,
        rng=rng,
    )
    return DegradationResult(
        original=original,
        blurred=blurred,
        noise=noise,
        noisy_observation=noisy_observation,
        blur_parameters=BlurParameters(sigma=sigma, kernel_size=kernel_size),
        noise_parameters=NoiseParameters(
            standard_deviation=noise_standard_deviation,
            seed=(DEFAULT_CONFIG.random_seed if seed is None else seed)
            if rng is None
            else None,
        ),
    )
