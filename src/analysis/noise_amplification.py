"""SVD-coordinate analysis of additive Gaussian observation noise."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis.svd_analysis import SVDAnalysis
from forward_model.noise import add_gaussian_noise


@dataclass(frozen=True)
class NoiseAmplificationAnalysis:
    """Raw noise and its unregularized inverse-direction contributions."""

    noise: NDArray[np.float64]
    transformed_noise_coefficients: NDArray[np.float64]
    contribution_magnitudes: NDArray[np.float64]
    safe_denominator_mask: NDArray[np.bool_]
    denominator_threshold: float
    noise_norm: float
    transformed_noise_norm: float
    noise_standard_deviation: float
    seed: int


def analyze_noise_amplification(
    svd_analysis: SVDAnalysis,
    observation_shape: tuple[int, int],
    noise_standard_deviation: float,
    seed: int,
    *,
    denominator_threshold: float | None = None,
) -> NoiseAmplificationAnalysis:
    """Generate noise and analyze ``|u_i.T epsilon / sigma_i|`` safely.

    This function describes inverse-direction noise amplification only. It does
    not construct, filter, or return an image reconstruction.
    """
    if (
        not isinstance(observation_shape, tuple)
        or len(observation_shape) != 2
        or not all(isinstance(value, int) for value in observation_shape)
        or any(value <= 0 for value in observation_shape)
    ):
        raise ValueError("observation_shape must contain two positive integers")
    if int(np.prod(observation_shape)) != svd_analysis.matrix_shape[0]:
        raise ValueError(
            "the observation shape must match the SVD matrix row count"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    _, noise_image = add_gaussian_noise(
        np.zeros(observation_shape, dtype=np.float64),
        noise_standard_deviation=noise_standard_deviation,
        seed=seed,
    )
    noise = noise_image.ravel(order="C")
    transformed = svd_analysis.left_singular_vectors.T @ noise
    magnitudes = np.abs(transformed)
    threshold = (
        svd_analysis.rank_threshold
        if denominator_threshold is None
        else float(denominator_threshold)
    )
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("denominator_threshold must be finite and non-negative")

    singular_values = svd_analysis.singular_values
    safe = singular_values > threshold
    contributions = np.empty_like(magnitudes)
    np.divide(magnitudes, singular_values, out=contributions, where=safe)
    unsafe = ~safe
    contributions[unsafe & (magnitudes > 0.0)] = np.inf
    contributions[unsafe & (magnitudes == 0.0)] = 0.0

    return NoiseAmplificationAnalysis(
        noise=noise,
        transformed_noise_coefficients=transformed,
        contribution_magnitudes=contributions,
        safe_denominator_mask=safe,
        denominator_threshold=threshold,
        noise_norm=float(np.linalg.norm(noise)),
        transformed_noise_norm=float(np.linalg.norm(transformed)),
        noise_standard_deviation=float(noise_standard_deviation),
        seed=seed,
    )
