"""Blur-strength sensitivity analysis using separable Gaussian structure."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from analysis.svd_analysis import analyze_structured_svd, singular_value_decay
from forward_model.gaussian import gaussian_blur_matrix_1d


BLUR_SENSITIVITY_FIELDS = (
    "sigma",
    "rows",
    "columns",
    "kernel_size",
    "relative_tolerance",
    "rank_threshold",
    "numerical_rank",
    "total_singular_values",
    "largest_singular_value",
    "smallest_singular_value",
    "condition_number",
    "median_normalized_singular_value",
    "components_for_90_percent_energy",
)


@dataclass(frozen=True)
class BlurSensitivityResult:
    """Spectrum summary for one Gaussian blur strength."""

    sigma: float
    image_shape: tuple[int, int]
    kernel_size: int
    relative_tolerance: float
    rank_threshold: float
    numerical_rank: int
    total_singular_values: int
    condition_number: float
    largest_singular_value: float
    smallest_singular_value: float
    median_normalized_singular_value: float
    components_for_90_percent_energy: int
    singular_values: np.ndarray
    normalized_singular_values: np.ndarray


def analyze_blur_strengths(
    sigma_values: Iterable[float],
    image_shape: tuple[int, int],
    kernel_size: int,
    relative_tolerance: float,
) -> list[BlurSensitivityResult]:
    """Analyze Gaussian spectra for fixed shape, kernel, and rank tolerance."""
    sigma_list = list(sigma_values)
    if len(sigma_list) == 0:
        raise ValueError("sigma_values must contain at least one value")
    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 2
        or not all(isinstance(value, int) for value in image_shape)
        or any(value <= 0 for value in image_shape)
    ):
        raise ValueError("image_shape must contain two positive integers")

    results: list[BlurSensitivityResult] = []
    for sigma in sigma_list:
        vertical = gaussian_blur_matrix_1d(
            image_shape[0],
            sigma=sigma,
            kernel_size=kernel_size,
        )
        horizontal = gaussian_blur_matrix_1d(
            image_shape[1],
            sigma=sigma,
            kernel_size=kernel_size,
        )
        analysis = analyze_structured_svd(
            vertical,
            horizontal,
            relative_tolerance=relative_tolerance,
        )
        decay = singular_value_decay(analysis.singular_values)
        energy_index = int(
            np.searchsorted(decay.cumulative_energy, 0.9, side="left") + 1
        )
        results.append(
            BlurSensitivityResult(
                sigma=float(sigma),
                image_shape=image_shape,
                kernel_size=kernel_size,
                relative_tolerance=relative_tolerance,
                rank_threshold=analysis.rank_threshold,
                numerical_rank=analysis.numerical_rank,
                total_singular_values=analysis.singular_values.size,
                condition_number=analysis.condition_number,
                largest_singular_value=analysis.largest_singular_value,
                smallest_singular_value=analysis.smallest_singular_value,
                median_normalized_singular_value=float(
                    np.median(decay.normalized_singular_values)
                ),
                components_for_90_percent_energy=energy_index,
                singular_values=analysis.singular_values,
                normalized_singular_values=decay.normalized_singular_values,
            )
        )
    return results


def blur_sensitivity_records(
    results: Iterable[BlurSensitivityResult],
) -> list[dict[str, float | int]]:
    """Convert sensitivity results to stable CSV-ready records."""
    return [
        {
            "sigma": result.sigma,
            "rows": result.image_shape[0],
            "columns": result.image_shape[1],
            "kernel_size": result.kernel_size,
            "relative_tolerance": result.relative_tolerance,
            "rank_threshold": result.rank_threshold,
            "numerical_rank": result.numerical_rank,
            "total_singular_values": result.total_singular_values,
            "largest_singular_value": result.largest_singular_value,
            "smallest_singular_value": result.smallest_singular_value,
            "condition_number": result.condition_number,
            "median_normalized_singular_value": (
                result.median_normalized_singular_value
            ),
            "components_for_90_percent_energy": (
                result.components_for_90_percent_energy
            ),
        }
        for result in results
    ]
