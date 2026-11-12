"""Demonstrate discrete Picard quantities and SVD noise amplification."""

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.noise_amplification import analyze_noise_amplification  # noqa: E402
from analysis.picard_analysis import analyze_picard  # noqa: E402
from analysis.svd_analysis import analyze_svd  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402
from forward_model.degradation import degrade_image  # noqa: E402
from forward_model.gaussian import gaussian_blur_operator_2d  # noqa: E402
from preprocessing.images import normalize_image  # noqa: E402
from visualization.svd_analysis import (  # noqa: E402
    plot_noise_amplification,
    plot_picard_analysis,
)


IMAGE_SHAPE = (8, 8)
RANK_RTOL = 1e-12


def _camera_image() -> np.ndarray:
    camera = Image.fromarray(data.camera())
    resized = camera.resize(
        (IMAGE_SHAPE[1], IMAGE_SHAPE[0]),
        resample=Image.Resampling.LANCZOS,
    )
    return normalize_image(np.asarray(resized))


def main() -> None:
    """Analyze one reproducible observation and its known noise realization."""
    original = _camera_image()
    degradation = degrade_image(
        original,
        sigma=DEFAULT_CONFIG.gaussian_blur_sigma,
        kernel_size=DEFAULT_CONFIG.gaussian_kernel_size,
        noise_standard_deviation=DEFAULT_CONFIG.noise_standard_deviation,
        seed=DEFAULT_CONFIG.random_seed,
    )
    operator = gaussian_blur_operator_2d(
        IMAGE_SHAPE,
        sigma=DEFAULT_CONFIG.gaussian_blur_sigma,
        kernel_size=DEFAULT_CONFIG.gaussian_kernel_size,
    )
    svd = analyze_svd(operator, relative_tolerance=RANK_RTOL)
    observation = degradation.noisy_observation.ravel(order="C")
    picard = analyze_picard(svd, observation)
    noise = analyze_noise_amplification(
        svd,
        IMAGE_SHAPE,
        DEFAULT_CONFIG.noise_standard_deviation,
        DEFAULT_CONFIG.random_seed,
    )
    np.testing.assert_array_equal(noise.noise, degradation.noise.ravel(order="C"))

    metrics_path = (
        DEFAULT_CONFIG.output_directories["metrics"]
        / "phase3_picard_noise.csv"
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "index",
                "singular_value",
                "expansion_coefficient",
                "coefficient_magnitude",
                "coefficient_to_singular_value_ratio",
                "transformed_noise_coefficient",
                "noise_contribution_magnitude",
                "safe_denominator",
            ),
        )
        writer.writeheader()
        for index in range(svd.singular_values.size):
            writer.writerow(
                {
                    "index": index + 1,
                    "singular_value": svd.singular_values[index],
                    "expansion_coefficient": picard.expansion_coefficients[index],
                    "coefficient_magnitude": picard.coefficient_magnitudes[index],
                    "coefficient_to_singular_value_ratio": (
                        picard.coefficient_to_singular_value_ratios[index]
                    ),
                    "transformed_noise_coefficient": (
                        noise.transformed_noise_coefficients[index]
                    ),
                    "noise_contribution_magnitude": (
                        noise.contribution_magnitudes[index]
                    ),
                    "safe_denominator": noise.safe_denominator_mask[index],
                }
            )

    picard_figure, _ = plot_picard_analysis(picard, show=False)
    picard_path = (
        DEFAULT_CONFIG.output_directories["plots"] / "phase3_picard.png"
    )
    picard_figure.savefig(picard_path, dpi=150)
    plt.close(picard_figure)

    noise_figure, _ = plot_noise_amplification(
        svd.singular_values,
        noise,
        show=False,
    )
    noise_path = (
        DEFAULT_CONFIG.output_directories["plots"]
        / "phase3_noise_amplification.png"
    )
    noise_figure.savefig(noise_path, dpi=150)
    plt.close(noise_figure)

    finite_contributions = noise.contribution_magnitudes[
        np.isfinite(noise.contribution_magnitudes)
    ]
    print(f"Noise norm: {noise.noise_norm}")
    print(f"Transformed noise norm: {noise.transformed_noise_norm}")
    print(f"Maximum finite noise contribution: {np.max(finite_contributions)}")
    print(f"Metrics: {metrics_path}")
    print(f"Picard plot: {picard_path}")
    print(f"Noise plot: {noise_path}")


if __name__ == "__main__":
    main()
