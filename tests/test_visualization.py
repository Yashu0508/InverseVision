"""Tests for basic degradation visualization."""

import matplotlib.pyplot as plt
import numpy as np

from visualization.degradation import plot_degradation
from visualization.reconstruction import (
    plot_metric_comparison,
    plot_reconstruction_comparison,
)


def test_plot_degradation_does_not_change_numerical_arrays() -> None:
    original = np.zeros((3, 3))
    blurred = np.full((3, 3), 0.5)
    noisy = np.full((3, 3), 1.2)
    noisy_copy = noisy.copy()

    figure, axes = plot_degradation(original, blurred, noisy, show=False)

    assert len(axes) == 3
    np.testing.assert_array_equal(noisy, noisy_copy)
    plt.close(figure)


def test_reconstruction_plot_displays_failures() -> None:
    image = np.zeros((3, 3))
    figure, axes = plot_reconstruction_comparison(
        image,
        image,
        {"Successful": image, "Failed": None},
        failures={"Failed": "singular matrix"},
        show=False,
    )
    assert len(axes) == 4
    assert "FAILED" in axes[-1].texts[0].get_text()
    plt.close(figure)


def test_metric_plot_retains_failed_method() -> None:
    records = [
        {
            "solver": "Successful",
            "status": "success",
            "relative_reconstruction_error": 0.1,
            "mean_squared_error": 0.01,
            "peak_signal_to_noise_ratio": 20.0,
            "structural_similarity_index": 0.9,
            "residual_norm": 0.01,
            "execution_time_seconds": 0.001,
        },
        {
            "solver": "Failed",
            "status": "failed",
            "relative_reconstruction_error": "",
            "mean_squared_error": "",
            "peak_signal_to_noise_ratio": "",
            "structural_similarity_index": "",
            "residual_norm": "",
            "execution_time_seconds": 0.002,
        },
    ]
    figure, axes = plot_metric_comparison(records, show=False)
    assert axes.shape == (2, 3)
    assert any(text.get_text() == "FAILED" for text in axes[0, 0].texts)
    plt.close(figure)
