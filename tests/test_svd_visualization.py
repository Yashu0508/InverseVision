"""Smoke tests for Phase 3 diagnostic visualizations."""

import matplotlib.pyplot as plt
import numpy as np

from analysis.blur_sensitivity import analyze_blur_strengths
from analysis.noise_amplification import analyze_noise_amplification
from analysis.picard_analysis import analyze_picard
from analysis.svd_analysis import analyze_svd
from visualization.svd_analysis import (
    plot_blur_sensitivity,
    plot_noise_amplification,
    plot_picard_analysis,
    plot_singular_value_analysis,
    plot_structured_full_comparison,
)


def test_svd_picard_and_noise_plots() -> None:
    svd = analyze_svd(np.diag([2.0, 1.0, 0.5, 0.25]), relative_tolerance=1e-12)
    picard = analyze_picard(svd, np.ones(4))
    noise = analyze_noise_amplification(svd, (2, 2), 0.1, 4)

    svd_figure, svd_axes = plot_singular_value_analysis(svd, show=False)
    picard_figure, picard_axes = plot_picard_analysis(picard, show=False)
    noise_figure, noise_axes = plot_noise_amplification(
        svd.singular_values,
        noise,
        show=False,
    )

    assert svd_axes.shape == (2, 2)
    assert len(picard_axes) == 2
    assert len(noise_axes) == 2
    plt.close(svd_figure)
    plt.close(picard_figure)
    plt.close(noise_figure)


def test_blur_sensitivity_plot() -> None:
    results = analyze_blur_strengths([0.5, 1.0], (4, 4), 3, 1e-12)
    figure, axes = plot_blur_sensitivity(results, show=False)
    assert len(axes) == 2
    plt.close(figure)


def test_structured_full_comparison_plot() -> None:
    singular_values = np.array([1.0, 0.5, 0.1])
    timings = {
        "full_min_seconds": 0.9,
        "full_median_seconds": 1.0,
        "full_max_seconds": 1.2,
        "structured_min_seconds": 0.08,
        "structured_median_seconds": 0.1,
        "structured_max_seconds": 0.15,
    }
    figure, axes = plot_structured_full_comparison(
        singular_values,
        singular_values.copy(),
        timings,
        show=False,
    )
    assert len(axes) == 2
    plt.close(figure)
