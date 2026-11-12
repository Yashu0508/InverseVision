"""Visualization helpers for degradation experiments."""

from visualization.degradation import plot_degradation
from visualization.reconstruction import (
    plot_metric_comparison,
    plot_reconstruction_comparison,
)
from visualization.regularization import (
    plot_parameter_tradeoffs,
    plot_tikhonov_filters,
    plot_tsvd_filters,
)
from visualization.svd_analysis import (
    plot_blur_sensitivity,
    plot_noise_amplification,
    plot_picard_analysis,
    plot_singular_value_analysis,
    plot_structured_full_comparison,
)

__all__ = [
    "plot_degradation",
    "plot_metric_comparison",
    "plot_blur_sensitivity",
    "plot_noise_amplification",
    "plot_parameter_tradeoffs",
    "plot_picard_analysis",
    "plot_reconstruction_comparison",
    "plot_singular_value_analysis",
    "plot_structured_full_comparison",
    "plot_tikhonov_filters",
    "plot_tsvd_filters",
]
