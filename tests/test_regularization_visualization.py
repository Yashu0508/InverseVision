"""Smoke tests for Phase 4 parameter and filter plots."""

import matplotlib.pyplot as plt
import numpy as np

from visualization.regularization import (
    plot_parameter_tradeoffs,
    plot_tikhonov_filters,
    plot_tsvd_filters,
)


def test_regularization_tradeoff_plot() -> None:
    records = [
        {
            "parameter_value": 1,
            "relative_reconstruction_error": 0.5,
            "residual_norm": 0.2,
        },
        {
            "parameter_value": 2,
            "relative_reconstruction_error": 0.3,
            "residual_norm": 0.1,
        },
    ]
    figure, axes = plot_parameter_tradeoffs(
        records,
        "parameter_value",
        "Parameter",
        show=False,
    )
    assert len(axes) == 2
    plt.close(figure)


def test_regularization_filter_plots() -> None:
    singular_values = np.array([2.0, 1.0, 0.5])
    tsvd_figure, tsvd_axes = plot_tsvd_filters(
        singular_values,
        [1, 2],
        0.0,
        show=False,
    )
    tikhonov_figure, tikhonov_axes = plot_tikhonov_filters(
        singular_values,
        [0.1, 0.01],
        show=False,
    )
    assert len(tsvd_axes) == 2
    assert len(tikhonov_axes) == 2
    plt.close(tsvd_figure)
    plt.close(tikhonov_figure)
