"""Comparison plots for Phase 2 baseline reconstructions and metrics."""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_reconstruction_comparison(
    original: np.ndarray,
    noisy_observation: np.ndarray,
    reconstructions: Mapping[str, np.ndarray | None],
    *,
    failures: Mapping[str, str] | None = None,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot original, observation, and solver outputs without clipping arrays."""
    original_array = np.asarray(original)
    observation_array = np.asarray(noisy_observation)
    if original_array.ndim != 2 or observation_array.shape != original_array.shape:
        raise ValueError("original and observation must be same-shaped 2-D images")
    if len(reconstructions) == 0:
        raise ValueError("at least one reconstruction entry is required")

    panel_count = 2 + len(reconstructions)
    figure, axes = plt.subplots(
        1,
        panel_count,
        figsize=(4 * panel_count, 4),
        constrained_layout=True,
    )
    axes[0].imshow(original_array, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original")
    axes[1].imshow(observation_array, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Noisy observation")

    failure_messages = failures or {}
    for axis, (name, reconstruction) in zip(
        axes[2:],
        reconstructions.items(),
        strict=True,
    ):
        if reconstruction is None:
            axis.text(
                0.5,
                0.5,
                f"FAILED\n{failure_messages.get(name, 'No result')}",
                ha="center",
                va="center",
                wrap=True,
                color="darkred",
            )
        else:
            image = np.asarray(reconstruction)
            if image.shape != original_array.shape or not np.all(np.isfinite(image)):
                raise ValueError(f"reconstruction '{name}' is invalid")
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(name)

    for axis in axes:
        axis.axis("off")
    if show:
        plt.show()
    return figure, axes


def plot_metric_comparison(
    records: Sequence[Mapping[str, object]],
    *,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot six baseline metrics, retaining visible failed-method entries."""
    if len(records) == 0:
        raise ValueError("at least one metric record is required")

    metric_specs = (
        ("relative_reconstruction_error", "Relative reconstruction error"),
        ("mean_squared_error", "MSE"),
        ("peak_signal_to_noise_ratio", "PSNR (dB)"),
        ("structural_similarity_index", "SSIM"),
        ("residual_norm", "Residual norm"),
        ("execution_time_seconds", "Execution time (s)"),
    )
    labels = [str(record["solver"]) for record in records]
    statuses = [str(record["status"]) for record in records]
    colors = ["tab:blue" if status == "success" else "tab:red" for status in statuses]

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (field, title) in zip(axes.ravel(), metric_specs, strict=True):
        values: list[float] = []
        annotations: list[str | None] = []
        for record, status in zip(records, statuses, strict=True):
            value = record.get(field)
            numeric = float(value) if value not in (None, "") else float("nan")
            if status != "success":
                values.append(0.0)
                annotations.append("FAILED")
            elif np.isposinf(numeric):
                values.append(0.0)
                annotations.append("∞")
            elif not np.isfinite(numeric):
                values.append(0.0)
                annotations.append("N/A")
            else:
                values.append(numeric)
                annotations.append(None)

        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
        for bar, annotation in zip(bars, annotations, strict=True):
            if annotation is not None:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    annotation,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    if show:
        plt.show()
    return figure, axes
