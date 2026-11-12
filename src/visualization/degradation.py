"""Basic visualization for image-degradation results."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_degradation(
    original: np.ndarray,
    blurred: np.ndarray,
    noisy_observation: np.ndarray,
    *,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot original, blurred, and noisy images with shared grayscale limits.

    Display limits are fixed to ``[0, 1]``. This affects rendering only; the
    input numerical arrays, including out-of-range noisy values, are unchanged.
    """
    arrays = [np.asarray(original), np.asarray(blurred), np.asarray(noisy_observation)]
    if any(array.ndim != 2 or array.size == 0 for array in arrays):
        raise ValueError("all images must be non-empty two-dimensional arrays")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("all images must have the same shape")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all images must contain only finite values")

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    titles = ("Original", "Gaussian blurred", "Blurred with Gaussian noise")
    for axis, image, title in zip(axes, arrays, titles, strict=True):
        typed_axis: Axes = axis
        typed_axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        typed_axis.set_title(title)
        typed_axis.axis("off")

    if show:
        plt.show()
    return figure, axes
