"""Visualization utilities for Phase 3 numerical conditioning analysis."""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from analysis.blur_sensitivity import BlurSensitivityResult
from analysis.noise_amplification import NoiseAmplificationAnalysis
from analysis.picard_analysis import PicardAnalysis
from analysis.svd_analysis import SVDAnalysis, singular_value_decay


def _positive_finite(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0.0)


def plot_singular_value_analysis(
    analysis: SVDAnalysis,
    *,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot linear, log, normalized, and cumulative-energy spectra."""
    values = analysis.singular_values
    indices = np.arange(1, values.size + 1)
    decay = singular_value_decay(values)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(indices, values, marker=".")
    axes[0, 0].set_title("Singular values — linear scale")
    axes[0, 0].set_ylabel(r"$\sigma_i$")

    axes[0, 1].semilogy(indices, values, marker=".")
    if analysis.rank_threshold > 0.0:
        axes[0, 1].axhline(
            analysis.rank_threshold,
            color="tab:red",
            linestyle="--",
            label="Rank threshold",
        )
    axes[0, 1].set_title("Singular values — logarithmic scale")
    axes[0, 1].set_ylabel(r"$\sigma_i$")
    if analysis.rank_threshold > 0.0:
        axes[0, 1].legend()

    axes[1, 0].semilogy(
        indices,
        decay.normalized_singular_values,
        marker=".",
    )
    axes[1, 0].set_title("Normalized singular values")
    axes[1, 0].set_ylabel(r"$\sigma_i/\sigma_1$")

    axes[1, 1].plot(indices, decay.cumulative_energy, marker=".")
    axes[1, 1].set_ylim(0.0, 1.02)
    axes[1, 1].set_title("Cumulative squared singular-value energy")
    axes[1, 1].set_ylabel(r"$E_k$")

    for axis in axes.ravel():
        axis.set_xlabel("Singular-value index")
        axis.grid(alpha=0.25)
    if show:
        plt.show()
    return figure, axes


def plot_picard_analysis(
    analysis: PicardAnalysis,
    *,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot singular values, expansion coefficients, and safe ratios."""
    indices = np.arange(1, analysis.singular_values.size + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].semilogy(
        indices,
        analysis.singular_values,
        label=r"$\sigma_i$",
    )
    axes[0].semilogy(
        indices,
        analysis.coefficient_magnitudes,
        label=r"$|u_i^T b|$",
    )
    if analysis.denominator_threshold > 0.0:
        axes[0].axhline(
            analysis.denominator_threshold,
            color="tab:red",
            linestyle="--",
            label="Denominator threshold",
        )
    axes[0].set_title("Discrete Picard quantities")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend()

    ratios = analysis.coefficient_to_singular_value_ratios
    valid = _positive_finite(ratios)
    axes[1].semilogy(indices[valid], ratios[valid], marker=".")
    unavailable = int(np.count_nonzero(~valid))
    axes[1].set_title(
        "Coefficient-to-singular-value ratio"
        + (f" ({unavailable} non-finite/zero omitted)" if unavailable else "")
    )
    axes[1].set_ylabel(r"$|u_i^T b|/\sigma_i$")
    for axis in axes:
        axis.set_xlabel("Singular direction")
        axis.grid(alpha=0.25)
    if show:
        plt.show()
    return figure, axes


def plot_noise_amplification(
    singular_values: np.ndarray,
    analysis: NoiseAmplificationAnalysis,
    *,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Compare transformed noise with spectrum and inverse contributions."""
    values = np.asarray(singular_values)
    if values.shape != analysis.transformed_noise_coefficients.shape:
        raise ValueError("singular values and transformed noise must align")
    indices = np.arange(1, values.size + 1)
    coefficient_magnitudes = np.abs(analysis.transformed_noise_coefficients)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].semilogy(indices, values, label=r"$\sigma_i$")
    axes[0].semilogy(
        indices,
        coefficient_magnitudes,
        label=r"$|u_i^T\epsilon|$",
    )
    axes[0].set_title("Noise coefficients and singular values")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend()

    contributions = analysis.contribution_magnitudes
    valid = _positive_finite(contributions)
    axes[1].semilogy(indices[valid], contributions[valid], marker=".")
    unavailable = int(np.count_nonzero(~valid))
    axes[1].set_title(
        "Unregularized noise contribution magnitude"
        + (f" ({unavailable} non-finite/zero omitted)" if unavailable else "")
    )
    axes[1].set_ylabel(r"$|u_i^T\epsilon|/\sigma_i$")
    for axis in axes:
        axis.set_xlabel("Singular direction")
        axis.grid(alpha=0.25)
    if show:
        plt.show()
    return figure, axes


def plot_blur_sensitivity(
    results: Sequence[BlurSensitivityResult],
    *,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot normalized spectra and condition number versus Gaussian sigma."""
    if len(results) == 0:
        raise ValueError("results must not be empty")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for result in results:
        indices = np.arange(1, result.singular_values.size + 1)
        axes[0].semilogy(
            indices,
            result.normalized_singular_values,
            label=fr"$\sigma={result.sigma:g}$",
        )
    axes[0].set_title("Normalized singular-value spectra")
    axes[0].set_xlabel("Singular-value index")
    axes[0].set_ylabel(r"$\sigma_i/\sigma_1$")
    axes[0].legend()

    sigma_values = [result.sigma for result in results]
    condition_numbers = np.asarray(
        [result.condition_number for result in results],
        dtype=np.float64,
    )
    finite = np.isfinite(condition_numbers) & (condition_numbers > 0.0)
    axes[1].semilogy(
        np.asarray(sigma_values)[finite],
        condition_numbers[finite],
        marker="o",
    )
    if np.any(~finite):
        display_level = (
            float(np.max(condition_numbers[finite])) * 10.0
            if np.any(finite)
            else 1.0
        )
        for sigma in np.asarray(sigma_values)[~finite]:
            axes[1].scatter(sigma, display_level, marker="^", color="tab:red")
            axes[1].annotate(
                "∞",
                (sigma, display_level),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
            )
    axes[1].set_title("Condition number versus Gaussian blur strength")
    axes[1].set_xlabel(r"Gaussian $\sigma$ (pixels)")
    axes[1].set_ylabel(r"Numerical $\kappa_2(A)$")
    for axis in axes:
        axis.grid(alpha=0.25)
    if show:
        plt.show()
    return figure, axes


def plot_structured_full_comparison(
    singular_values_full: np.ndarray,
    singular_values_structured: np.ndarray,
    timing_summary: Mapping[str, float],
    *,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot spectrum agreement and repeated-timing median with spread."""
    full = np.asarray(singular_values_full)
    structured = np.asarray(singular_values_structured)
    if full.shape != structured.shape:
        raise ValueError("full and structured spectra must have the same shape")
    indices = np.arange(1, full.size + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].semilogy(indices, full, label="Full dense SVD")
    axes[0].semilogy(
        indices,
        structured,
        linestyle="--",
        label="Structured products",
    )
    axes[0].set_title("Singular-value agreement")
    axes[0].set_xlabel("Singular-value index")
    axes[0].set_ylabel(r"$\sigma_i$")
    axes[0].legend()

    labels = ["Full", "Structured"]
    medians = [
        timing_summary["full_median_seconds"],
        timing_summary["structured_median_seconds"],
    ]
    lower = [
        medians[0] - timing_summary["full_min_seconds"],
        medians[1] - timing_summary["structured_min_seconds"],
    ]
    upper = [
        timing_summary["full_max_seconds"] - medians[0],
        timing_summary["structured_max_seconds"] - medians[1],
    ]
    axes[1].bar(labels, medians, yerr=np.array([lower, upper]), capsize=5)
    axes[1].set_title("Repeated timing: median and min/max range")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(axis="y", alpha=0.25)
    if show:
        plt.show()
    return figure, axes
