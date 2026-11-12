"""Plots for TSVD and Tikhonov parameter studies."""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from regularization.tikhonov import tikhonov_filter_factors
from regularization.tsvd import tsvd_filter_factors


def plot_parameter_tradeoffs(
    records: Sequence[Mapping[str, object]],
    parameter_field: str,
    parameter_label: str,
    *,
    logarithmic_parameter: bool = False,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot reconstruction error and residual norm against a parameter."""
    if len(records) == 0:
        raise ValueError("records must not be empty")
    parameters = np.asarray(
        [float(record[parameter_field]) for record in records],
        dtype=np.float64,
    )
    errors = np.asarray(
        [float(record["relative_reconstruction_error"]) for record in records]
    )
    residuals = np.asarray(
        [float(record["residual_norm"]) for record in records]
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(parameters, errors, marker="o")
    axes[0].set_title("Relative reconstruction error")
    axes[0].set_ylabel(r"$\|\hat{x}-x\|_2/\|x\|_2$")
    axes[1].plot(parameters, residuals, marker="o")
    axes[1].set_title("Data residual norm")
    axes[1].set_ylabel(r"$\|A\hat{x}-b\|_2$")
    for axis in axes:
        axis.set_xlabel(parameter_label)
        axis.grid(alpha=0.25)
        if logarithmic_parameter:
            axis.set_xscale("log")
            axis.invert_xaxis()
    if np.all(errors > 0.0):
        axes[0].set_yscale("log")
    if np.all(residuals > 0.0):
        axes[1].set_yscale("log")
    if show:
        plt.show()
    return figure, axes


def plot_tsvd_filters(
    singular_values: np.ndarray,
    retained_ranks: Sequence[int],
    rank_threshold: float,
    *,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot TSVD inverse filters and dimensionless retention filters."""
    values = np.asarray(singular_values, dtype=np.float64)
    indices = np.arange(1, values.size + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for retained_rank in retained_ranks:
        factors = tsvd_filter_factors(
            values,
            retained_rank,
            denominator_threshold=rank_threshold,
        )
        positive = factors.inverse_filter > 0.0
        axes[0].semilogy(
            indices[positive],
            factors.inverse_filter[positive],
            marker=".",
            label=f"k={retained_rank}",
        )
        axes[1].step(
            indices,
            factors.retention_filter,
            where="mid",
            label=f"k={retained_rank}",
        )
    axes[0].set_title("TSVD inverse filter")
    axes[0].set_ylabel(r"$f_i^{(k)}$")
    axes[1].set_title("TSVD dimensionless retention filter")
    axes[1].set_ylabel(r"$\phi_i^{(k)}$")
    axes[1].set_ylim(-0.05, 1.05)
    for axis in axes:
        axis.set_xlabel("Singular direction")
        axis.grid(alpha=0.25)
        axis.legend()
    if show:
        plt.show()
    return figure, axes


def plot_tikhonov_filters(
    singular_values: np.ndarray,
    regularization_parameters: Sequence[float],
    *,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot Tikhonov inverse and dimensionless filters for fixed lambdas."""
    values = np.asarray(singular_values, dtype=np.float64)
    indices = np.arange(1, values.size + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for parameter in regularization_parameters:
        factors = tikhonov_filter_factors(values, parameter)
        axes[0].semilogy(
            indices,
            factors.inverse_filter,
            label=fr"$\lambda={parameter:g}$",
        )
        axes[1].semilogy(
            indices,
            factors.retention_filter,
            label=fr"$\lambda={parameter:g}$",
        )
    axes[0].set_title("Tikhonov inverse filter")
    axes[0].set_ylabel(r"$f_i^{(\lambda)}$")
    axes[1].set_title("Tikhonov dimensionless retention filter")
    axes[1].set_ylabel(r"$\phi_i^{(\lambda)}$")
    for axis in axes:
        axis.set_xlabel("Singular direction")
        axis.grid(alpha=0.25)
        axis.legend()
    if show:
        plt.show()
    return figure, axes
