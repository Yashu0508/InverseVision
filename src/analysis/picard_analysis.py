"""Discrete Picard coefficient analysis without truncation or reconstruction."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis.svd_analysis import SVDAnalysis


@dataclass(frozen=True)
class PicardAnalysis:
    """Ordered SVD coefficients and safe coefficient/singular-value ratios."""

    singular_values: NDArray[np.float64]
    expansion_coefficients: NDArray[np.float64]
    coefficient_magnitudes: NDArray[np.float64]
    coefficient_to_singular_value_ratios: NDArray[np.float64]
    safe_denominator_mask: NDArray[np.bool_]
    denominator_threshold: float


def _safe_magnitude_ratios(
    magnitudes: NDArray[np.float64],
    singular_values: NDArray[np.float64],
    threshold: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    safe = singular_values > threshold
    ratios = np.empty_like(magnitudes)
    np.divide(magnitudes, singular_values, out=ratios, where=safe)
    unsafe = ~safe
    ratios[unsafe & (magnitudes > 0.0)] = np.inf
    ratios[unsafe & (magnitudes == 0.0)] = 0.0
    return ratios, safe


def analyze_picard(
    svd_analysis: SVDAnalysis,
    observation: np.ndarray,
    *,
    denominator_threshold: float | None = None,
) -> PicardAnalysis:
    """Compute ``u_i.T @ b`` and safe magnitude-to-spectrum ratios.

    By default, singular values at or below the SVD rank threshold are treated
    as unsafe denominators. Their ratios are reported as infinity for nonzero
    coefficients and zero for a simultaneous zero numerator. No truncation
    index is selected and no reconstruction is formed.
    """
    observed = np.asarray(observation)
    if observed.ndim != 1:
        raise ValueError("observation must be a one-dimensional vector")
    if observed.size != svd_analysis.matrix_shape[0]:
        raise ValueError("observation length must equal the matrix row count")
    if not np.issubdtype(observed.dtype, np.number) or np.iscomplexobj(observed):
        raise TypeError("observation must be a real numerical vector")
    if not np.all(np.isfinite(observed)):
        raise ValueError("observation must contain only finite values")

    threshold = (
        svd_analysis.rank_threshold
        if denominator_threshold is None
        else float(denominator_threshold)
    )
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("denominator_threshold must be finite and non-negative")

    coefficients = (
        svd_analysis.left_singular_vectors.T
        @ observed.astype(np.float64, copy=False)
    )
    magnitudes = np.abs(coefficients)
    ratios, safe = _safe_magnitude_ratios(
        magnitudes,
        svd_analysis.singular_values,
        threshold,
    )
    return PicardAnalysis(
        singular_values=svd_analysis.singular_values.copy(),
        expansion_coefficients=coefficients,
        coefficient_magnitudes=magnitudes,
        coefficient_to_singular_value_ratios=ratios,
        safe_denominator_mask=safe,
        denominator_threshold=threshold,
    )
