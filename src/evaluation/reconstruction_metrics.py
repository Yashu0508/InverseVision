"""Metrics separating reconstruction accuracy from data residual accuracy."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from skimage.metrics import structural_similarity


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Evaluation metrics for one raw, unclipped reconstruction."""

    relative_reconstruction_error: float
    mean_squared_error: float
    peak_signal_to_noise_ratio: float
    structural_similarity_index: float
    residual_norm: float
    relative_residual: float


def _validated_pair(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    reference_array = np.asarray(reference)
    reconstruction_array = np.asarray(reconstruction)
    if reference_array.shape != reconstruction_array.shape:
        raise ValueError("reference and reconstruction must have the same shape")
    if reference_array.size == 0:
        raise ValueError("images must be non-empty")
    if not np.issubdtype(reference_array.dtype, np.number) or not np.issubdtype(
        reconstruction_array.dtype,
        np.number,
    ):
        raise TypeError("images must have numerical dtypes")
    if np.iscomplexobj(reference_array) or np.iscomplexobj(reconstruction_array):
        raise TypeError("complex-valued images are not supported")
    if not np.all(np.isfinite(reference_array)) or not np.all(
        np.isfinite(reconstruction_array)
    ):
        raise ValueError("images must contain only finite values")
    return (
        reference_array.astype(np.float64, copy=False),
        reconstruction_array.astype(np.float64, copy=False),
    )


def relative_reconstruction_error(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> float:
    """Return ``||reconstruction-reference||_2 / ||reference||_2``."""
    truth, estimate = _validated_pair(reference, reconstruction)
    error_norm = float(np.linalg.norm(estimate.ravel() - truth.ravel()))
    reference_norm = float(np.linalg.norm(truth.ravel()))
    if reference_norm == 0.0:
        return 0.0 if error_norm == 0.0 else float("inf")
    return error_norm / reference_norm


def mean_squared_error(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> float:
    """Return the mean squared pixel error without clipping inputs."""
    truth, estimate = _validated_pair(reference, reconstruction)
    return float(np.mean((estimate - truth) ** 2))


def peak_signal_to_noise_ratio(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> float:
    """Return normalized-image PSNR, using a fixed data range of one."""
    mse = mean_squared_error(reference, reconstruction)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def structural_similarity_index(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> float:
    """Return SSIM for two-dimensional normalized-image data.

    The reference must lie in ``[0, 1]``. The raw reconstruction is not
    clipped. For small images, the largest valid odd window no greater than
    seven is selected.
    """
    truth, estimate = _validated_pair(reference, reconstruction)
    if truth.ndim != 2:
        raise ValueError("SSIM requires two-dimensional grayscale images")
    if np.min(truth) < 0.0 or np.max(truth) > 1.0:
        raise ValueError("SSIM reference intensities must lie in [0, 1]")
    window_size = min(7, *truth.shape)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        raise ValueError("SSIM requires both image dimensions to be at least 3")
    return float(
        structural_similarity(
            truth,
            estimate,
            data_range=1.0,
            win_size=window_size,
        )
    )


def residual_metrics(
    matrix: np.ndarray | sparse.spmatrix,
    reconstruction: np.ndarray,
    observation: np.ndarray,
) -> tuple[float, float]:
    """Return residual norm and relative residual for ``b - A @ x``."""
    if sparse.issparse(matrix):
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix.data)):
            raise ValueError("matrix must be two-dimensional and finite")
    else:
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must be two-dimensional and finite")
    estimate = np.asarray(reconstruction)
    observed = np.asarray(observation)
    if estimate.ndim != 1 or observed.ndim != 1:
        raise ValueError("reconstruction and observation must be vectors")
    if matrix.shape != (observed.size, estimate.size):
        raise ValueError("matrix shape is incompatible with the supplied vectors")
    if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(observed)):
        raise ValueError("vectors must contain only finite values")

    residual = observed - matrix @ estimate
    residual_norm = float(np.linalg.norm(residual))
    observation_norm = float(np.linalg.norm(observed))
    relative = (
        residual_norm / observation_norm
        if observation_norm > 0.0
        else (0.0 if residual_norm == 0.0 else float("inf"))
    )
    return residual_norm, relative


def evaluate_reconstruction(
    reference_image: np.ndarray,
    reconstruction_image: np.ndarray,
    matrix: np.ndarray | sparse.spmatrix,
    observation: np.ndarray,
) -> ReconstructionMetrics:
    """Evaluate raw reconstruction accuracy and observation residual accuracy."""
    reconstruction_vector = np.asarray(reconstruction_image).ravel(order="C")
    residual_norm, relative_residual = residual_metrics(
        matrix,
        reconstruction_vector,
        observation,
    )
    return ReconstructionMetrics(
        relative_reconstruction_error=relative_reconstruction_error(
            reference_image,
            reconstruction_image,
        ),
        mean_squared_error=mean_squared_error(
            reference_image,
            reconstruction_image,
        ),
        peak_signal_to_noise_ratio=peak_signal_to_noise_ratio(
            reference_image,
            reconstruction_image,
        ),
        structural_similarity_index=structural_similarity_index(
            reference_image,
            reconstruction_image,
        ),
        residual_norm=residual_norm,
        relative_residual=relative_residual,
    )
