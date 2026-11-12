"""CSV-ready records for solver results evaluated on known ground truth."""

from collections.abc import Mapping

import numpy as np
from scipy import sparse

from evaluation.reconstruction_metrics import evaluate_reconstruction


REGULARIZATION_METRIC_FIELDS = (
    "method",
    "status",
    "parameter_name",
    "parameter_value",
    "relative_reconstruction_error",
    "mean_squared_error",
    "peak_signal_to_noise_ratio",
    "structural_similarity_index",
    "residual_norm",
    "relative_residual",
    "execution_time_seconds",
    "diagnostic",
)


def evaluated_result_record(
    method: str,
    parameter_name: str,
    parameter_value: int | float | str,
    result: object,
    reference_image: np.ndarray,
    matrix: np.ndarray | sparse.spmatrix,
    observation: np.ndarray,
    *,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate a compatible solver result and return a stable record."""
    reconstruction_image = getattr(result, "reconstruction_image")
    metrics = evaluate_reconstruction(
        reference_image,
        reconstruction_image,
        matrix,
        observation,
    )
    record: dict[str, object] = {
        "method": method,
        "status": getattr(result, "status"),
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "relative_reconstruction_error": metrics.relative_reconstruction_error,
        "mean_squared_error": metrics.mean_squared_error,
        "peak_signal_to_noise_ratio": metrics.peak_signal_to_noise_ratio,
        "structural_similarity_index": metrics.structural_similarity_index,
        "residual_norm": metrics.residual_norm,
        "relative_residual": metrics.relative_residual,
        "execution_time_seconds": getattr(result, "execution_time_seconds"),
        "diagnostic": getattr(result, "diagnostic"),
    }
    if extra:
        record.update(extra)
    return record
