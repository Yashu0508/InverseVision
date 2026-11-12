"""Descriptive summaries and transparent instability flags for experiments."""

from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np


SUMMARY_METRICS = (
    "selected_parameter",
    "relative_reconstruction_error",
    "mean_squared_error",
    "peak_signal_to_noise_ratio",
    "structural_similarity_index",
    "residual_norm",
    "relative_residual",
    "reconstruction_norm",
    "execution_time_seconds",
    "unstable_solution",
)


def classify_unstable_solution(
    relative_error: float,
    reconstruction_norm: float,
    reference_norm: float,
    *,
    maximum_relative_error: float = 10.0,
    maximum_norm_ratio: float = 10.0,
) -> bool:
    """Flag nonfinite or extreme solutions using explicit diagnostic cutoffs.

    The thresholds are operational flags for this study, not universal
    scientific definitions of instability and not selection criteria.
    """
    values = (
        relative_error, reconstruction_norm, reference_norm,
        maximum_relative_error, maximum_norm_ratio,
    )
    if not all(np.isfinite(value) for value in values) or reference_norm <= 0:
        return True
    return bool(
        relative_error >= maximum_relative_error
        or reconstruction_norm / reference_norm >= maximum_norm_ratio
    )


def aggregate_records(
    records: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate finite numeric metrics by axis, condition, method and criterion.

    Sample standard deviation (ddof=1) is reported for at least two values;
    it is NaN for a single observation rather than being presented as zero.
    Failed runs and NaN metrics are excluded from numeric aggregates but remain
    available in the raw records.
    """
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["sensitivity_axis"]),
            str(record["aggregation_condition"]),
            str(record["method"]),
            str(record["criterion"]),
        )
        grouped[key].append(record)

    summaries: list[dict[str, object]] = []
    for key in sorted(grouped):
        axis, condition, method, criterion = key
        for metric in SUMMARY_METRICS:
            values = np.asarray([
                float(row[metric])
                for row in grouped[key]
                if row.get("status") == "success"
                and row.get(metric) is not None
                and np.isfinite(float(row[metric]))
            ], dtype=np.float64)
            if not values.size:
                continue
            summaries.append({
                "sensitivity_axis": axis,
                "aggregation_condition": condition,
                "method": method,
                "criterion": criterion,
                "metric": metric,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            })
    return summaries
