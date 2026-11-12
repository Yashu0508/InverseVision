"""Compare repeated full and separable-structure SVD computations."""

import csv
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.svd_analysis import analyze_structured_svd, analyze_svd  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402
from forward_model.gaussian import (  # noqa: E402
    gaussian_blur_matrix_1d,
    gaussian_blur_operator_2d,
)
from visualization.svd_analysis import (  # noqa: E402
    plot_structured_full_comparison,
)


IMAGE_SHAPE = (8, 8)
RANK_RTOL = 1e-12
TIMING_REPEATS = 7


def _time_repeated(function, repeats: int) -> np.ndarray:
    timings = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        start = perf_counter()
        function()
        timings[index] = perf_counter() - start
    return timings


def main() -> None:
    """Compare spectrum agreement, repeated timing, and explicit storage."""
    vertical = gaussian_blur_matrix_1d(
        IMAGE_SHAPE[0],
        DEFAULT_CONFIG.gaussian_blur_sigma,
        DEFAULT_CONFIG.gaussian_kernel_size,
    )
    horizontal = gaussian_blur_matrix_1d(
        IMAGE_SHAPE[1],
        DEFAULT_CONFIG.gaussian_blur_sigma,
        DEFAULT_CONFIG.gaussian_kernel_size,
    )
    dense_operator = gaussian_blur_operator_2d(
        IMAGE_SHAPE,
        DEFAULT_CONFIG.gaussian_blur_sigma,
        DEFAULT_CONFIG.gaussian_kernel_size,
    ).toarray()

    def full_callable():
        return analyze_svd(
            dense_operator,
            relative_tolerance=RANK_RTOL,
        )

    def structured_callable():
        return analyze_structured_svd(
            vertical,
            horizontal,
            relative_tolerance=RANK_RTOL,
        )
    full_callable()
    structured_callable()
    full_timings = _time_repeated(full_callable, TIMING_REPEATS)
    structured_timings = _time_repeated(structured_callable, TIMING_REPEATS)
    full = full_callable()
    structured = structured_callable()

    absolute_difference = np.abs(
        full.singular_values - structured.singular_values
    )
    relative_difference = absolute_difference / np.maximum(
        full.singular_values,
        np.finfo(np.float64).tiny,
    )
    timing_summary = {
        "full_min_seconds": float(np.min(full_timings)),
        "full_median_seconds": float(np.median(full_timings)),
        "full_max_seconds": float(np.max(full_timings)),
        "structured_min_seconds": float(np.min(structured_timings)),
        "structured_median_seconds": float(np.median(structured_timings)),
        "structured_max_seconds": float(np.max(structured_timings)),
    }
    summary = {
        "rows": IMAGE_SHAPE[0],
        "columns": IMAGE_SHAPE[1],
        "timing_repeats": TIMING_REPEATS,
        "rank_relative_tolerance": RANK_RTOL,
        "maximum_absolute_spectrum_difference": float(
            np.max(absolute_difference)
        ),
        "maximum_relative_spectrum_difference": float(
            np.max(relative_difference)
        ),
        "dense_operator_input_bytes": dense_operator.nbytes,
        "structured_factor_input_bytes": vertical.nbytes + horizontal.nbytes,
        "full_analysis_output_bytes": (
            full.left_singular_vectors.nbytes
            + full.singular_values.nbytes
            + full.right_singular_vectors_transpose.nbytes
        ),
        "structured_spectrum_output_bytes": (
            structured.vertical_singular_values.nbytes
            + structured.horizontal_singular_values.nbytes
            + structured.singular_values.nbytes
        ),
        **timing_summary,
    }
    summary_path = (
        DEFAULT_CONFIG.output_directories["metrics"]
        / "phase3_structured_vs_full_svd.csv"
    )
    with summary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=tuple(summary))
        writer.writeheader()
        writer.writerow(summary)

    figure, _ = plot_structured_full_comparison(
        full.singular_values,
        structured.singular_values,
        timing_summary,
        show=False,
    )
    plot_path = (
        DEFAULT_CONFIG.output_directories["plots"]
        / "phase3_structured_vs_full_svd.png"
    )
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Summary: {summary_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
