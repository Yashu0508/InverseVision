"""Demonstrate full SVD and conditioning diagnostics on a small blur matrix."""

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.svd_analysis import analyze_svd, singular_value_decay  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402
from forward_model.gaussian import gaussian_blur_operator_2d  # noqa: E402
from visualization.svd_analysis import plot_singular_value_analysis  # noqa: E402


IMAGE_SHAPE = (8, 8)
RANK_RTOL = 1e-12


def main() -> None:
    """Compute, tabulate, and plot one explicit small-operator SVD."""
    operator = gaussian_blur_operator_2d(
        IMAGE_SHAPE,
        sigma=DEFAULT_CONFIG.gaussian_blur_sigma,
        kernel_size=DEFAULT_CONFIG.gaussian_kernel_size,
    )
    analysis = analyze_svd(operator, relative_tolerance=RANK_RTOL)
    decay = singular_value_decay(analysis.singular_values)

    spectrum_path = (
        DEFAULT_CONFIG.output_directories["metrics"] / "phase3_svd_spectrum.csv"
    )
    with spectrum_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "index",
                "singular_value",
                "normalized_singular_value",
                "cumulative_energy",
            ),
        )
        writer.writeheader()
        for index, (value, normalized, energy) in enumerate(
            zip(
                analysis.singular_values,
                decay.normalized_singular_values,
                decay.cumulative_energy,
                strict=True,
            ),
            start=1,
        ):
            writer.writerow(
                {
                    "index": index,
                    "singular_value": value,
                    "normalized_singular_value": normalized,
                    "cumulative_energy": energy,
                }
            )

    summary_path = (
        DEFAULT_CONFIG.output_directories["metrics"] / "phase3_svd_summary.csv"
    )
    with summary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "rows",
                "columns",
                "relative_tolerance",
                "rank_threshold",
                "numerical_rank",
                "total_singular_values",
                "largest_singular_value",
                "smallest_singular_value",
                "condition_number",
                "diagnostic",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "rows": operator.shape[0],
                "columns": operator.shape[1],
                "relative_tolerance": analysis.relative_tolerance,
                "rank_threshold": analysis.rank_threshold,
                "numerical_rank": analysis.numerical_rank,
                "total_singular_values": analysis.singular_values.size,
                "largest_singular_value": analysis.largest_singular_value,
                "smallest_singular_value": analysis.smallest_singular_value,
                "condition_number": analysis.condition_number,
                "diagnostic": analysis.diagnostic,
            }
        )

    figure, _ = plot_singular_value_analysis(analysis, show=False)
    plot_path = (
        DEFAULT_CONFIG.output_directories["plots"] / "phase3_svd_analysis.png"
    )
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    print(f"Image shape: {IMAGE_SHAPE}")
    print(f"Numerical rank: {analysis.numerical_rank}/{analysis.singular_values.size}")
    print(f"Condition number: {analysis.condition_number}")
    print(f"Smallest singular value: {analysis.smallest_singular_value}")
    print(f"Rank threshold: {analysis.rank_threshold}")
    print(f"Summary: {summary_path}")
    print(f"Spectrum: {spectrum_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
