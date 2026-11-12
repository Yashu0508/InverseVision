"""Analyze singular-value sensitivity to Gaussian blur strength."""

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.blur_sensitivity import (  # noqa: E402
    BLUR_SENSITIVITY_FIELDS,
    analyze_blur_strengths,
    blur_sensitivity_records,
)
from config import DEFAULT_CONFIG  # noqa: E402
from visualization.svd_analysis import plot_blur_sensitivity  # noqa: E402


IMAGE_SHAPE = (16, 16)
SIGMA_VALUES = (0.5, 1.0, 1.5, 2.0, 3.0)
RANK_RTOL = 1e-12


def main() -> None:
    """Run a fixed-convention structured spectrum comparison over sigma."""
    results = analyze_blur_strengths(
        SIGMA_VALUES,
        IMAGE_SHAPE,
        DEFAULT_CONFIG.gaussian_kernel_size,
        RANK_RTOL,
    )
    summary_path = (
        DEFAULT_CONFIG.output_directories["metrics"]
        / "phase3_blur_sensitivity.csv"
    )
    with summary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=BLUR_SENSITIVITY_FIELDS,
        )
        writer.writeheader()
        writer.writerows(blur_sensitivity_records(results))

    figure, _ = plot_blur_sensitivity(results, show=False)
    plot_path = (
        DEFAULT_CONFIG.output_directories["plots"]
        / "phase3_blur_sensitivity.png"
    )
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    print(f"Image shape: {IMAGE_SHAPE}")
    print(f"Rank relative tolerance: {RANK_RTOL}")
    for result in results:
        print(
            f"sigma={result.sigma:g}: rank={result.numerical_rank}/"
            f"{result.total_singular_values}, "
            f"condition={result.condition_number}, "
            f"smallest={result.smallest_singular_value}"
        )
    print(f"Summary: {summary_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
