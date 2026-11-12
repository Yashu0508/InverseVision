"""Evaluate explicitly selected Tikhonov parameters on one shared problem."""

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG  # noqa: E402
from evaluation.experiment_records import (  # noqa: E402
    REGULARIZATION_METRIC_FIELDS,
    evaluated_result_record,
)
from regularization.tikhonov import (  # noqa: E402
    solve_tikhonov_augmented,
    solve_tikhonov_svd,
)
from utils.phase4_protocol import (  # noqa: E402
    PHASE4_TIKHONOV_PARAMETERS,
    create_phase4_problem,
)
from visualization.reconstruction import (  # noqa: E402
    plot_reconstruction_comparison,
)
from visualization.regularization import (  # noqa: E402
    plot_parameter_tradeoffs,
    plot_tikhonov_filters,
)


CSV_FIELDS = REGULARIZATION_METRIC_FIELDS + (
    "augmented_max_absolute_difference",
    "augmented_relative_difference",
    "augmented_execution_time_seconds",
)


def main() -> None:
    """Run the fixed lambda sweep without selecting an optimal parameter."""
    problem = create_phase4_problem()
    records: list[dict[str, object]] = []
    reconstructions: dict[float, np.ndarray] = {}

    for parameter in PHASE4_TIKHONOV_PARAMETERS:
        svd_result = solve_tikhonov_svd(
            problem.operator,
            problem.observation,
            problem.original.shape,
            parameter,
            svd_analysis=problem.svd_analysis,
        )
        augmented_result = solve_tikhonov_augmented(
            problem.operator,
            problem.observation,
            problem.original.shape,
            parameter,
        )
        maximum_difference = float(
            np.max(
                np.abs(
                    svd_result.reconstruction_vector
                    - augmented_result.reconstruction_vector
                )
            )
        )
        relative_difference = float(
            np.linalg.norm(
                svd_result.reconstruction_vector
                - augmented_result.reconstruction_vector
            )
            / np.linalg.norm(svd_result.reconstruction_vector)
        )
        record = evaluated_result_record(
            "Tikhonov SVD",
            "lambda",
            parameter,
            svd_result,
            problem.original,
            problem.operator,
            problem.observation,
            extra={
                "augmented_max_absolute_difference": maximum_difference,
                "augmented_relative_difference": relative_difference,
                "augmented_execution_time_seconds": (
                    augmented_result.execution_time_seconds
                ),
            },
        )
        records.append(record)
        reconstructions[parameter] = svd_result.reconstruction_image
        label = f"{parameter:.0e}".replace("+", "")
        np.save(
            DEFAULT_CONFIG.output_directories["images"]
            / f"phase4_tikhonov_lambda_{label}_raw.npy",
            svd_result.reconstruction_image,
        )

    csv_path = (
        DEFAULT_CONFIG.output_directories["metrics"]
        / "phase4_tikhonov_results.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    comparison_figure, _ = plot_reconstruction_comparison(
        problem.original,
        problem.degradation.noisy_observation,
        {
            fr"Tikhonov $\lambda={parameter:g}$": reconstructions[parameter]
            for parameter in PHASE4_TIKHONOV_PARAMETERS
        },
        show=False,
    )
    comparison_path = (
        DEFAULT_CONFIG.output_directories["plots"]
        / "phase4_tikhonov_reconstructions.png"
    )
    comparison_figure.savefig(comparison_path, dpi=150)
    plt.close(comparison_figure)

    tradeoff_figure, _ = plot_parameter_tradeoffs(
        records,
        "parameter_value",
        r"Regularization parameter $\lambda$",
        logarithmic_parameter=True,
        show=False,
    )
    tradeoff_path = (
        DEFAULT_CONFIG.output_directories["plots"]
        / "phase4_tikhonov_tradeoffs.png"
    )
    tradeoff_figure.savefig(tradeoff_path, dpi=150)
    plt.close(tradeoff_figure)

    filter_figure, _ = plot_tikhonov_filters(
        problem.svd_analysis.singular_values,
        PHASE4_TIKHONOV_PARAMETERS,
        show=False,
    )
    filter_path = (
        DEFAULT_CONFIG.output_directories["plots"]
        / "phase4_tikhonov_filters.png"
    )
    filter_figure.savefig(filter_path, dpi=150)
    plt.close(filter_figure)

    print(f"CSV: {csv_path}")
    print(f"Reconstructions: {comparison_path}")
    print(f"Tradeoffs: {tradeoff_path}")
    print(f"Filters: {filter_path}")
    for record in records:
        print(record)


if __name__ == "__main__":
    main()
