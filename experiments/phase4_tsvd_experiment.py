"""Evaluate explicitly selected TSVD ranks on the shared Phase 4 problem."""

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
from regularization.tsvd import solve_tsvd  # noqa: E402
from utils.phase4_protocol import (  # noqa: E402
    PHASE4_TSVD_RANKS,
    create_phase4_problem,
)
from visualization.reconstruction import (  # noqa: E402
    plot_reconstruction_comparison,
)
from visualization.regularization import (  # noqa: E402
    plot_parameter_tradeoffs,
    plot_tsvd_filters,
)


CSV_FIELDS = REGULARIZATION_METRIC_FIELDS + (
    "numerical_rank",
    "rank_threshold",
)


def main() -> None:
    """Run the fixed TSVD rank sweep without selecting an optimal rank."""
    problem = create_phase4_problem()
    numerical_rank = problem.svd_analysis.numerical_rank
    records: list[dict[str, object]] = []
    reconstructions: dict[int, np.ndarray] = {}
    invalid_ranks: list[int] = []

    for retained_rank in PHASE4_TSVD_RANKS:
        if retained_rank > numerical_rank:
            invalid_ranks.append(retained_rank)
            continue
        result = solve_tsvd(
            problem.operator,
            problem.observation,
            problem.original.shape,
            retained_rank,
            svd_analysis=problem.svd_analysis,
        )
        record = evaluated_result_record(
            "TSVD",
            "retained_rank",
            retained_rank,
            result,
            problem.original,
            problem.operator,
            problem.observation,
            extra={
                "numerical_rank": numerical_rank,
                "rank_threshold": result.rank_threshold,
            },
        )
        records.append(record)
        reconstructions[retained_rank] = result.reconstruction_image
        np.save(
            DEFAULT_CONFIG.output_directories["images"]
            / f"phase4_tsvd_k_{retained_rank}_raw.npy",
            result.reconstruction_image,
        )

    if invalid_ranks:
        print(
            "Skipped explicitly requested ranks above numerical rank "
            f"{numerical_rank}: {invalid_ranks}"
        )

    csv_path = (
        DEFAULT_CONFIG.output_directories["metrics"] / "phase4_tsvd_results.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    representative_ranks = [
        rank for rank in (5, 20, 60, 100, 144) if rank in reconstructions
    ]
    comparison_figure, _ = plot_reconstruction_comparison(
        problem.original,
        problem.degradation.noisy_observation,
        {f"TSVD k={rank}": reconstructions[rank] for rank in representative_ranks},
        show=False,
    )
    comparison_path = (
        DEFAULT_CONFIG.output_directories["plots"]
        / "phase4_tsvd_reconstructions.png"
    )
    comparison_figure.savefig(comparison_path, dpi=150)
    plt.close(comparison_figure)

    tradeoff_figure, _ = plot_parameter_tradeoffs(
        records,
        "parameter_value",
        "Retained rank k",
        show=False,
    )
    tradeoff_path = (
        DEFAULT_CONFIG.output_directories["plots"] / "phase4_tsvd_tradeoffs.png"
    )
    tradeoff_figure.savefig(tradeoff_path, dpi=150)
    plt.close(tradeoff_figure)

    filter_ranks = [
        rank for rank in (5, 20, 60, 100, 144) if rank <= numerical_rank
    ]
    filter_figure, _ = plot_tsvd_filters(
        problem.svd_analysis.singular_values,
        filter_ranks,
        problem.svd_analysis.rank_threshold,
        show=False,
    )
    filter_path = (
        DEFAULT_CONFIG.output_directories["plots"] / "phase4_tsvd_filters.png"
    )
    filter_figure.savefig(filter_path, dpi=150)
    plt.close(filter_figure)

    print(f"Numerical rank: {numerical_rank}")
    print(f"Evaluated ranks: {[record['parameter_value'] for record in records]}")
    print(f"CSV: {csv_path}")
    print(f"Reconstructions: {comparison_path}")
    print(f"Tradeoffs: {tradeoff_path}")
    print(f"Filters: {filter_path}")
    for record in records:
        print(record)


if __name__ == "__main__":
    main()
