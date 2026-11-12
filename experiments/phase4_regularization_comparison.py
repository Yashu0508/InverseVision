"""Compare Phase 2 baselines with fixed Phase 4 regularization parameters."""

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG  # noqa: E402
from evaluation.experiment_records import (  # noqa: E402
    REGULARIZATION_METRIC_FIELDS,
    evaluated_result_record,
)
from numerical_methods.baseline_solvers import (  # noqa: E402
    solve_direct_inverse,
    solve_normal_equations,
    solve_qr_least_squares,
)
from regularization.tikhonov import solve_tikhonov_svd  # noqa: E402
from regularization.tsvd import solve_tsvd  # noqa: E402
from utils.phase4_protocol import (  # noqa: E402
    PHASE4_TIKHONOV_PARAMETERS,
    PHASE4_TSVD_RANKS,
    create_phase4_problem,
)
from visualization.reconstruction import (  # noqa: E402
    plot_reconstruction_comparison,
)


def main() -> None:
    """Evaluate all methods against one unchanged original and observation."""
    problem = create_phase4_problem()
    records: list[dict[str, object]] = []
    reconstructions: dict[str, object] = {}

    baseline_methods = (
        ("Direct inverse", solve_direct_inverse),
        ("QR least squares", solve_qr_least_squares),
        ("Normal equations", solve_normal_equations),
    )
    for name, solver in baseline_methods:
        result = solver(
            problem.operator,
            problem.observation,
            problem.original.shape,
        )
        records.append(
            evaluated_result_record(
                name,
                "none",
                "",
                result,
                problem.original,
                problem.operator,
                problem.observation,
            )
        )
        reconstructions[name] = result.reconstruction_image

    tsvd_results: dict[int, object] = {}
    for retained_rank in PHASE4_TSVD_RANKS:
        if retained_rank > problem.svd_analysis.numerical_rank:
            continue
        result = solve_tsvd(
            problem.operator,
            problem.observation,
            problem.original.shape,
            retained_rank,
            svd_analysis=problem.svd_analysis,
        )
        records.append(
            evaluated_result_record(
                "TSVD",
                "retained_rank",
                retained_rank,
                result,
                problem.original,
                problem.operator,
                problem.observation,
            )
        )
        tsvd_results[retained_rank] = result

    tikhonov_results: dict[float, object] = {}
    for parameter in PHASE4_TIKHONOV_PARAMETERS:
        result = solve_tikhonov_svd(
            problem.operator,
            problem.observation,
            problem.original.shape,
            parameter,
            svd_analysis=problem.svd_analysis,
        )
        records.append(
            evaluated_result_record(
                "Tikhonov SVD",
                "lambda",
                parameter,
                result,
                problem.original,
                problem.operator,
                problem.observation,
            )
        )
        tikhonov_results[parameter] = result

    csv_path = (
        DEFAULT_CONFIG.output_directories["metrics"]
        / "phase4_regularization_comparison.csv"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=REGULARIZATION_METRIC_FIELDS,
        )
        writer.writeheader()
        writer.writerows(records)

    for retained_rank in (20, 80):
        if retained_rank in tsvd_results:
            reconstructions[f"TSVD k={retained_rank}"] = tsvd_results[
                retained_rank
            ].reconstruction_image
    for parameter in (1e-1, 1e-3):
        reconstructions[fr"Tikhonov $\lambda={parameter:g}$"] = (
            tikhonov_results[parameter].reconstruction_image
        )

    comparison_figure, _ = plot_reconstruction_comparison(
        problem.original,
        problem.degradation.noisy_observation,
        reconstructions,
        show=False,
    )
    comparison_path = (
        DEFAULT_CONFIG.output_directories["plots"]
        / "phase4_regularization_comparison.png"
    )
    comparison_figure.savefig(comparison_path, dpi=150)
    plt.close(comparison_figure)

    tradeoff_figure, axis = plt.subplots(
        figsize=(9, 6),
        constrained_layout=True,
    )
    colors = {
        "Direct inverse": "tab:red",
        "QR least squares": "tab:red",
        "Normal equations": "tab:red",
        "TSVD": "tab:blue",
        "Tikhonov SVD": "tab:green",
    }
    for record in records:
        axis.scatter(
            float(record["residual_norm"]),
            float(record["relative_reconstruction_error"]),
            color=colors[str(record["method"])],
            alpha=0.8,
        )
    for method, color in colors.items():
        axis.scatter([], [], color=color, label=method)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Data residual $\|A\hat{x}-b\|_2$")
    axis.set_ylabel("Relative reconstruction error")
    axis.set_title("Data fit versus reconstruction accuracy")
    axis.grid(alpha=0.25)
    axis.legend()
    tradeoff_path = (
        DEFAULT_CONFIG.output_directories["plots"]
        / "phase4_regularization_comparison_tradeoff.png"
    )
    tradeoff_figure.savefig(tradeoff_path, dpi=150)
    plt.close(tradeoff_figure)

    print(f"CSV: {csv_path}")
    print(f"Representative reconstructions: {comparison_path}")
    print(f"Tradeoff plot: {tradeoff_path}")
    for record in records:
        print(record)


if __name__ == "__main__":
    main()
