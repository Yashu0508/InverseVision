"""Compare observation-only automatic parameter selection for Phase 5."""

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.reconstruction_metrics import evaluate_reconstruction  # noqa: E402
from regularization.parameter_selection import (  # noqa: E402
    select_parameter,
    tikhonov_candidate_lambdas,
    tsvd_candidate_ranks,
)
from regularization.tikhonov import solve_tikhonov_svd  # noqa: E402
from regularization.tsvd import solve_tsvd  # noqa: E402
from utils.phase4_protocol import (
    PHASE4_NOISE_STANDARD_DEVIATION,
    PHASE4_RANK_RELATIVE_TOLERANCE,
    create_phase4_problem,
)  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
PLOTS = ROOT / "results" / "plots"
IMAGES = ROOT / "results" / "images"
CRITERIA = ("l_curve", "gcv", "discrepancy")
DISCREPANCY_FACTOR = 1.05


def main() -> None:
    for directory in (METRICS, PLOTS, IMAGES):
        directory.mkdir(parents=True, exist_ok=True)
    problem = create_phase4_problem()
    matrix, observation, svd = problem.operator, problem.observation, problem.svd_analysis
    rank_candidates = tsvd_candidate_ranks(svd)
    lambda_candidates = tikhonov_candidate_lambdas(svd)

    selectors = []
    for method, candidates in (("TSVD", rank_candidates), ("Tikhonov", lambda_candidates)):
        for criterion in CRITERIA:
            selectors.append(select_parameter(
                method, criterion, matrix, observation, svd, candidates,
                noise_standard_deviation=PHASE4_NOISE_STANDARD_DEVIATION,
                discrepancy_factor=DISCREPANCY_FACTOR,
            ))

    # Ground truth is first accessed after all six selections are finalized.
    summary_rows = []
    reconstructions = {}
    for result in selectors:
        if result.method == "TSVD":
            solved = solve_tsvd(
                matrix, observation, problem.original.shape,
                int(result.selected_parameter),
                relative_tolerance=PHASE4_RANK_RELATIVE_TOLERANCE,
                svd_analysis=svd,
            )
        else:
            solved = solve_tikhonov_svd(
                matrix, observation, problem.original.shape,
                result.selected_parameter,
                relative_tolerance=PHASE4_RANK_RELATIVE_TOLERANCE,
                svd_analysis=svd,
            )
        reconstructions[(result.method, result.criterion)] = solved.reconstruction_image
        metrics = evaluate_reconstruction(
            problem.original, solved.reconstruction_image, matrix, observation
        )
        summary_rows.append({
            "method": result.method,
            "criterion": result.criterion,
            "selected_parameter": result.selected_parameter,
            "selected_index": result.selected_index,
            "residual_norm": metrics.residual_norm,
            "relative_residual": metrics.relative_residual,
            "relative_reconstruction_error": metrics.relative_reconstruction_error,
            "mse": metrics.mean_squared_error,
            "psnr_db": metrics.peak_signal_to_noise_ratio,
            "ssim": metrics.structural_similarity_index,
            "discrepancy_factor": DISCREPANCY_FACTOR,
            "noise_standard_deviation": PHASE4_NOISE_STANDARD_DEVIATION,
            "seed": 42,
            "image_shape": "12x12",
            "sigma": 1.5,
            "kernel_size": 9,
            "relative_tolerance": PHASE4_RANK_RELATIVE_TOLERANCE,
        })
        np.save(IMAGES / f"phase5_{result.method.lower()}_{result.criterion}_raw.npy", solved.reconstruction_image)

    with (METRICS / "phase5_selection_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    curve_path = METRICS / "phase5_selection_curves.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "method", "candidate_index", "parameter", "residual_norm",
            "solution_norm", "effective_degrees_of_freedom", "gcv_numerator",
            "gcv_denominator", "lcurve_curvature", "gcv",
            "discrepancy_absolute_error",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method, candidates in (("TSVD", rank_candidates), ("Tikhonov", lambda_candidates)):
            results = {
                criterion: next(item for item in selectors if item.method == method and item.criterion == criterion)
                for criterion in CRITERIA
            }
            base = results["l_curve"]
            for index, parameter in enumerate(candidates):
                writer.writerow({
                    "method": method,
                    "candidate_index": index,
                    "parameter": parameter,
                    "residual_norm": base.residual_norms[index],
                    "solution_norm": base.solution_norms[index],
                    "effective_degrees_of_freedom": base.effective_degrees_of_freedom[index],
                    "gcv_numerator": base.residual_norms[index] ** 2 / observation.size,
                    "gcv_denominator": 1.0 - base.effective_degrees_of_freedom[index] / observation.size,
                    "lcurve_curvature": results["l_curve"].criterion_values[index] if np.isfinite(results["l_curve"].criterion_values[index]) else "",
                    "gcv": results["gcv"].criterion_values[index] if np.isfinite(results["gcv"].criterion_values[index]) else "",
                    "discrepancy_absolute_error": results["discrepancy"].criterion_values[index],
                })

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, method in enumerate(("TSVD", "Tikhonov")):
        candidates = rank_candidates if method == "TSVD" else lambda_candidates
        group = {item.criterion: item for item in selectors if item.method == method}
        parameter_axis = candidates if method == "TSVD" else np.log10(candidates)
        label = "retained rank k" if method == "TSVD" else "log10(lambda)"
        ax = axes[row, 0]
        base = group["l_curve"]
        ax.loglog(base.residual_norms, base.solution_norms, marker=".", markersize=3)
        ax.set(title=f"{method}: L-curve", xlabel="residual norm", ylabel="solution norm")
        ax.grid(True, alpha=0.3)
        ax = axes[row, 1]
        ax.plot(parameter_axis, group["gcv"].criterion_values)
        ax.set(title=f"{method}: GCV", xlabel=label, ylabel="GCV score")
        ax.grid(True, alpha=0.3)
        ax = axes[row, 2]
        ax.plot(parameter_axis, group["discrepancy"].residual_norms, label="residual norm")
        target = DISCREPANCY_FACTOR * PHASE4_NOISE_STANDARD_DEVIATION * np.sqrt(observation.size)
        ax.axhline(target, color="tab:red", linestyle="--", label="target")
        ax.set(title=f"{method}: discrepancy", xlabel=label, ylabel="residual norm")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "phase5_parameter_selection.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, ((method, criterion), image) in zip(axes.flat, reconstructions.items()):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        selected = next(r.selected_parameter for r in selectors if r.method == method and r.criterion == criterion)
        ax.set_title(f"{method} {criterion}\nparameter={selected:.3g}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(PLOTS / "phase5_selected_reconstructions.png", dpi=160)
    plt.close(fig)
    print(f"Wrote {curve_path}")
    print(f"Wrote {METRICS / 'phase5_selection_summary.csv'}")
    print(f"Wrote Phase 5 plots and six raw reconstruction arrays under results/")


if __name__ == "__main__":
    main()
