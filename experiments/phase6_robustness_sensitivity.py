"""Run controlled Phase 6 robustness and sensitivity experiments."""

import csv
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.svd_analysis import MAX_FULL_SVD_DIMENSION  # noqa: E402
from evaluation.reconstruction_metrics import evaluate_reconstruction  # noqa: E402
from evaluation.robustness import aggregate_records, classify_unstable_solution  # noqa: E402
from numerical_methods.baseline_solvers import (  # noqa: E402
    solve_direct_inverse,
    solve_normal_equations,
    solve_qr_least_squares,
)
from regularization.parameter_selection import (  # noqa: E402
    select_parameter,
    tikhonov_candidate_lambdas,
    tsvd_candidate_ranks,
)
from regularization.tikhonov import solve_tikhonov_svd  # noqa: E402
from regularization.tsvd import solve_tsvd  # noqa: E402
from utils.experiment_setup import create_camera_deblurring_problem  # noqa: E402
from utils.phase6_protocol import (  # noqa: E402
    PHASE6_DISCREPANCY_FACTOR,
    PHASE6_KERNEL_SIZE,
    PHASE6_NOISE_STANDARD_DEVIATION,
    PHASE6_RANK_RELATIVE_TOLERANCE,
    build_phase6_cases,
)


RESULTS = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS / "metrics"
PLOTS_DIR = RESULTS / "plots"
IMAGES_DIR = RESULTS / "images"
LOGS_DIR = RESULTS / "logs"
UNSTABLE_RELATIVE_ERROR_LIMIT = 10.0
UNSTABLE_SOLUTION_NORM_RATIO_LIMIT = 10.0
SELECTOR_CRITERIA = ("l_curve", "gcv", "discrepancy")
BASELINE_SOLVERS = (
    ("direct_inverse", solve_direct_inverse),
    ("qr_least_squares", solve_qr_least_squares),
    ("normal_equations", solve_normal_equations),
)


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError(f"refusing to write empty result file: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _method_records(case, problem, setup_seconds: float):
    """Select parameters before reading the reference image, then solve."""
    matrix = problem.operator
    observation = problem.observation
    svd = problem.svd_analysis
    rank_candidates = tsvd_candidate_ranks(svd)
    lambda_candidates = tikhonov_candidate_lambdas(svd)
    pending = []
    selection_records = []

    # No reference image is passed to any selector. All selectors share this
    # case's exact matrix, observation, and SVD.
    for method, candidates in (
        ("TSVD", rank_candidates),
        ("Tikhonov", lambda_candidates),
    ):
        for criterion in SELECTOR_CRITERIA:
            try:
                selection = select_parameter(
                    method,
                    criterion,
                    matrix,
                    observation,
                    svd,
                    candidates,
                    noise_standard_deviation=case.noise_standard_deviation,
                    discrepancy_factor=PHASE6_DISCREPANCY_FACTOR,
                )
                selection_records.append({
                    "case_id": case.case_id,
                    "sensitivity_axis": case.sensitivity_axis,
                    "condition_value": case.condition_value,
                    "method": method,
                    "criterion": criterion,
                    "selection_status": "success",
                    "selected_parameter": selection.selected_parameter,
                    "selected_index": selection.selected_index,
                    "criterion_score": selection.criterion_values[selection.selected_index],
                    "diagnostic": selection.diagnostic,
                })
                pending.append((method, criterion, selection, None, None))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                selection_records.append({
                    "case_id": case.case_id,
                    "sensitivity_axis": case.sensitivity_axis,
                    "condition_value": case.condition_value,
                    "method": method,
                    "criterion": criterion,
                    "selection_status": "failed",
                    "selected_parameter": np.nan,
                    "selected_index": np.nan,
                    "criterion_score": np.nan,
                    "diagnostic": f"{type(error).__name__}: {error}",
                })
                pending.append((method, criterion, None, None, error))

    for method_name, solver in BASELINE_SOLVERS:
        try:
            result = solver(matrix, observation, case.image_shape)
            pending.append((method_name, "fixed", None, result, None))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            pending.append((method_name, "fixed", None, None, error))

    # Ground truth is not accessed until all parameter selections have ended.
    reference = problem.original
    reference_norm = float(np.linalg.norm(reference.ravel(order="C")))
    records = []
    for method, criterion, selection, solved, error in pending:
        if solved is None and error is None and selection is not None:
            try:
                if method == "TSVD":
                    solved = solve_tsvd(
                        matrix, observation, case.image_shape,
                        int(selection.selected_parameter),
                        relative_tolerance=PHASE6_RANK_RELATIVE_TOLERANCE,
                        svd_analysis=svd,
                    )
                else:
                    solved = solve_tikhonov_svd(
                        matrix, observation, case.image_shape,
                        selection.selected_parameter,
                        relative_tolerance=PHASE6_RANK_RELATIVE_TOLERANCE,
                        svd_analysis=svd,
                    )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as solve_error:
                error = solve_error

        parameter = selection.selected_parameter if selection is not None else np.nan
        parameter_score = (
            selection.criterion_values[selection.selected_index]
            if selection is not None else np.nan
        )
        if solved is None:
            metrics = None
            reconstruction_norm = np.nan
            execution_time = np.nan
            status = "failed"
            diagnostic = f"{type(error).__name__}: {error}" if error else "no solution"
            unstable = True
            residual_norm = relative_residual = np.nan
        else:
            metrics = evaluate_reconstruction(
                reference, solved.reconstruction_image, matrix, observation
            )
            reconstruction_norm = float(np.linalg.norm(solved.reconstruction_vector))
            execution_time = float(solved.execution_time_seconds)
            status = "success"
            diagnostic = solved.diagnostic
            residual_norm = metrics.residual_norm
            relative_residual = metrics.relative_residual
            unstable = classify_unstable_solution(
                metrics.relative_reconstruction_error,
                reconstruction_norm,
                reference_norm,
                maximum_relative_error=UNSTABLE_RELATIVE_ERROR_LIMIT,
                maximum_norm_ratio=UNSTABLE_SOLUTION_NORM_RATIO_LIMIT,
            )
            safe_case_id = case.case_id.replace(".", "_")
            filename = f"phase6_{safe_case_id}_{method}_{criterion}_raw.npy"
            np.save(IMAGES_DIR / filename, solved.reconstruction_image)

        records.append({
            "case_id": case.case_id,
            "sensitivity_axis": case.sensitivity_axis,
            "condition_value": case.condition_value,
            "aggregation_condition": (
                "all_noise_seeds" if case.sensitivity_axis == "noise"
                else case.condition_value
            ),
            "image_height": case.image_shape[0],
            "image_width": case.image_shape[1],
            "pixel_count": case.image_shape[0] * case.image_shape[1],
            "sigma": case.sigma,
            "kernel_size": case.kernel_size,
            "boundary": case.boundary,
            "noise_standard_deviation": case.noise_standard_deviation,
            "random_seed": case.random_seed,
            "relative_rank_tolerance": PHASE6_RANK_RELATIVE_TOLERANCE,
            "discrepancy_factor": PHASE6_DISCREPANCY_FACTOR,
            "method": method,
            "criterion": criterion,
            "parameter_type": (
                "rank" if method == "TSVD"
                else "lambda" if method == "Tikhonov"
                else "none"
            ),
            "selected_parameter": parameter,
            "selected_rank": int(parameter) if method == "TSVD" and np.isfinite(parameter) else np.nan,
            "selected_lambda": parameter if method == "Tikhonov" else np.nan,
            "criterion_score": parameter_score,
            "status": status,
            "diagnostic": diagnostic,
            "relative_reconstruction_error": (
                metrics.relative_reconstruction_error if metrics else np.nan
            ),
            "mean_squared_error": metrics.mean_squared_error if metrics else np.nan,
            "peak_signal_to_noise_ratio": (
                metrics.peak_signal_to_noise_ratio if metrics else np.nan
            ),
            "structural_similarity_index": (
                metrics.structural_similarity_index if metrics else np.nan
            ),
            "residual_norm": residual_norm,
            "relative_residual": relative_residual,
            "reconstruction_norm": reconstruction_norm,
            "reference_norm": reference_norm,
            "numerical_rank": svd.numerical_rank,
            "condition_number": svd.condition_number,
            "execution_time_seconds": execution_time,
            "problem_setup_seconds": setup_seconds,
            "unstable_solution": unstable,
            "instability_rule": (
                "relative error >= 10 or reconstruction/reference norm >= 10"
            ),
        })
    return records, selection_records


def _plot_sensitivity(records: list[dict[str, object]]) -> None:
    successful = [row for row in records if row["status"] == "success"]
    methods = sorted({f"{row['method']}:{row['criterion']}" for row in successful})
    colors = plt.get_cmap("tab10").colors

    for axis, x_field, xlabel, title in (
        ("noise", "random_seed", "Noise random seed", "Noise-realization sensitivity"),
        ("blur", "sigma", "Gaussian sigma", "Blur-strength sensitivity"),
        ("resolution", "pixel_count", "Image pixel count", "Image-resolution sensitivity"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        group = [row for row in successful if row["sensitivity_axis"] == axis]
        for index, label in enumerate(methods):
            method, criterion = label.split(":", 1)
            points = sorted(
                (row for row in group if row["method"] == method and row["criterion"] == criterion),
                key=lambda row: float(row[x_field]),
            )
            if points:
                ax.plot(
                    [row[x_field] for row in points],
                    [row["relative_reconstruction_error"] for row in points],
                    marker="o", label=label, color=colors[index % len(colors)],
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Relative reconstruction error (dimensionless)")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize="small", ncol=2)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"phase6_{axis}_sensitivity.png", dpi=180)
        plt.close(fig)

    # Selected regularization parameters across every experimental condition.
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for row_index, (method, value_field, ylabel) in enumerate((
        ("TSVD", "selected_rank", "Selected TSVD rank"),
        ("Tikhonov", "selected_lambda", "Selected Tikhonov lambda"),
    )):
        for col_index, axis in enumerate(("noise", "blur", "resolution")):
            ax = axes[row_index, col_index]
            group = [
                item for item in successful
                if item["method"] == method and item["sensitivity_axis"] == axis
                and item["criterion"] != "fixed"
            ]
            x_field = {"noise": "random_seed", "blur": "sigma", "resolution": "pixel_count"}[axis]
            for criterion in SELECTOR_CRITERIA:
                points = sorted(
                    (item for item in group if item["criterion"] == criterion),
                    key=lambda item: float(item[x_field]),
                )
                if points:
                    ax.plot(
                        [item[x_field] for item in points],
                        [item[value_field] for item in points],
                        marker="o", label=criterion,
                    )
            ax.set_title(f"{method}: {axis} variation")
            ax.set_xlabel({"noise": "seed", "blur": "Gaussian sigma", "resolution": "pixel count"}[axis])
            ax.set_ylabel(ylabel)
            if method == "Tikhonov":
                ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase6_selected_parameters.png", dpi=180)
    plt.close(fig)

    # Heatmap preserves each condition and method combination in one view.
    cases = list(dict.fromkeys(row["case_id"] for row in successful))
    columns = list(dict.fromkeys(
        f"{row['method']}:{row['criterion']}" for row in successful
    ))
    values = np.full((len(cases), len(columns)), np.nan)
    for row in successful:
        i = cases.index(row["case_id"])
        j = columns.index(f"{row['method']}:{row['criterion']}")
        values[i, j] = row["relative_reconstruction_error"]
    fig, ax = plt.subplots(figsize=(13, 8))
    image = ax.imshow(np.log10(values), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(columns)), labels=columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cases)), labels=cases)
    ax.set_title("Phase 6 relative reconstruction error (log10)")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("log10(relative reconstruction error)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase6_error_heatmap.png", dpi=180)
    plt.close(fig)

    noise_rows = [
        row for row in successful
        if row["sensitivity_axis"] == "noise"
        and row["method"] != "direct_inverse"
    ]
    box_labels = []
    box_values = []
    for label in methods:
        method, criterion = label.split(":", 1)
        values_for_method = [
            float(row["relative_reconstruction_error"])
            for row in noise_rows
            if row["method"] == method and row["criterion"] == criterion
        ]
        if values_for_method:
            box_labels.append(label)
            box_values.append(values_for_method)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(box_values, tick_labels=box_labels, showmeans=True)
    ax.set_yscale("log")
    ax.set_ylabel("Relative reconstruction error (dimensionless)")
    ax.set_title("Method-wise variability across independent noise seeds")
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase6_noise_variability.png", dpi=180)
    plt.close(fig)


def run_phase6_experiment() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run all Phase 6 cases and persist raw, aggregated, and visual outputs."""
    for directory in (METRICS_DIR, PLOTS_DIR, IMAGES_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "phase6_experiment.log"
    logger = logging.getLogger("phase6")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logger.addHandler(logging.StreamHandler())
    cases = build_phase6_cases()
    all_records = []
    all_selection_records = []

    configuration = {
        "phase": 6,
        "image_source": "skimage.data.camera resized with Pillow LANCZOS and normalized",
        "cases": [case.__dict__ for case in cases],
        "noise_seeds": [case.random_seed for case in cases if case.sensitivity_axis == "noise"],
        "noise_standard_deviation": PHASE6_NOISE_STANDARD_DEVIATION,
        "kernel_size": PHASE6_KERNEL_SIZE,
        "boundary": "zero, no boundary renormalization",
        "rank_relative_tolerance": PHASE6_RANK_RELATIVE_TOLERANCE,
        "discrepancy_factor": PHASE6_DISCREPANCY_FACTOR,
        "candidate_grids": {
            "tsvd": "integer ranks 1 through numerical rank",
            "tikhonov": "201 log-spaced lambdas from 1e-6*s_max to 10*s_max",
        },
        "methods": [
            "TSVD:L-curve", "TSVD:GCV", "TSVD:discrepancy",
            "Tikhonov:L-curve", "Tikhonov:GCV", "Tikhonov:discrepancy",
            "direct_inverse", "qr_least_squares", "normal_equations",
        ],
        "maximum_exact_dense_svd_dimension": MAX_FULL_SVD_DIMENSION,
        "instability_flag_thresholds": {
            "relative_reconstruction_error_at_least": UNSTABLE_RELATIVE_ERROR_LIMIT,
            "reconstruction_to_reference_norm_at_least": UNSTABLE_SOLUTION_NORM_RATIO_LIMIT,
        },
        "phase5_outputs_are_not_modified": True,
    }
    with (METRICS_DIR / "phase6_experiment_config.json").open("w", encoding="utf-8") as stream:
        json.dump(configuration, stream, indent=2)

    for case in cases:
        logger.info("Running %s", case.case_id)
        setup_start = perf_counter()
        problem = create_camera_deblurring_problem(
            image_shape=case.image_shape,
            sigma=case.sigma,
            kernel_size=case.kernel_size,
            noise_standard_deviation=case.noise_standard_deviation,
            random_seed=case.random_seed,
            rank_relative_tolerance=PHASE6_RANK_RELATIVE_TOLERANCE,
        )
        setup_seconds = perf_counter() - setup_start
        if max(problem.operator.shape) > MAX_FULL_SVD_DIMENSION:
            raise RuntimeError("Phase 6 case unexpectedly exceeds the exact SVD limit")
        operator_consistency = np.linalg.norm(
            problem.operator @ problem.original.ravel(order="C")
            - problem.degradation.blurred.ravel(order="C")
        )
        if operator_consistency > 1e-12:
            raise RuntimeError(
                f"{case.case_id} forward-operator consistency failed: "
                f"{operator_consistency:g}"
            )
        records, selections = _method_records(case, problem, setup_seconds)
        all_records.extend(records)
        all_selection_records.extend(selections)
        unstable_count = sum(
            record["unstable_solution"] for record in records
            if record["status"] == "success"
        )
        logger.info(
            "%s rank=%d condition=%.6g methods=%d unstable=%d",
            case.case_id, problem.svd_analysis.numerical_rank,
            problem.svd_analysis.condition_number, len(records), unstable_count,
        )

    _write_csv(METRICS_DIR / "phase6_raw_results.csv", all_records)
    _write_csv(METRICS_DIR / "phase6_selected_parameters.csv", all_selection_records)
    summaries = aggregate_records(all_records)
    _write_csv(METRICS_DIR / "phase6_aggregated_summary.csv", summaries)
    _plot_sensitivity(all_records)
    logger.info(
        "Completed %d cases and %d method records; wrote Phase 6 artifacts.",
        len(cases), len(all_records),
    )
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()
    return all_records, summaries


if __name__ == "__main__":
    run_phase6_experiment()
