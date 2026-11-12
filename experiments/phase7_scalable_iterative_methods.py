"""Validate and benchmark matrix-free iterative image deblurring methods."""

import csv
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.svd_analysis import MAX_FULL_SVD_DIMENSION, analyze_svd  # noqa: E402
from evaluation.reconstruction_metrics import (  # noqa: E402
    mean_squared_error,
    peak_signal_to_noise_ratio,
    relative_reconstruction_error,
    structural_similarity_index,
)
from forward_model.gaussian import gaussian_blur_operator_2d  # noqa: E402
from numerical_methods.baseline_solvers import solve_qr_least_squares  # noqa: E402
from numerical_methods.iterative_solvers import (  # noqa: E402
    IterativeResult,
    solve_cgls,
    solve_lsqr,
    solve_lsqr_checkpoints,
)
from regularization.tikhonov import solve_tikhonov_svd  # noqa: E402
from utils.phase7_protocol import (  # noqa: E402
    create_matrix_free_problem,
    structured_operator_condition_number,
)


RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
IMAGES_DIR = RESULTS_DIR / "images"
LOGS_DIR = RESULTS_DIR / "logs"

EXPERIMENT_CASES = (
    {"case_id": "resolution_32_sigma_1_5", "image_shape": (32, 32), "sigma": 1.5},
    {"case_id": "resolution_64_sigma_1_5", "image_shape": (64, 64), "sigma": 1.5},
    {"case_id": "resolution_128_sigma_1_5", "image_shape": (128, 128), "sigma": 1.5},
    {"case_id": "conditioning_64_sigma_0_6", "image_shape": (64, 64), "sigma": 0.6},
)
NOISE_STANDARD_DEVIATION = 0.01
RANDOM_SEED = 42
KERNEL_SIZE = 9
ITERATION_BUDGETS = (8, 16, 32, 64)
TOLERANCE_VALUES = (5e-2, 5e-3)
DAMPING_PARAMETER = 0.03
MAXIMUM_ITERATIONS = 64


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty results: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _operator_storage_estimates(image_shape: tuple[int, int]) -> tuple[int, int]:
    """Return float64 storage estimates, excluding Python/SciPy object overhead."""
    rows, columns = image_shape
    pixel_count = rows * columns
    dense_bytes = pixel_count * pixel_count * np.dtype(np.float64).itemsize
    separable_bytes = (rows * rows + columns * columns) * np.dtype(np.float64).itemsize
    return dense_bytes, separable_bytes


def _metric_values(reference: np.ndarray, vector: np.ndarray, image_shape):
    image = np.asarray(vector).reshape(image_shape, order="C")
    return {
        "relative_reconstruction_error": relative_reconstruction_error(reference, image),
        "mean_squared_error": mean_squared_error(reference, image),
        "peak_signal_to_noise_ratio": peak_signal_to_noise_ratio(reference, image),
        "structural_similarity_index": structural_similarity_index(reference, image),
        "reconstruction_norm": float(np.linalg.norm(vector)),
    }


def _make_record(
    *,
    case: dict[str, object],
    problem,
    method: str,
    run_type: str,
    result: IterativeResult,
    requested_limit: int,
    tolerance: float,
    reference: np.ndarray,
    image_name: str | None = None,
    iteration: int | None = None,
    residual_norm: float | None = None,
    normal_residual_norm: float | None = None,
    reconstruction_vector: np.ndarray | None = None,
):
    vector = (
        result.reconstruction_vector
        if reconstruction_vector is None
        else np.asarray(reconstruction_vector, dtype=np.float64)
    )
    metrics = _metric_values(reference, vector, problem.image_shape)
    true_residual_norm = (
        result.residual_norm if residual_norm is None else float(residual_norm)
    )
    true_normal_norm = (
        result.normal_equation_residual_norm
        if normal_residual_norm is None else float(normal_residual_norm)
    )
    observation_norm = float(np.linalg.norm(problem.observation))
    dense_bytes, separable_bytes = _operator_storage_estimates(problem.image_shape)
    return {
        "case_id": case["case_id"],
        "image_height": problem.image_shape[0],
        "image_width": problem.image_shape[1],
        "pixel_count": problem.operator.shape[1],
        "operator_rows": problem.operator.shape[0],
        "operator_columns": problem.operator.shape[1],
        "sigma": problem.sigma,
        "kernel_size": problem.kernel_size,
        "boundary": problem.boundary,
        "noise_standard_deviation": problem.noise_standard_deviation,
        "random_seed": problem.random_seed,
        "method": method,
        "run_type": run_type,
        "requested_iteration_limit": requested_limit,
        "iteration_count": iteration if iteration is not None else result.iteration_count,
        "relative_tolerance": tolerance,
        "absolute_tolerance": result.absolute_tolerance,
        "damping_parameter_lambda": result.damping_parameter,
        "relative_reconstruction_error": metrics["relative_reconstruction_error"],
        "mean_squared_error": metrics["mean_squared_error"],
        "peak_signal_to_noise_ratio": metrics["peak_signal_to_noise_ratio"],
        "structural_similarity_index": metrics["structural_similarity_index"],
        "residual_norm": true_residual_norm,
        "relative_residual": true_residual_norm / observation_norm if observation_norm else true_residual_norm,
        "normal_equation_residual_norm": true_normal_norm,
        "reconstruction_norm": metrics["reconstruction_norm"],
        "termination_code": result.termination_code,
        "termination_status": result.termination_status,
        "condition_estimate": result.condition_estimate,
        "operator_condition_number_from_1d_factors": case.get(
            "operator_condition_number_from_1d_factors", np.nan
        ),
        "execution_time_seconds": result.execution_time_seconds,
        "theoretical_dense_operator_bytes": dense_bytes,
        "theoretical_separable_factor_bytes": separable_bytes,
        "memory_measurement": "theoretical float64 array storage only; peak RAM not measured",
        "raw_reconstruction_file": image_name or "",
    }


def run_small_system_validation() -> list[dict[str, object]]:
    """Compare iterative methods with permitted small dense reference methods."""
    shape = (8, 8)
    sigma = 0.8
    damping = 0.05
    problem = create_matrix_free_problem(
        shape, sigma=sigma, kernel_size=7,
        noise_standard_deviation=NOISE_STANDARD_DEVIATION,
        random_seed=RANDOM_SEED,
    )
    dense_reference_operator = gaussian_blur_operator_2d(
        shape, sigma=sigma, kernel_size=7
    )
    svd = analyze_svd(dense_reference_operator, relative_tolerance=1e-12)
    reference_ls = solve_qr_least_squares(
        dense_reference_operator, problem.observation, shape
    )
    reference_tikh = solve_tikhonov_svd(
        dense_reference_operator,
        problem.observation,
        shape,
        damping,
        relative_tolerance=1e-12,
        svd_analysis=svd,
    )
    validation_rows = []

    for tolerance in TOLERANCE_VALUES + (1e-12,):
        result = solve_lsqr(
            problem.operator,
            problem.observation,
            iteration_limit=128,
            relative_tolerance=tolerance,
            absolute_tolerance=tolerance,
            condition_limit=1e14,
        )
        validation_rows.append({
            "method": "LSQR",
            "tolerance": tolerance,
            "iteration_limit": 128,
            "iterations": result.iteration_count,
            "termination_status": result.termination_status,
            "relative_solution_difference_from_QR": (
                np.linalg.norm(result.reconstruction_vector - reference_ls.reconstruction_vector)
                / np.linalg.norm(reference_ls.reconstruction_vector)
            ),
            "relative_solution_difference_from_Tikhonov": np.nan,
            "residual_norm": result.residual_norm,
            "normal_equation_residual_norm": result.normal_equation_residual_norm,
            "diagnostic": "LSQR converges toward the unregularized QR least-squares solution.",
        })

        damped = solve_lsqr(
            problem.operator,
            problem.observation,
            iteration_limit=128,
            relative_tolerance=tolerance,
            absolute_tolerance=tolerance,
            damping_parameter=damping,
            condition_limit=1e14,
        )
        validation_rows.append({
            "method": "damped_LSQR",
            "tolerance": tolerance,
            "iteration_limit": 128,
            "iterations": damped.iteration_count,
            "termination_status": damped.termination_status,
            "relative_solution_difference_from_QR": np.nan,
            "relative_solution_difference_from_Tikhonov": (
                np.linalg.norm(damped.reconstruction_vector - reference_tikh.reconstruction_vector)
                / np.linalg.norm(reference_tikh.reconstruction_vector)
            ),
            "residual_norm": damped.residual_norm,
            "normal_equation_residual_norm": damped.normal_equation_residual_norm,
            "diagnostic": (
                "damp=lambda; reference objective is ||Ax-b||^2 + "
                "lambda^2||x||^2."
            ),
        })

    cgls = solve_cgls(
        problem.operator, problem.observation, iteration_limit=128,
        relative_tolerance=1e-12, absolute_tolerance=1e-14,
    )
    validation_rows.append({
        "method": "CGLS",
        "tolerance": 1e-12,
        "iteration_limit": 128,
        "iterations": cgls.iteration_count,
        "termination_status": cgls.termination_status,
        "relative_solution_difference_from_QR": (
            np.linalg.norm(cgls.reconstruction_vector - reference_ls.reconstruction_vector)
            / np.linalg.norm(reference_ls.reconstruction_vector)
        ),
        "relative_solution_difference_from_Tikhonov": np.nan,
        "residual_norm": cgls.residual_norm,
        "normal_equation_residual_norm": cgls.normal_equation_residual_norm,
        "diagnostic": "CGLS converges toward the unregularized QR least-squares solution.",
    })
    return validation_rows


def _save_reconstruction(
    case_id: str,
    method: str,
    run_type: str,
    limit: int,
    vector: np.ndarray,
    image_shape: tuple[int, int],
    tolerance: float | None = None,
) -> str:
    safe_case = str(case_id).replace(".", "_")
    tolerance_label = f"_tol_{tolerance:g}" if tolerance is not None else ""
    name = (
        f"phase7_{safe_case}_{method}_{run_type}_{limit}"
        f"{tolerance_label}_raw.npy"
    )
    np.save(IMAGES_DIR / name, np.asarray(vector).reshape(image_shape, order="C"))
    return name


def _run_case(case: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    case = dict(case)
    problem = create_matrix_free_problem(
        case["image_shape"],
        sigma=case["sigma"],
        kernel_size=KERNEL_SIZE,
        noise_standard_deviation=NOISE_STANDARD_DEVIATION,
        random_seed=RANDOM_SEED,
    )
    if not isinstance(problem.operator, LinearOperator):
        raise RuntimeError("large-image path must use scipy LinearOperator")
    if problem.operator.shape[0] != problem.image_shape[0] * problem.image_shape[1]:
        raise RuntimeError("matrix-free operator has an unexpected shape")
    if max(problem.image_shape) > MAX_FULL_SVD_DIMENSION:
        raise RuntimeError("image side exceeds the one-dimensional operator limit")
    structured_condition = structured_operator_condition_number(
        problem.image_shape,
        sigma=problem.sigma,
        kernel_size=problem.kernel_size,
    )
    case["operator_condition_number_from_1d_factors"] = structured_condition
    forward_consistency = np.linalg.norm(
        problem.operator.matvec(problem.original.ravel(order="C"))
        - problem.blurred.ravel(order="C")
    )
    if forward_consistency != 0.0:
        raise RuntimeError(f"observation was not generated by its operator: {forward_consistency}")
    input_array_files = {}
    for array_name, array in (
        ("original", problem.original),
        ("blurred", problem.blurred),
        ("noise", problem.noise),
        ("noisy_observation", problem.noisy_observation),
    ):
        filename = f"phase7_{case['case_id']}_{array_name}_raw.npy"
        np.save(IMAGES_DIR / filename, array)
        input_array_files[f"{array_name}_array_file"] = filename

    # All solver outputs are computed without using the reference image.
    cgls = solve_cgls(
        problem.operator, problem.observation,
        iteration_limit=MAXIMUM_ITERATIONS,
        relative_tolerance=1e-10,
        absolute_tolerance=0.0,
    )
    lsqr_checkpoints = solve_lsqr_checkpoints(
        problem.operator, problem.observation, ITERATION_BUDGETS,
        relative_tolerance=1e-14, absolute_tolerance=1e-14,
        condition_limit=1e12,
    )
    damped_checkpoints = solve_lsqr_checkpoints(
        problem.operator, problem.observation, ITERATION_BUDGETS,
        relative_tolerance=1e-14, absolute_tolerance=1e-14,
        damping_parameter=DAMPING_PARAMETER,
        condition_limit=1e12,
    )
    tolerance_runs = []
    for method, damping in (("LSQR", 0.0), ("damped_LSQR", DAMPING_PARAMETER)):
        for tolerance in TOLERANCE_VALUES:
            tolerance_runs.append((
                method,
                tolerance,
                solve_lsqr(
                    problem.operator,
                    problem.observation,
                    iteration_limit=MAXIMUM_ITERATIONS,
                    relative_tolerance=tolerance,
                    absolute_tolerance=tolerance,
                    damping_parameter=damping,
                    condition_limit=1e12,
                ),
            ))
    for tolerance in TOLERANCE_VALUES:
        tolerance_runs.append((
            "CGLS",
            tolerance,
            solve_cgls(
                problem.operator, problem.observation,
                iteration_limit=MAXIMUM_ITERATIONS,
                relative_tolerance=tolerance,
                absolute_tolerance=0.0,
            ),
        ))

    # Ground truth is used only below, after every solver run has terminated.
    reference = problem.original
    records = []
    case_id = str(case["case_id"])
    trajectory_points = (0, 1, 2, 4, 8, 16, 32, 64)
    for diagnostic, vector in zip(cgls.history, cgls.reconstruction_history):
        if diagnostic.iteration not in trajectory_points:
            continue
        name = ""
        if diagnostic.iteration == MAXIMUM_ITERATIONS:
            name = _save_reconstruction(
                case_id, "CGLS", "trajectory", diagnostic.iteration,
                vector, problem.image_shape,
            )
        trajectory_record = _make_record(
            case=case, problem=problem, method="CGLS",
            run_type="iteration_trajectory", result=cgls,
            requested_limit=MAXIMUM_ITERATIONS,
            tolerance=cgls.relative_tolerance, reference=reference,
            image_name=name, iteration=diagnostic.iteration,
            residual_norm=diagnostic.residual_norm,
            normal_residual_norm=diagnostic.normal_equation_residual_norm,
            reconstruction_vector=vector,
        )
        if diagnostic.iteration < cgls.iteration_count:
            trajectory_record["termination_status"] = (
                "intermediate trajectory checkpoint; run continued"
            )
        records.append(trajectory_record)

    for method, results in (
        ("LSQR", lsqr_checkpoints),
        ("damped_LSQR", damped_checkpoints),
    ):
        for result in results:
            name = _save_reconstruction(
                case_id, method, "iteration_budget", result.iteration_limit,
                result.reconstruction_vector, problem.image_shape,
            )
            records.append(_make_record(
                case=case, problem=problem, method=method,
                run_type="iteration_budget", result=result,
                requested_limit=result.iteration_limit,
                tolerance=result.relative_tolerance, reference=reference,
                image_name=name,
            ))

    for method, tolerance, result in tolerance_runs:
        name = _save_reconstruction(
            case_id, method, "tolerance", result.iteration_limit,
            result.reconstruction_vector, problem.image_shape,
            tolerance=tolerance,
        )
        records.append(_make_record(
            case=case, problem=problem, method=method,
            run_type="tolerance", result=result,
            requested_limit=result.iteration_limit,
            tolerance=tolerance, reference=reference, image_name=name,
        ))
    return records, [{
        "case_id": case_id,
        "image_shape": f"{problem.image_shape[0]}x{problem.image_shape[1]}",
        "sigma": problem.sigma,
        "kernel_size": problem.kernel_size,
        "noise_standard_deviation": problem.noise_standard_deviation,
        "random_seed": problem.random_seed,
        "boundary": problem.boundary,
        "operator_shape": f"{problem.operator.shape[0]}x{problem.operator.shape[1]}",
        "dense_svd_used": False,
        "full_dense_matrix_constructed": False,
        "full_image_dense_svd_used": False,
        "one_dimensional_factor_svd_used_for_exact_condition": True,
        "forward_consistency_error": forward_consistency,
        "operator_condition_number_from_1d_factors": structured_condition,
        **input_array_files,
    }]


def _plot_results(records: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for sigma, linestyle in ((0.6, "-"), (1.5, "--")):
        rows = [
            row for row in records
            if row["image_width"] == 64
            and row["sigma"] == sigma
            and row["method"] == "CGLS"
            and row["run_type"] == "iteration_trajectory"
        ]
        rows.sort(key=lambda row: row["iteration_count"])
        label = f"CGLS sigma={sigma:g}"
        axes[0].semilogy(
            [row["iteration_count"] for row in rows],
            [row["relative_reconstruction_error"] for row in rows],
            marker="o", linestyle=linestyle, label=label,
        )
        axes[1].semilogy(
            [row["iteration_count"] for row in rows],
            [row["residual_norm"] for row in rows],
            marker="o", linestyle=linestyle, label=label,
        )
    axes[0].set(xlabel="CGLS iteration", ylabel="Relative reconstruction error",
                title="Early stopping and reconstruction accuracy")
    axes[1].set(xlabel="CGLS iteration", ylabel="Residual norm",
                title="CGLS data residual")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase7_cgls_convergence_and_early_stopping.png", dpi=180)
    plt.close(fig)

    # Compare final requested iteration budgets at each scalable resolution.
    scalable = [
        row for row in records
        if row["sigma"] == 1.5
        and row["run_type"] == "iteration_budget"
        and row["requested_iteration_limit"] == max(ITERATION_BUDGETS)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method in ("LSQR", "CGLS", "damped_LSQR"):
        rows = sorted(
            (row for row in scalable if row["method"] == method),
            key=lambda row: row["pixel_count"],
        )
        if method == "CGLS":
            rows = []
            for case_id in (
                "resolution_32_sigma_1_5",
                "resolution_64_sigma_1_5",
                "resolution_128_sigma_1_5",
            ):
                rows.append(next(
                    row for row in records
                    if row["case_id"] == case_id
                    and row["method"] == "CGLS"
                    and row["run_type"] == "iteration_trajectory"
                    and row["iteration_count"] == MAXIMUM_ITERATIONS
                ))
        axes[0].loglog(
            [row["pixel_count"] for row in rows],
            [row["relative_reconstruction_error"] for row in rows],
            marker="o", label=method,
        )
        axes[1].loglog(
            [row["pixel_count"] for row in rows],
            [row["execution_time_seconds"] for row in rows],
            marker="o", label=method,
        )
    axes[0].set(xlabel="Image pixel count", ylabel="Relative reconstruction error",
                title="Reconstruction quality at iteration budget 64")
    axes[1].set(xlabel="Image pixel count", ylabel="Solver time (seconds)",
                title="Iterative solver time versus resolution")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase7_scalability_comparison.png", dpi=180)
    plt.close(fig)

    tolerance_rows = [
        row for row in records
        if row["run_type"] == "tolerance" and row["case_id"] == "resolution_64_sigma_1_5"
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in ("LSQR", "CGLS", "damped_LSQR"):
        rows = sorted(
            (row for row in tolerance_rows if row["method"] == method),
            key=lambda row: row["relative_tolerance"],
        )
        ax.loglog(
            [row["relative_tolerance"] for row in rows],
            [row["relative_reconstruction_error"] for row in rows],
            marker="o", label=method,
        )
    ax.set(xlabel="Requested relative stopping tolerance",
           ylabel="Relative reconstruction error",
           title="Tolerance sensitivity at 64×64, sigma=1.5")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase7_tolerance_sensitivity.png", dpi=180)
    plt.close(fig)

    problem = create_matrix_free_problem(
        (128, 128), sigma=1.5, kernel_size=KERNEL_SIZE,
        noise_standard_deviation=NOISE_STANDARD_DEVIATION,
        random_seed=RANDOM_SEED,
    )
    lsqr_reconstruction = next(
        np.load(IMAGES_DIR / row["raw_reconstruction_file"])
        for row in records
        if row["case_id"] == "resolution_128_sigma_1_5"
        and row["method"] == "LSQR"
        and row["run_type"] == "iteration_budget"
        and row["requested_iteration_limit"] == MAXIMUM_ITERATIONS
    )
    reconstruction = lsqr_reconstruction.reshape((128, 128), order="C")
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    image_items = (
        ("Original", problem.original),
        ("Blurred", problem.blurred),
        ("Noisy observation", problem.noisy_observation),
        ("LSQR reconstruction", reconstruction),
        ("CGLS reconstruction", next(
            np.load(IMAGES_DIR / row["raw_reconstruction_file"])
            for row in records
            if row["case_id"] == "resolution_128_sigma_1_5"
            and row["method"] == "CGLS"
            and row["run_type"] == "iteration_trajectory"
            and row["iteration_count"] == MAXIMUM_ITERATIONS
            and row["raw_reconstruction_file"]
        )),
        ("Damped LSQR reconstruction", next(
            np.load(IMAGES_DIR / row["raw_reconstruction_file"])
            for row in records
            if row["case_id"] == "resolution_128_sigma_1_5"
            and row["method"] == "damped_LSQR"
            and row["run_type"] == "iteration_budget"
            and row["requested_iteration_limit"] == MAXIMUM_ITERATIONS
        )),
    )
    for ax, (title, image) in zip(axes.flat, image_items):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
    figure.suptitle("128×128 matrix-free reconstructions (display limits only)")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "phase7_representative_128_reconstructions.png", dpi=180)
    plt.close(figure)


def run_phase7_experiment() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run reference validation, scalable cases, and save all Phase 7 outputs."""
    for directory in (METRICS_DIR, PLOTS_DIR, IMAGES_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase7")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(
        LOGS_DIR / "phase7_experiment.log", mode="w", encoding="utf-8"
    ))
    logger.addHandler(logging.StreamHandler())

    config = {
        "phase": 7,
        "cases": EXPERIMENT_CASES,
        "noise_standard_deviation": NOISE_STANDARD_DEVIATION,
        "random_seed": RANDOM_SEED,
        "kernel_size": KERNEL_SIZE,
        "boundary": "zero, no boundary-row renormalization",
        "vectorization": "C-order",
        "iteration_budgets": ITERATION_BUDGETS,
        "tolerance_values": TOLERANCE_VALUES,
        "damping_parameter_lambda": DAMPING_PARAMETER,
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "damped_objective": "||A x-b||_2^2 + lambda^2 ||x||_2^2",
        "matrix_free": True,
        "full_dense_matrix_or_svd_for_scalable_cases": False,
        "peak_memory": "not measured; only theoretical float64 storage estimates recorded",
    }
    with (METRICS_DIR / "phase7_experiment_config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)

    validation = run_small_system_validation()
    _write_csv(METRICS_DIR / "phase7_small_system_validation.csv", validation)
    logger.info("Completed small-system comparison against QR and SVD Tikhonov references.")

    all_records = []
    case_records = []
    for case in EXPERIMENT_CASES:
        logger.info("Running matrix-free case %s", case["case_id"])
        case_start = perf_counter()
        records, metadata = _run_case(case)
        elapsed = perf_counter() - case_start
        for record in records:
            record["case_execution_seconds"] = elapsed
        all_records.extend(records)
        case_records.extend(metadata)
        logger.info(
            "%s shape=%s sigma=%s records=%d case_seconds=%.3f",
            case["case_id"], case["image_shape"], case["sigma"], len(records), elapsed,
        )
    _write_csv(METRICS_DIR / "phase7_iterative_results.csv", all_records)
    _write_csv(METRICS_DIR / "phase7_case_metadata.csv", case_records)
    _plot_results(all_records)
    logger.info("Completed %d matrix-free cases and %d experiment rows.", len(case_records), len(all_records))
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()
    return all_records, validation


if __name__ == "__main__":
    run_phase7_experiment()
