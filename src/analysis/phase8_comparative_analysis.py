"""Compatibility-aware comparative tables and figures for Phase 8 (P8-04)."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TABLE_SPECS = {
    "phase8_table1_baseline_inverse_methods.csv": (
        "source_file", "source_row", "method", "comparison_scope",
        "relative_reconstruction_error", "mean_squared_error", "psnr_db",
        "ssim", "residual_norm", "relative_residual",
    ),
    "phase8_table2_regularization_methods.csv": (
        "source_file", "source_row", "method", "parameter_name",
        "parameter_value", "comparison_scope", "relative_reconstruction_error",
        "mean_squared_error", "psnr_db", "ssim", "residual_norm",
        "relative_residual", "configuration_note",
    ),
    "phase8_table3_parameter_selection.csv": (
        "source_file", "source_row", "method", "criterion", "selected_parameter",
        "comparison_scope", "relative_reconstruction_error", "mse", "psnr_db",
        "ssim", "residual_norm", "relative_residual", "failure_note",
    ),
    "phase8_table4_robustness_sensitivity.csv": (
        "sensitivity_axis", "condition_value", "method", "criterion",
        "n_unique_cases", "mean_relative_reconstruction_error",
        "standard_deviation_relative_reconstruction_error", "mean_residual_norm",
        "unstable_case_count", "comparison_scope", "source_file",
    ),
    "phase8_table5_iterative_solver_comparison.csv": (
        "case_id", "method", "iteration_count", "requested_iteration_limit",
        "relative_reconstruction_error", "residual_norm", "relative_residual",
        "execution_time_seconds", "relative_tolerance", "termination_status",
        "comparison_scope", "source_file", "source_row",
    ),
    "phase8_table6_spectral_conditioning.csv": (
        "evidence_type", "source_file", "source_row", "comparison_scope",
        "image_shape", "sigma", "numerical_rank", "condition_number",
        "largest_singular_value", "smallest_singular_value", "storage_or_validation_note",
    ),
    "phase8_table7_scalability.csv": (
        "case_id", "image_shape", "method", "iteration_count", "runtime_seconds",
        "theoretical_dense_operator_bytes", "theoretical_separable_factor_bytes",
        "storage_interpretation", "comparison_scope", "source_file",
    ),
}

FIGURE_FILENAMES = (
    "phase8_figure01_baseline_inverse_methods.png",
    "phase8_figure02_regularization_accuracy_residual.png",
    "phase8_figure03_lcurve.png",
    "phase8_figure04_gcv_discrepancy_curves.png",
    "phase8_figure05_noise_seed_robustness.png",
    "phase8_figure06_blur_strength_sensitivity.png",
    "phase8_figure07_resolution_sensitivity.png",
    "phase8_figure08_iterative_convergence.png",
    "phase8_figure09_error_vs_iteration.png",
    "phase8_figure10_spectral_decay.png",
    "phase8_figure11_scalability_runtime.png",
    "phase8_figure12_storage_scaling.png",
)

SOURCE_FILES = (
    "phase2_baseline_comparison.csv",
    "phase3_blur_sensitivity.csv",
    "phase3_structured_vs_full_svd.csv",
    "phase3_svd_spectrum.csv",
    "phase3_svd_summary.csv",
    "phase4_regularization_comparison.csv",
    "phase5_selection_curves.csv",
    "phase5_selection_summary.csv",
    "phase6_raw_results.csv",
    "phase7_case_metadata.csv",
    "phase7_iterative_results.csv",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV and reject absent or malformed header rows."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(not name for name in reader.fieldnames):
            raise ValueError(f"Malformed CSV header in {path}")
        return list(reader)


def write_csv(path: Path, fields: Iterable[str], rows: list[dict[str, object]]) -> None:
    """Write a deterministic table, retaining declared headers for empty data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], field: str) -> float:
    """Return a finite float, or NaN when a saved metric is unavailable."""
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def scope_for_row(
    audit: dict[tuple[str, str, str], str], source_file: str,
    source_row: int, metrics: Iterable[str],
) -> str:
    """Classify a source row without silently promoting unknown evidence."""
    statuses = {audit.get((source_file, str(source_row), metric), "unknown") for metric in metrics}
    if statuses == {"compatible"}:
        return "strict_compatible"
    if "incompatible" in statuses:
        return "excluded_incompatible"
    return "descriptive_unknown_configuration"


def compatibility_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], str]:
    """Index compatibility decisions by source observation and metric name."""
    output: dict[tuple[str, str, str], str] = {}
    for row in rows:
        record_id = row["canonical_record_id"]
        metric = record_id.rsplit(":", 1)[-1]
        output[(row["source_file"], row["source_row"], metric)] = row["compatibility_status"]
    return output


def _source_rows(sources: dict[str, list[dict[str, str]]], filename: str):
    return enumerate(sources[filename], start=1)


def baseline_table(sources, audit):
    rows = []
    metrics = ("relative_reconstruction_error", "residual_norm", "relative_residual")
    for index, row in _source_rows(sources, "phase2_baseline_comparison.csv"):
        rows.append({
            "source_file": "phase2_baseline_comparison.csv", "source_row": index,
            "method": row["solver"], "comparison_scope": scope_for_row(audit, "phase2_baseline_comparison.csv", index, metrics),
            "relative_reconstruction_error": row["relative_reconstruction_error"],
            "mean_squared_error": row["mean_squared_error"],
            "psnr_db": row["peak_signal_to_noise_ratio"], "ssim": row["structural_similarity_index"],
            "residual_norm": row["residual_norm"], "relative_residual": row["relative_residual"],
        })
    return rows


def regularization_table(sources, audit):
    rows = []
    metrics = ("relative_reconstruction_error", "residual_norm", "relative_residual")
    for index, row in _source_rows(sources, "phase4_regularization_comparison.csv"):
        if row["method"] not in {"TSVD", "Tikhonov SVD"}:
            continue
        rows.append({
            "source_file": "phase4_regularization_comparison.csv", "source_row": index,
            "method": row["method"], "parameter_name": row["parameter_name"],
            "parameter_value": row["parameter_value"],
            "comparison_scope": scope_for_row(audit, "phase4_regularization_comparison.csv", index, metrics),
            "relative_reconstruction_error": row["relative_reconstruction_error"],
            "mean_squared_error": row["mean_squared_error"], "psnr_db": row["peak_signal_to_noise_ratio"],
            "ssim": row["structural_similarity_index"], "residual_norm": row["residual_norm"],
            "relative_residual": row["relative_residual"],
            "configuration_note": "Phase 4 saved regularization outcome; configuration is audit-unknown.",
        })
    return rows


def parameter_selection_table(sources, audit):
    rows = []
    metrics = ("relative_reconstruction_error", "residual_norm", "relative_residual")
    for index, row in _source_rows(sources, "phase5_selection_summary.csv"):
        note = ""
        if row["method"] == "TSVD" and row["criterion"] == "gcv":
            note = "Documented high-rank GCV selection; assess as a noise-dominated failure case."
        rows.append({
            "source_file": "phase5_selection_summary.csv", "source_row": index,
            "method": row["method"], "criterion": row["criterion"],
            "selected_parameter": row["selected_parameter"],
            "comparison_scope": scope_for_row(audit, "phase5_selection_summary.csv", index, metrics),
            "relative_reconstruction_error": row["relative_reconstruction_error"],
            "mse": row["mse"], "psnr_db": row["psnr_db"], "ssim": row["ssim"],
            "residual_norm": row["residual_norm"], "relative_residual": row["relative_residual"],
            "failure_note": note,
        })
    return rows


def robustness_table(sources, audit):
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for index, row in _source_rows(sources, "phase6_raw_results.csv"):
        scope = scope_for_row(audit, "phase6_raw_results.csv", index, ("relative_reconstruction_error", "residual_norm"))
        if scope != "strict_compatible":
            continue
        groups[(row["sensitivity_axis"], row["condition_value"], row["method"], row["criterion"])].append(row)
    output = []
    for key in sorted(groups):
        group = groups[key]
        errors = np.array([number(row, "relative_reconstruction_error") for row in group], dtype=float)
        residuals = np.array([number(row, "residual_norm") for row in group], dtype=float)
        output.append({
            "sensitivity_axis": key[0], "condition_value": key[1], "method": key[2], "criterion": key[3],
            "n_unique_cases": len({row["case_id"] for row in group}),
            "mean_relative_reconstruction_error": float(np.nanmean(errors)),
            "standard_deviation_relative_reconstruction_error": float(np.nanstd(errors, ddof=0)),
            "mean_residual_norm": float(np.nanmean(residuals)),
            "unstable_case_count": sum(row["unstable_solution"] == "True" for row in group),
            "comparison_scope": "strict_compatible", "source_file": "phase6_raw_results.csv",
        })
    return output


def iterative_table(sources, audit):
    output = []
    metrics = ("relative_reconstruction_error", "residual_norm", "relative_residual", "execution_time_seconds")
    for index, row in _source_rows(sources, "phase7_iterative_results.csv"):
        if row["run_type"] != "iteration_budget":
            continue
        scope = scope_for_row(audit, "phase7_iterative_results.csv", index, metrics)
        if scope != "strict_compatible":
            continue
        output.append({
            "case_id": row["case_id"], "method": row["method"], "iteration_count": row["iteration_count"],
            "requested_iteration_limit": row["requested_iteration_limit"],
            "relative_reconstruction_error": row["relative_reconstruction_error"], "residual_norm": row["residual_norm"],
            "relative_residual": row["relative_residual"], "execution_time_seconds": row["execution_time_seconds"],
            "relative_tolerance": row["relative_tolerance"], "termination_status": row["termination_status"],
            "comparison_scope": scope, "source_file": "phase7_iterative_results.csv", "source_row": index,
        })
    return sorted(output, key=lambda row: (row["case_id"], row["method"], int(row["requested_iteration_limit"])))


def spectral_table(sources, audit):
    output = []
    for index, row in _source_rows(sources, "phase3_svd_summary.csv"):
        output.append({"evidence_type": "SVD summary", "source_file": "phase3_svd_summary.csv", "source_row": index,
            "comparison_scope": scope_for_row(audit, "phase3_svd_summary.csv", index, ("condition_number",)),
            "image_shape": f"{row['rows']}x{row['columns']}", "sigma": "", "numerical_rank": row["numerical_rank"],
            "condition_number": row["condition_number"], "largest_singular_value": row["largest_singular_value"],
            "smallest_singular_value": row["smallest_singular_value"], "storage_or_validation_note": row["diagnostic"]})
    for index, row in _source_rows(sources, "phase3_blur_sensitivity.csv"):
        output.append({"evidence_type": "blur sensitivity", "source_file": "phase3_blur_sensitivity.csv", "source_row": index,
            "comparison_scope": scope_for_row(audit, "phase3_blur_sensitivity.csv", index, ("condition_number",)),
            "image_shape": f"{row['rows']}x{row['columns']}", "sigma": row["sigma"], "numerical_rank": row["numerical_rank"],
            "condition_number": row["condition_number"], "largest_singular_value": row["largest_singular_value"],
            "smallest_singular_value": row["smallest_singular_value"], "storage_or_validation_note": "Different blur strengths; not an independent reconstruction comparison."})
    for index, row in _source_rows(sources, "phase3_structured_vs_full_svd.csv"):
        output.append({"evidence_type": "structured SVD validation", "source_file": "phase3_structured_vs_full_svd.csv", "source_row": index,
            "comparison_scope": scope_for_row(audit, "phase3_structured_vs_full_svd.csv", index, ("full_min_seconds",)),
            "image_shape": f"{row['rows']}x{row['columns']}", "sigma": "", "numerical_rank": "", "condition_number": "",
            "largest_singular_value": "", "smallest_singular_value": "",
            "storage_or_validation_note": f"max spectrum difference={row['maximum_absolute_spectrum_difference']}; full bytes={row['dense_operator_input_bytes']}; structured bytes={row['structured_factor_input_bytes']}"})
    return output


def scalability_table(sources, audit):
    output = []
    metrics = ("execution_time_seconds", "relative_reconstruction_error")
    for index, row in _source_rows(sources, "phase7_iterative_results.csv"):
        if row["run_type"] != "iteration_budget":
            continue
        scope = scope_for_row(audit, "phase7_iterative_results.csv", index, metrics)
        if scope != "strict_compatible":
            continue
        output.append({
            "case_id": row["case_id"], "image_shape": f"{row['image_height']}x{row['image_width']}", "method": row["method"],
            "iteration_count": row["iteration_count"], "runtime_seconds": row["execution_time_seconds"],
            "theoretical_dense_operator_bytes": row["theoretical_dense_operator_bytes"],
            "theoretical_separable_factor_bytes": row["theoretical_separable_factor_bytes"],
            "storage_interpretation": "Theoretical float64 array storage; peak RAM was not measured.",
            "comparison_scope": scope, "source_file": "phase7_iterative_results.csv",
        })
    return sorted(output, key=lambda row: (row["image_shape"], row["method"], int(row["iteration_count"])))


def _save_figure(path: Path, title: str, xlabel: str, ylabel: str, *, log_y: bool = False):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_y:
        plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def generate_figures(sources: dict[str, list[dict[str, str]]], figure_dir: Path) -> None:
    """Generate only figures based on saved source data, with scope in titles."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    p = lambda name: figure_dir / name
    baseline = sources["phase2_baseline_comparison.csv"]
    plt.figure(figsize=(8, 4.8)); names = [r["solver"] for r in baseline]
    plt.bar(names, [number(r, "relative_reconstruction_error") for r in baseline], label="relative reconstruction error")
    _save_figure(p(FIGURE_FILENAMES[0]), "Figure 1. Baseline inverse methods (descriptive; audit-unknown)", "Method", "Relative reconstruction error", log_y=True)

    reg = [r for r in sources["phase4_regularization_comparison.csv"] if r["method"] in {"TSVD", "Tikhonov SVD"}]
    plt.figure(figsize=(7, 5));
    for method in sorted({r["method"] for r in reg}):
        group = [r for r in reg if r["method"] == method]
        plt.scatter([number(r, "residual_norm") for r in group], [number(r, "relative_reconstruction_error") for r in group], label=method)
    plt.legend(); _save_figure(p(FIGURE_FILENAMES[1]), "Figure 2. Regularization outcomes (descriptive; audit-unknown)", "Residual norm", "Relative reconstruction error", log_y=True)

    curves = sources["phase5_selection_curves.csv"]
    plt.figure(figsize=(7, 5));
    for method in ("TSVD", "Tikhonov"):
        group = [r for r in curves if r["method"] == method]
        plt.plot([number(r, "residual_norm") for r in group], [number(r, "solution_norm") for r in group], marker=".", label=method)
    plt.xscale("log"); plt.yscale("log"); plt.legend(); _save_figure(p(FIGURE_FILENAMES[2]), "Figure 3. L-curves (descriptive; parameter candidates)", "Residual norm", "Solution norm")

    plt.figure(figsize=(8, 5));
    for method in ("TSVD", "Tikhonov"):
        group = [r for r in curves if r["method"] == method]
        plt.plot([number(r, "parameter") for r in group], [number(r, "gcv") for r in group], label=f"{method} GCV")
        plt.plot([number(r, "parameter") for r in group], [number(r, "discrepancy_absolute_error") for r in group], linestyle="--", label=f"{method} discrepancy error")
    plt.legend(fontsize=8); _save_figure(p(FIGURE_FILENAMES[3]), "Figure 4. GCV and discrepancy diagnostics (descriptive)", "Candidate parameter", "Criterion value", log_y=True)

    raw = sources["phase6_raw_results.csv"]
    for fig_index, axis, title, x_label in ((4, "noise", "Figure 5. Noise-seed robustness (strict compatible)", "Noise seed"), (5, "blur", "Figure 6. Blur-strength sensitivity (strict compatible)", "Gaussian sigma"), (6, "resolution", "Figure 7. Resolution sensitivity (strict compatible)", "Pixels per side")):
        plt.figure(figsize=(8, 5))
        group = [r for r in raw if r["sensitivity_axis"] == axis]
        for label in sorted({f"{r['method']} / {r['criterion']}" for r in group}):
            data = [r for r in group if f"{r['method']} / {r['criterion']}" == label]
            x = [number(r, "random_seed" if axis == "noise" else "sigma" if axis == "blur" else "image_height") for r in data]
            order = np.argsort(x); plt.plot(np.array(x)[order], np.array([number(r, "relative_reconstruction_error") for r in data])[order], marker="o", label=label)
        plt.legend(fontsize=7); _save_figure(p(FIGURE_FILENAMES[fig_index]), title, x_label, "Relative reconstruction error")

    iterative = sources["phase7_iterative_results.csv"]
    trajectories = [r for r in iterative if r["run_type"] == "iteration_trajectory"]
    for fig_index, metric, title, ylabel in ((7, "residual_norm", "Figure 8. Iterative convergence trajectories (strict compatible)", "Residual norm"), (8, "relative_reconstruction_error", "Figure 9. Reconstruction error versus iteration (strict compatible)", "Relative reconstruction error")):
        plt.figure(figsize=(8, 5))
        for method in sorted({r["method"] for r in trajectories}):
            data = [r for r in trajectories if r["method"] == method]
            plt.plot([number(r, "iteration_count") for r in data], [number(r, metric) for r in data], marker="o", label=method)
        plt.legend(); _save_figure(p(FIGURE_FILENAMES[fig_index]), title, "Iteration", ylabel, log_y=True)

    spectrum = sources["phase3_svd_spectrum.csv"]
    plt.figure(figsize=(7, 5)); plt.semilogy([number(r, "index") for r in spectrum], [number(r, "singular_value") for r in spectrum])
    _save_figure(p(FIGURE_FILENAMES[9]), "Figure 10. Singular-value decay (descriptive spectral evidence)", "Component index", "Singular value")

    budgets = [r for r in iterative if r["run_type"] == "iteration_budget"]
    plt.figure(figsize=(8, 5))
    for method in sorted({r["method"] for r in budgets}):
        data = [r for r in budgets if r["method"] == method]
        plt.scatter([number(r, "pixel_count") for r in data], [number(r, "execution_time_seconds") for r in data], label=method)
    plt.legend(); _save_figure(p(FIGURE_FILENAMES[10]), "Figure 11. Iterative runtime evidence (mixed compatibility; descriptive)", "Pixel count", "Execution time (s)")

    metadata = sources["phase7_case_metadata.csv"]
    plt.figure(figsize=(7, 5)); x = [int(r["image_shape"].split("x")[0]) ** 2 for r in metadata]
    plt.loglog(x, [float(r["image_shape"].split("x")[0]) ** 4 * 8 for r in metadata], marker="o", label="theoretical dense operator storage")
    plt.loglog(x, [2 * float(r["image_shape"].split("x")[0]) ** 2 * 8 for r in metadata], marker="o", label="theoretical separable factors")
    plt.legend(fontsize=8); _save_figure(p(FIGURE_FILENAMES[11]), "Figure 12. Theoretical storage scaling (not measured peak RAM)", "Pixel count", "Float64 bytes")


def run_comparative_analysis_stage(metrics_dir: Path, phase8_dir: Path, figure_dir: Path) -> None:
    """Create P8-04 outputs from persisted evidence only; never rerun experiments."""
    required = [phase8_dir / name for name in (
        "phase8_canonical_long_form.csv", "phase8_record_type_audit.csv",
        "phase8_compatibility_audit.csv", "phase8_reconciled_metadata.csv",
    )]
    required.extend(metrics_dir / name for name in SOURCE_FILES)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Comparative analysis is missing required saved evidence: " + ", ".join(missing))
    classification = read_csv(phase8_dir / "phase8_record_type_audit.csv")
    if any(row["classification_status"] != "classified" for row in classification):
        raise ValueError("Record-type audit contains unresolved classifications.")
    canonical_ids = {row["canonical_record_id"] for row in read_csv(phase8_dir / "phase8_canonical_long_form.csv")}
    audit_rows = read_csv(phase8_dir / "phase8_compatibility_audit.csv")
    if any(row["canonical_record_id"] not in canonical_ids for row in audit_rows):
        raise ValueError("Compatibility audit contains IDs absent from canonical evidence.")
    sources = {name: read_csv(metrics_dir / name) for name in SOURCE_FILES}
    audit = compatibility_index(audit_rows)
    tables = {
        "phase8_table1_baseline_inverse_methods.csv": baseline_table(sources, audit),
        "phase8_table2_regularization_methods.csv": regularization_table(sources, audit),
        "phase8_table3_parameter_selection.csv": parameter_selection_table(sources, audit),
        "phase8_table4_robustness_sensitivity.csv": robustness_table(sources, audit),
        "phase8_table5_iterative_solver_comparison.csv": iterative_table(sources, audit),
        "phase8_table6_spectral_conditioning.csv": spectral_table(sources, audit),
        "phase8_table7_scalability.csv": scalability_table(sources, audit),
    }
    for name, rows in tables.items():
        write_csv(phase8_dir / name, TABLE_SPECS[name], rows)
    status_counts = defaultdict(int)
    for row in audit_rows:
        status_counts[row["compatibility_status"]] += 1
    write_csv(phase8_dir / "phase8_comparative_analysis_audit.csv", ("compatibility_status", "canonical_metric_records"), [
        {"compatibility_status": status, "canonical_metric_records": status_counts[status]}
        for status in ("compatible", "incompatible", "unknown")
    ])
    generate_figures(sources, figure_dir)
