"""Analytical record-type taxonomy for Phase 8 canonical observations."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path


AUDIT_FIELDS = (
    "source_phase",
    "source_file",
    "source_row",
    "canonical_record_id",
    "existing_record_type",
    "assigned_record_type",
    "analytical_category",
    "classification_rule",
    "classification_status",
    "reason",
)


def classify_record(record: dict[str, str]) -> dict[str, str]:
    """Classify one canonical observation from source-specific context.

    The result describes analytical role only. It does not assert configuration
    compatibility or statistical independence beyond differentiating summaries,
    candidate evaluations, trajectories, and individual case observations.
    """
    source_file = record["source_file"]
    assigned, category, rule, status, reason = _classification(source_file, record)
    return {
        "source_phase": record["source_phase"],
        "source_file": source_file,
        "source_row": record["source_row"],
        "canonical_record_id": record["canonical_record_id"],
        "existing_record_type": record["record_type"],
        "assigned_record_type": assigned,
        "analytical_category": category,
        "classification_rule": rule,
        "classification_status": status,
        "reason": reason,
    }


def _classification(
    source_file: str, record: dict[str, str]
) -> tuple[str, str, str, str, str]:
    if source_file == "phase2_baseline_comparison.csv":
        return (
            "unregularized_baseline_reconstruction_result",
            "reconstruction_baseline",
            "phase2_baseline_adapter",
            "classified",
            "Direct, QR, and normal-equation metrics are baseline reconstruction results.",
        )
    if source_file in {
        "phase3_svd_spectrum.csv",
        "phase3_picard_noise.csv",
    }:
        kind = "singular_value_observation" if source_file.endswith("svd_spectrum.csv") else "picard_noise_diagnostic"
        return (kind, "spectral_conditioning", "phase3_component_adapter", "classified", "Spectral component diagnostic.")
    if source_file in {
        "phase3_svd_summary.csv",
        "phase3_blur_sensitivity.csv",
    }:
        return (
            "spectral_or_conditioning_summary",
            "spectral_conditioning",
            "phase3_summary_adapter",
            "classified",
            "Phase 3 summary; not a reconstruction-method result.",
        )
    if source_file == "phase3_structured_vs_full_svd.csv":
        return (
            "structured_svd_validation_diagnostic",
            "spectral_conditioning",
            "phase3_structured_validation_adapter",
            "classified",
            "Agreement, storage, and timing validation for structured SVD analysis.",
        )
    if source_file == "phase4_regularization_comparison.csv":
        return (
            "regularization_comparison_result",
            "explicit_regularization",
            "phase4_comparison_adapter",
            "classified",
            "Phase 4 comparison row retained as a reconstruction result.",
        )
    if source_file in {"phase4_tsvd_results.csv", "phase4_tikhonov_results.csv"}:
        return (
            "explicit_regularization_reconstruction_result",
            "explicit_regularization",
            "phase4_method_adapter",
            "classified",
            "TSVD or Tikhonov reconstruction metric at an explicit parameter.",
        )
    if source_file == "phase5_selection_curves.csv":
        return (
            "parameter_candidate_evaluation",
            "parameter_selection",
            "phase5_candidate_curve_adapter",
            "classified",
            "Candidate curve point; it is not a selected parameter.",
        )
    if source_file == "phase5_selection_summary.csv":
        return (
            "selected_parameter_result",
            "parameter_selection",
            "phase5_selection_summary_adapter",
            "classified",
            "Selected parameter and post-selection evaluation metric.",
        )
    if source_file == "phase6_raw_results.csv":
        attributes = json.loads(record["source_attributes_json"])
        if attributes.get("sensitivity_axis") == "noise":
            assigned = "noise_seed_individual_case_result"
        elif attributes.get("sensitivity_axis") == "blur":
            assigned = "blur_condition_individual_case_result"
        elif attributes.get("sensitivity_axis") == "resolution":
            assigned = "resolution_individual_case_result"
        else:
            assigned = "unclassified_phase6_case_result"
        status = "classified" if not assigned.startswith("unclassified") else "classification_pending"
        return (
            assigned,
            "robustness_sensitivity",
            "phase6_raw_case_adapter",
            status,
            "Individual case-level result; not an aggregate replication." if status == "classified" else "Unknown Phase 6 sensitivity axis.",
        )
    if source_file == "phase6_aggregated_summary.csv":
        return (
            "aggregated_sensitivity_summary",
            "robustness_sensitivity",
            "phase6_aggregate_adapter",
            "classified",
            "Descriptive aggregate; it is not an independent experiment.",
        )
    if source_file == "phase6_selected_parameters.csv":
        return (
            "selected_parameter_by_case",
            "robustness_sensitivity",
            "phase6_selection_adapter",
            "classified",
            "Selection outcome tied to a Phase 6 sensitivity case.",
        )
    if source_file == "phase7_case_metadata.csv":
        return (
            "experimental_metadata_diagnostic",
            "experimental_metadata",
            "phase7_metadata_adapter",
            "classified",
            "Case metadata or consistency diagnostic, not solver performance.",
        )
    if source_file == "phase7_small_system_validation.csv":
        return (
            "small_system_iterative_validation",
            "iterative_solver_validation",
            "phase7_small_validation_adapter",
            "classified",
            "Small explicit reference validation, separate from scalable runs.",
        )
    if source_file == "phase7_iterative_results.csv":
        attributes = json.loads(record["source_attributes_json"])
        if attributes.get("run_type") == "iteration_trajectory":
            assigned = "iteration_history_observation"
        elif attributes.get("run_type") == "tolerance":
            assigned = "tolerance_stopping_observation"
        elif attributes.get("run_type") == "iteration_budget":
            assigned = "iteration_budget_solver_observation"
        else:
            assigned = "unclassified_iterative_solver_observation"
        status = "classified" if not assigned.startswith("unclassified") else "classification_pending"
        return (
            assigned,
            "iterative_solvers",
            "phase7_run_type_adapter",
            status,
            "Run type preserves trajectory/stopping/budget distinction." if status == "classified" else "Unknown Phase 7 run type.",
        )
    return (
        "unclassified",
        "unclassified",
        "no_matching_source_adapter",
        "classification_pending",
        "No source-specific classification rule is defined.",
    )


def classify_records(records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Classify all canonical records deterministically."""
    return [classify_record(record) for record in records]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def run_record_type_classification_stage(
    canonical_path: Path, output_path: Path
) -> list[dict[str, str]]:
    """Write only the record-type classification audit."""
    audit = classify_records(_read_csv(canonical_path))
    if len(audit) != len({row["canonical_record_id"] for row in audit}):
        raise ValueError("Each canonical record must have one audit outcome")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path, audit)
    return audit
