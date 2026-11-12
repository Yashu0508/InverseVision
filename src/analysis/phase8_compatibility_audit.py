"""Purpose-specific, evidence-based compatibility audit for Phase 8 records."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path


AUDIT_FIELDS = (
    "comparison_id",
    "comparison_purpose",
    "source_phase",
    "source_file",
    "source_row",
    "canonical_record_id",
    "analytical_category",
    "assigned_record_type",
    "configuration_signature_json",
    "compatibility_status",
    "matched_configuration_dimensions",
    "mismatched_configuration_dimensions",
    "unknown_configuration_dimensions",
    "reason",
    "evidence_provenance",
)

CORE_DIMENSIONS = (
    "image_shape",
    "blur_sigma",
    "kernel_size",
    "boundary",
    "noise_standard_deviation",
    "random_seed",
)


def _attributes(record: dict[str, str]) -> dict[str, str]:
    return json.loads(record["source_attributes_json"])


def configuration_signature(record: dict[str, str]) -> dict[str, str]:
    """Return only explicit configuration values recorded with this observation."""
    attributes = _attributes(record)
    image_shape = record.get("image_shape", "")
    if not image_shape and record.get("image_height") and record.get("image_width"):
        image_shape = f"{record['image_height']}x{record['image_width']}"
    return {
        "image_shape": image_shape,
        "blur_sigma": record.get("blur_sigma", ""),
        "kernel_size": record.get("kernel_size", ""),
        "boundary": record.get("boundary", ""),
        "noise_standard_deviation": record.get("noise_standard_deviation", ""),
        "random_seed": record.get("random_seed", ""),
        "numerical_rank_tolerance": attributes.get(
            "relative_rank_tolerance", attributes.get("relative_tolerance", "")
        ),
        "run_type": attributes.get("run_type", ""),
        "requested_iteration_limit": attributes.get("requested_iteration_limit", ""),
        "relative_tolerance": attributes.get("relative_tolerance", ""),
        "absolute_tolerance": attributes.get("absolute_tolerance", ""),
        "damping_parameter_lambda": attributes.get("damping_parameter_lambda", ""),
        "parameter_selection_criterion": record.get("parameter_selection_method", ""),
        "parameter_name": record.get("parameter_name", ""),
        "parameter_value": record.get("parameter_value", ""),
        "case_id": record.get("case_id", ""),
        "observation_identity": record.get("experiment_id", ""),
    }


def evaluate_configuration(
    reference: dict[str, str],
    candidate: dict[str, str],
    required_dimensions: tuple[str, ...],
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Compare explicit dimensions; absence is unknown rather than a match."""
    matched: list[str] = []
    mismatched: list[str] = []
    unknown: list[str] = []
    for dimension in required_dimensions:
        left, right = reference.get(dimension, ""), candidate.get(dimension, "")
        if not left or not right:
            unknown.append(dimension)
        elif left == right:
            matched.append(dimension)
        else:
            mismatched.append(dimension)
    status = "unknown" if unknown else "incompatible" if mismatched else "compatible"
    return status, tuple(matched), tuple(mismatched), tuple(unknown)


def _plan(record: dict[str, str], classification: dict[str, str]) -> tuple[str, str, tuple[str, ...], str]:
    """Choose a narrow, scientifically meaningful audit purpose for one record."""
    source_file = record["source_file"]
    assigned = classification["assigned_record_type"]
    attributes = _attributes(record)
    if assigned == "noise_seed_individual_case_result":
        return (
            f"phase6:noise-seed:{record['method']}:{record['parameter_selection_method']}",
            "controlled_noise_seed_robustness",
            ("image_shape", "blur_sigma", "kernel_size", "boundary", "noise_standard_deviation"),
            "Noise seed is intentionally varied; other listed settings must match.",
        )
    if assigned == "blur_condition_individual_case_result":
        return (
            f"phase6:blur:{record['method']}:{record['parameter_selection_method']}",
            "controlled_blur_sensitivity",
            ("image_shape", "kernel_size", "boundary", "noise_standard_deviation", "random_seed"),
            "Blur sigma is intentionally varied; other listed settings must match.",
        )
    if assigned == "resolution_individual_case_result":
        return (
            f"phase6:resolution:{record['method']}:{record['parameter_selection_method']}",
            "controlled_resolution_sensitivity",
            ("blur_sigma", "kernel_size", "boundary", "noise_standard_deviation", "random_seed"),
            "Resolution is intentionally varied; other listed settings must match.",
        )
    if assigned == "aggregated_sensitivity_summary":
        return (
            f"phase6:aggregate:{attributes.get('sensitivity_axis', '')}:{record['method']}:{record['parameter_selection_method']}",
            "descriptive_aggregate_within_sensitivity_axis",
            (),
            "Aggregate summaries are audited separately and are not individual replications.",
        )
    if assigned == "iteration_history_observation":
        return (
            f"phase7:trajectory:{record['case_id']}:{record['method']}",
            "within_solver_convergence_trajectory",
            ("image_shape", "blur_sigma", "kernel_size", "boundary", "noise_standard_deviation", "random_seed", "run_type"),
            "Iteration is intentionally varied within one solver trajectory.",
        )
    if assigned == "iteration_budget_solver_observation":
        return (
            f"phase7:equal-budget:{record['case_id']}",
            "equal_iteration_budget_solver_comparison",
            ("image_shape", "blur_sigma", "kernel_size", "boundary", "noise_standard_deviation", "random_seed", "run_type", "requested_iteration_limit"),
            "Solver comparisons require a common case and requested iteration budget.",
        )
    if assigned == "tolerance_stopping_observation":
        return (
            f"phase7:tolerance:{record['case_id']}:{record['method']}",
            "within_solver_tolerance_sensitivity",
            ("image_shape", "blur_sigma", "kernel_size", "boundary", "noise_standard_deviation", "random_seed", "run_type"),
            "Requested tolerance is intentionally varied within one solver and case.",
        )
    if assigned == "experimental_metadata_diagnostic":
        return (
            f"metadata:{source_file}:{record['source_row']}",
            "metadata_diagnostic_only",
            (),
            "Metadata diagnostics are not numerical performance-comparison candidates.",
        )
    if assigned in {"parameter_candidate_evaluation", "selected_parameter_result", "selected_parameter_by_case"}:
        return (
            f"selection:{source_file}:{record['method']}:{record['parameter_selection_method']}",
            "parameter_selection_within_source",
            ("image_shape", "blur_sigma", "kernel_size", "boundary", "noise_standard_deviation", "random_seed"),
            "Selection records are audited within their source and never paired with candidate records as selections.",
        )
    if source_file == "phase3_blur_sensitivity.csv":
        return (
            "phase3:controlled-blur-spectrum",
            "spectral_blur_sensitivity",
            ("kernel_size",),
            "Blur sigma is intentionally varied in this spectral diagnostic.",
        )
    return (
        f"unresolved:{source_file}",
        "source_specific_descriptive_only",
        CORE_DIMENSIONS,
        "Configuration evidence is insufficient for a defensible cross-record comparison purpose.",
    )


def build_compatibility_audit(
    canonical_records: Iterable[dict[str, str]],
    classification_rows: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    """Create one compatibility outcome for each canonical observation."""
    classifications = {
        row["canonical_record_id"]: row for row in classification_rows
    }
    planned: list[tuple[dict[str, str], dict[str, str], dict[str, str], tuple[str, str, tuple[str, ...], str]]] = []
    for record in canonical_records:
        classification = classifications[record["canonical_record_id"]]
        signature = configuration_signature(record)
        planned.append((record, classification, signature, _plan(record, classification)))
    groups: dict[str, list[tuple[dict[str, str], dict[str, str], dict[str, str], tuple[str, str, tuple[str, ...], str]]]] = defaultdict(list)
    for item in planned:
        groups[item[3][0]].append(item)

    audit: list[dict[str, str]] = []
    for comparison_id in sorted(groups):
        group = sorted(groups[comparison_id], key=lambda item: item[0]["canonical_record_id"])
        reference = group[0][2]
        for record, classification, signature, (_, purpose, required, base_reason) in group:
            if purpose == "metadata_diagnostic_only":
                status, matched, mismatched, unknown = "unknown", (), (), ()
                reason = base_reason
            elif not required:
                status, matched, mismatched, unknown = "compatible", (), (), ()
                reason = base_reason
            else:
                status, matched, mismatched, unknown = evaluate_configuration(reference, signature, required)
                reason = base_reason
                if status == "unknown":
                    reason += " Required configuration evidence is missing."
                elif status == "incompatible":
                    reason += " Required configuration values differ."
            audit.append(
                {
                    "comparison_id": comparison_id,
                    "comparison_purpose": purpose,
                    "source_phase": record["source_phase"],
                    "source_file": record["source_file"],
                    "source_row": record["source_row"],
                    "canonical_record_id": record["canonical_record_id"],
                    "analytical_category": classification["analytical_category"],
                    "assigned_record_type": classification["assigned_record_type"],
                    "configuration_signature_json": json.dumps(signature, sort_keys=True),
                    "compatibility_status": status,
                    "matched_configuration_dimensions": ";".join(matched),
                    "mismatched_configuration_dimensions": ";".join(mismatched),
                    "unknown_configuration_dimensions": ";".join(unknown),
                    "reason": reason,
                    "evidence_provenance": "canonical fields plus source_attributes_json; source file and row retained",
                }
            )
    return audit


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)


def run_compatibility_audit_stage(
    canonical_path: Path, classification_path: Path, output_path: Path
) -> list[dict[str, str]]:
    """Write only the P8-02 compatibility audit."""
    canonical_records = _read_csv(canonical_path)
    audit = build_compatibility_audit(canonical_records, _read_csv(classification_path))
    canonical_ids = {record["canonical_record_id"] for record in canonical_records}
    audit_ids = {record["canonical_record_id"] for record in audit}
    if len(audit) != len(canonical_ids) or audit_ids != canonical_ids:
        raise ValueError("Compatibility audit must contain exactly one outcome per canonical ID")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path, audit)
    return audit
