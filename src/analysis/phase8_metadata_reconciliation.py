"""Evidence-backed metadata quality audit and non-mutating overlay for Phase 8."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


AUDIT_FIELDS = (
    "metadata_audit_id", "canonical_record_id", "source_phase", "source_file",
    "source_row", "metadata_field", "original_canonical_value",
    "reconciled_metadata_value", "metadata_status", "evidence_source",
    "evidence_location", "reconciliation_explanation", "conflict_details",
)
OVERLAY_FIELDS = (
    "reconciled_metadata_id", "canonical_record_id", "metadata_field",
    "reconciled_metadata_value", "metadata_status", "evidence_source",
    "evidence_location", "reconciliation_explanation",
)
FIELDS = (
    "configuration_identifier", "image_dimensions", "source_image_identity",
    "blur_model", "blur_sigma", "kernel_size", "boundary_condition",
    "noise_model", "noise_standard_deviation", "noise_seed",
    "observation_identity", "numerical_rank_tolerance", "method_or_solver",
    "stopping_or_iteration_setting",
)

STATIC_PROFILES = {
    "phase2_baseline_comparison.csv": {
        "image_dimensions": "12x12", "source_image_identity": "skimage.data.camera_normalized",
        "blur_model": "separable_gaussian_zero_boundary", "blur_sigma": "1.5",
        "kernel_size": "9", "boundary_condition": "zero", "noise_model": "additive_gaussian",
        "noise_standard_deviation": "0.01", "noise_seed": "42",
        "observation_identity": "phase2_shared_observation",
        "evidence": "experiments/phase2_baseline_reconstruction.py",
        "location": "EXPERIMENT_SHAPE and main degradation/operator calls",
    },
    "phase4_regularization_comparison.csv": {},
    "phase4_tsvd_results.csv": {},
    "phase4_tikhonov_results.csv": {},
    "phase5_selection_curves.csv": {},
    "phase5_selection_summary.csv": {},
    "phase3_svd_summary.csv": {
        "image_dimensions": "8x8", "blur_model": "separable_gaussian_zero_boundary",
        "blur_sigma": "1.5", "kernel_size": "9", "boundary_condition": "zero",
        "numerical_rank_tolerance": "1e-12", "evidence": "report/project_handoff_phase_0_phase_3.md",
        "location": "Computed SVD demonstration",
    },
    "phase3_svd_spectrum.csv": {
        "image_dimensions": "8x8", "blur_model": "separable_gaussian_zero_boundary",
        "blur_sigma": "1.5", "kernel_size": "9", "boundary_condition": "zero",
        "numerical_rank_tolerance": "1e-12", "evidence": "report/project_handoff_phase_0_phase_3.md",
        "location": "Computed SVD demonstration",
    },
    "phase3_picard_noise.csv": {
        "image_dimensions": "8x8", "source_image_identity": "skimage.data.camera_normalized",
        "blur_model": "separable_gaussian_zero_boundary", "blur_sigma": "1.5",
        "kernel_size": "9", "boundary_condition": "zero", "noise_model": "additive_gaussian",
        "noise_standard_deviation": "0.01", "noise_seed": "42",
        "numerical_rank_tolerance": "1e-12", "observation_identity": "phase3_picard_observation",
        "evidence": "report/project_handoff_phase_0_phase_3.md", "location": "Picard and noise findings",
    },
    "phase3_structured_vs_full_svd.csv": {
        "image_dimensions": "8x8", "blur_model": "separable_gaussian_zero_boundary",
        "blur_sigma": "1.5", "kernel_size": "9", "boundary_condition": "zero",
        "numerical_rank_tolerance": "1e-12", "evidence": "report/project_handoff_phase_0_phase_3.md",
        "location": "Structured versus full SVD",
    },
}

PHASE45_PROFILE = {
    "image_dimensions": "12x12", "source_image_identity": "skimage.data.camera_normalized",
    "blur_model": "separable_gaussian_zero_boundary", "blur_sigma": "1.5",
    "kernel_size": "9", "boundary_condition": "zero", "noise_model": "additive_gaussian",
    "noise_standard_deviation": "0.01", "noise_seed": "42",
    "numerical_rank_tolerance": "1e-12", "observation_identity": "phase4_phase5_shared_observation",
    "evidence": "report/project_handoff_phase_0_phase_5.md",
    "location": "Shared Phase 4/5 protocol",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _phase6_cases(config_path: Path) -> dict[str, dict[str, str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        item["case_id"]: {
            "image_dimensions": "x".join(map(str, item["image_shape"])),
            "source_image_identity": config["image_source"],
            "blur_model": "separable_gaussian_zero_boundary",
            "blur_sigma": str(item["sigma"]), "kernel_size": str(item["kernel_size"]),
            "boundary_condition": "zero", "noise_model": "additive_gaussian",
            "noise_standard_deviation": str(item["noise_standard_deviation"]),
            "noise_seed": str(item["random_seed"]),
            "numerical_rank_tolerance": str(config["rank_relative_tolerance"]),
            "observation_identity": f"phase6:{item['case_id']}",
            "evidence": "results/metrics/phase6_experiment_config.json",
            "location": f"cases[{item['case_id']}]",
        }
        for item in config["cases"]
    }


def _phase7_cases(config_path: Path) -> dict[str, dict[str, str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        item["case_id"]: {
            "image_dimensions": "x".join(map(str, item["image_shape"])),
            "source_image_identity": "skimage.data.camera_normalized",
            "blur_model": "separable_gaussian_zero_boundary",
            "blur_sigma": str(item["sigma"]), "kernel_size": str(config["kernel_size"]),
            "boundary_condition": "zero", "noise_model": "additive_gaussian",
            "noise_standard_deviation": str(config["noise_standard_deviation"]),
            "noise_seed": str(config["random_seed"]),
            "observation_identity": f"phase7:{item['case_id']}",
            "evidence": "results/metrics/phase7_experiment_config.json",
            "location": f"cases[{item['case_id']}]",
        }
        for item in config["cases"]
    }


def _canonical_value(record: dict[str, str], field: str) -> str:
    mapping = {
        "configuration_identifier": "case_id", "image_dimensions": "image_shape",
        "blur_sigma": "blur_sigma", "kernel_size": "kernel_size",
        "boundary_condition": "boundary", "noise_standard_deviation": "noise_standard_deviation",
        "noise_seed": "random_seed", "method_or_solver": "method",
    }
    if field == "method_or_solver":
        return record.get("method", "") or record.get("solver", "")
    if field == "stopping_or_iteration_setting":
        attributes = json.loads(record["source_attributes_json"])
        return attributes.get("run_type", "") or record.get("iteration", "")
    if field == "numerical_rank_tolerance":
        attributes = json.loads(record["source_attributes_json"])
        return attributes.get("relative_rank_tolerance", attributes.get("relative_tolerance", ""))
    return record.get(mapping.get(field, ""), "")


def _not_applicable(record: dict[str, str], field: str) -> bool:
    source = record["source_file"]
    if field in {"noise_model", "noise_standard_deviation", "noise_seed", "observation_identity"}:
        return source in {
            "phase3_svd_summary.csv", "phase3_svd_spectrum.csv",
            "phase3_structured_vs_full_svd.csv", "phase3_blur_sensitivity.csv",
        }
    if field in {"stopping_or_iteration_setting"}:
        return not source.startswith("phase7_")
    if field == "source_image_identity":
        return source in {
            "phase3_svd_summary.csv", "phase3_svd_spectrum.csv",
            "phase3_structured_vs_full_svd.csv", "phase3_blur_sensitivity.csv",
        }
    return False


def _recovery_profile(record: dict[str, str], phase6: dict[str, dict[str, str]], phase7: dict[str, dict[str, str]]) -> dict[str, str]:
    source = record["source_file"]
    if source in {
        "phase4_regularization_comparison.csv", "phase4_tsvd_results.csv",
        "phase4_tikhonov_results.csv", "phase5_selection_curves.csv", "phase5_selection_summary.csv",
    }:
        return PHASE45_PROFILE
    if source in STATIC_PROFILES:
        return STATIC_PROFILES[source]
    if source.startswith("phase6_"):
        return phase6.get(record.get("case_id", ""), {})
    if source.startswith("phase7_"):
        return phase7.get(record.get("case_id", ""), {})
    return {}


def _audit_row(record: dict[str, str], field: str, profile: dict[str, str]) -> dict[str, str]:
    original = _canonical_value(record, field)
    audit_id = f"{record['canonical_record_id']}:{field}"
    if original and field in profile and original != profile[field]:
        status, value, evidence, location, explanation, conflict = (
            "conflicting", "", profile["evidence"], profile["location"],
            "Canonical and attributable evidence disagree; neither value is silently selected.",
            f"canonical={original}; evidence={profile[field]}",
        )
    elif original:
        status, value, evidence, location, explanation = (
            "present_verified", original, "canonical observation", field,
            "Explicitly represented in the canonical observation.",
        )
        conflict = ""
    elif _not_applicable(record, field):
        status, value, evidence, location, explanation = (
            "not_applicable", "", "record-type applicability rule", record["record_type"],
            "This metadata dimension does not apply to the analytical role.",
        )
        conflict = ""
    elif field in profile:
        status, value, evidence, location, explanation = (
            "reconciled", profile[field], profile["evidence"], profile["location"],
            "Recovered from explicitly attributed repository evidence; canonical data is unchanged.",
        )
        conflict = ""
    else:
        status, value, evidence, location, explanation = (
            "missing", "", "", "", "No record-specific attributable evidence was found.",
        )
        conflict = ""
    return {
        "metadata_audit_id": audit_id, "canonical_record_id": record["canonical_record_id"],
        "source_phase": record["source_phase"], "source_file": record["source_file"],
        "source_row": record["source_row"], "metadata_field": field,
        "original_canonical_value": original, "reconciled_metadata_value": value,
        "metadata_status": status, "evidence_source": evidence, "evidence_location": location,
        "reconciliation_explanation": explanation, "conflict_details": conflict,
    }


def build_metadata_audits(
    canonical_records: list[dict[str, str]], phase6_config: Path, phase7_config: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return field-level quality audit and only explicitly recovered overlay values."""
    phase6, phase7 = _phase6_cases(phase6_config), _phase7_cases(phase7_config)
    audit: list[dict[str, str]] = []
    for record in canonical_records:
        profile = _recovery_profile(record, phase6, phase7)
        audit.extend(_audit_row(record, field, profile) for field in FIELDS)
    mapping: list[dict[str, str]] = []
    for row in audit:
        if row["metadata_status"] == "reconciled":
            mapping.append({
                "reconciled_metadata_id": row["metadata_audit_id"],
                "canonical_record_id": row["canonical_record_id"],
                "metadata_field": row["metadata_field"],
                "reconciled_metadata_value": row["reconciled_metadata_value"],
                "metadata_status": row["metadata_status"],
                "evidence_source": row["evidence_source"],
                "evidence_location": row["evidence_location"],
                "reconciliation_explanation": row["reconciliation_explanation"],
            })
    return audit, mapping


def add_unmapped_source_field_rows(
    audit: list[dict[str, str]], mapping_audit_path: Path
) -> None:
    """Surface schema-audit fields retained only in raw source attributes."""
    for row in _read_csv(mapping_audit_path):
        if row["mapping_justified"] != "no":
            continue
        audit.append({
            "metadata_audit_id": f"source-group:{row['source_file']}:{row['original_source_column']}",
            "canonical_record_id": "", "source_phase": row["source_phase"],
            "source_file": row["source_file"], "source_row": "*",
            "metadata_field": row["original_source_column"], "original_canonical_value": "",
            "reconciled_metadata_value": "", "metadata_status": "unmapped_source_field",
            "evidence_source": "results/metrics/phase8/phase8_schema_mapping_audit.csv",
            "evidence_location": row["original_source_column"],
            "reconciliation_explanation": "Source field is preserved in source_attributes_json without a justified canonical mapping.",
            "conflict_details": "",
        })


def run_metadata_reconciliation_stage(
    canonical_path: Path, mapping_audit_path: Path, phase6_config: Path, phase7_config: Path,
    quality_output: Path, overlay_output: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Write only P8-08 metadata quality artifacts without mutating inputs."""
    canonical = _read_csv(canonical_path)
    audit, overlay = build_metadata_audits(canonical, phase6_config, phase7_config)
    add_unmapped_source_field_rows(audit, mapping_audit_path)
    audit.sort(key=lambda row: row["metadata_audit_id"])
    overlay.sort(key=lambda row: row["reconciled_metadata_id"])
    quality_output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(quality_output, AUDIT_FIELDS, audit)
    _write_csv(overlay_output, OVERLAY_FIELDS, overlay)
    return audit, overlay


def status_counts(audit: list[dict[str, str]]) -> Counter[str]:
    """Count audit statuses for reporting without treating fields as records."""
    return Counter(row["metadata_status"] for row in audit)
