"""Canonical long-form representation for saved Phase 2--7 evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


CANONICAL_FIELDS = (
    "canonical_record_id",
    "source_phase",
    "source_file",
    "source_row",
    "record_type",
    "analytical_category",
    "experiment_id",
    "case_id",
    "method",
    "solver",
    "parameter_selection_method",
    "image_shape",
    "image_height",
    "image_width",
    "blur_sigma",
    "kernel_size",
    "boundary",
    "noise_standard_deviation",
    "random_seed",
    "metric_name",
    "metric_value",
    "metric_value_status",
    "metric_unit",
    "observation_index",
    "iteration",
    "parameter_name",
    "parameter_value",
    "source_attributes_json",
)

SOURCE_CLASSIFICATIONS = {
    "phase2_baseline_comparison.csv": ("method_result", "unregularized_reconstruction"),
    "phase3_blur_sensitivity.csv": ("blur_condition_summary", "spectral_conditioning"),
    "phase3_picard_noise.csv": ("spectral_component", "picard_noise_analysis"),
    "phase3_structured_vs_full_svd.csv": ("validation_summary", "structured_svd_validation"),
    "phase3_svd_spectrum.csv": ("spectral_component", "singular_value_analysis"),
    "phase3_svd_summary.csv": ("spectral_summary", "singular_value_analysis"),
    "phase4_regularization_comparison.csv": ("method_result", "explicit_regularization"),
    "phase4_tikhonov_results.csv": ("regularization_result", "explicit_regularization"),
    "phase4_tsvd_results.csv": ("regularization_result", "explicit_regularization"),
    "phase5_selection_curves.csv": ("parameter_candidate", "parameter_selection"),
    "phase5_selection_summary.csv": ("selection_result", "parameter_selection"),
    "phase6_aggregated_summary.csv": ("aggregate_statistic", "robustness_sensitivity"),
    "phase6_raw_results.csv": ("case_method_result", "robustness_sensitivity"),
    "phase6_selected_parameters.csv": ("selection_result", "robustness_sensitivity"),
    "phase7_case_metadata.csv": ("case_metadata", "iterative_method_setup"),
    "phase7_iterative_results.csv": ("iterative_solver_result", "iterative_methods"),
    "phase7_small_system_validation.csv": ("validation_result", "iterative_method_validation"),
}

CONFIGURATION_FIELDS = {
    "image_shape": "image_shape",
    "image_height": "image_height",
    "image_width": "image_width",
    "sigma": "blur_sigma",
    "kernel_size": "kernel_size",
    "boundary": "boundary",
    "noise_standard_deviation": "noise_standard_deviation",
    "random_seed": "random_seed",
}

METRIC_UNITS = {
    "relative_reconstruction_error": "dimensionless",
    "mean_squared_error": "normalized_intensity_squared",
    "mse": "normalized_intensity_squared",
    "peak_signal_to_noise_ratio": "dB",
    "psnr_db": "dB",
    "structural_similarity_index": "dimensionless",
    "ssim": "dimensionless",
    "residual_norm": "data_l2_norm",
    "relative_residual": "dimensionless",
    "normal_equation_residual_norm": "normal_equation_l2_norm",
    "reconstruction_norm": "solution_l2_norm",
    "reference_norm": "reference_l2_norm",
    "execution_time_seconds": "seconds",
    "augmented_execution_time_seconds": "seconds",
    "problem_setup_seconds": "seconds",
    "case_execution_seconds": "seconds",
    "full_min_seconds": "seconds",
    "full_median_seconds": "seconds",
    "full_max_seconds": "seconds",
    "structured_min_seconds": "seconds",
    "structured_median_seconds": "seconds",
    "structured_max_seconds": "seconds",
    "singular_value": "operator_gain",
    "normalized_singular_value": "dimensionless",
    "cumulative_energy": "dimensionless",
    "largest_singular_value": "operator_gain",
    "smallest_singular_value": "operator_gain",
    "condition_number": "dimensionless",
    "median_normalized_singular_value": "dimensionless",
    "expansion_coefficient": "unspecified",
    "coefficient_magnitude": "unspecified",
    "coefficient_to_singular_value_ratio": "unspecified",
    "transformed_noise_coefficient": "unspecified",
    "noise_contribution_magnitude": "unspecified",
    "maximum_absolute_spectrum_difference": "operator_gain",
    "maximum_relative_spectrum_difference": "dimensionless",
    "augmented_max_absolute_difference": "unspecified",
    "effective_degrees_of_freedom": "dimensionless",
    "gcv_numerator": "unspecified",
    "gcv_denominator": "dimensionless",
    "lcurve_curvature": "unspecified",
    "gcv": "unspecified",
    "discrepancy_absolute_error": "data_l2_norm",
    "criterion_score": "criterion_specific",
    "forward_consistency_error": "data_l2_norm",
    "operator_condition_number_from_1d_factors": "dimensionless",
    "condition_estimate": "dimensionless",
    "relative_solution_difference_from_QR": "dimensionless",
    "relative_solution_difference_from_Tikhonov": "dimensionless",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(not field for field in reader.fieldnames):
            raise ValueError(f"Malformed CSV header in {path}")
        return list(reader.fieldnames), list(reader)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _numeric_or_missing(value: str) -> tuple[str, str]:
    if value.strip() == "":
        return "", "missing"
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"Expected a numeric metric value, received {value!r}") from error
    if not math.isfinite(number):
        return "", "missing"
    return repr(number), "observed"


def _parameter_fields(row: dict[str, str]) -> tuple[str, str]:
    if row.get("parameter_name"):
        return row["parameter_name"], row.get("parameter_value", "")
    if row.get("parameter_type"):
        return row["parameter_type"], row.get("selected_parameter", "")
    if row.get("selected_parameter"):
        return "selected_parameter", row["selected_parameter"]
    if row.get("parameter"):
        return "candidate_parameter", row["parameter"]
    if row.get("damping_parameter_lambda"):
        return "damping_parameter_lambda", row["damping_parameter_lambda"]
    return "", ""


def _base_observation(
    source_phase: str,
    source_file: str,
    source_row: int,
    row: dict[str, str],
) -> dict[str, str]:
    record_type, category = SOURCE_CLASSIFICATIONS[source_file]
    parameter_name, parameter_value = _parameter_fields(row)
    case_id = row.get("case_id", "")
    return {
        "source_phase": source_phase,
        "source_file": source_file,
        "source_row": str(source_row),
        "record_type": record_type,
        "analytical_category": category,
        "experiment_id": case_id,
        "case_id": case_id,
        "method": row.get("method", ""),
        "solver": row.get("solver", ""),
        "parameter_selection_method": row.get("criterion", ""),
        "image_shape": row.get("image_shape", ""),
        "image_height": row.get("image_height", ""),
        "image_width": row.get("image_width", ""),
        "blur_sigma": row.get("sigma", ""),
        "kernel_size": row.get("kernel_size", ""),
        "boundary": row.get("boundary", ""),
        "noise_standard_deviation": row.get("noise_standard_deviation", ""),
        "random_seed": row.get("random_seed", ""),
        "observation_index": row.get("candidate_index", row.get("index", "")),
        "iteration": row.get("iteration_count", row.get("iterations", "")),
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "source_attributes_json": json.dumps(row, sort_keys=True),
    }


def _observation(
    base: dict[str, str], metric_name: str, raw_value: str
) -> dict[str, str]:
    value, status = _numeric_or_missing(raw_value)
    record = dict(base)
    record["canonical_record_id"] = (
        f"{base['source_phase']}:{base['source_file']}:"
        f"{base['source_row']}:{metric_name}"
    )
    record["metric_name"] = metric_name
    record["metric_value"] = value
    record["metric_value_status"] = status
    record["metric_unit"] = METRIC_UNITS[metric_name]
    return record


def _aggregate_observations(
    base: dict[str, str], row: dict[str, str]
) -> list[dict[str, str]]:
    observations = []
    source_metric = row["metric"]
    for statistic in ("mean", "standard_deviation", "median", "minimum", "maximum"):
        record = dict(base)
        value, status = _numeric_or_missing(row[statistic])
        metric_name = f"{source_metric}_{statistic}"
        record["canonical_record_id"] = (
            f"{base['source_phase']}:{base['source_file']}:"
            f"{base['source_row']}:{metric_name}"
        )
        record["metric_name"] = metric_name
        record["metric_value"] = value
        record["metric_value_status"] = status
        record["metric_unit"] = "source_metric_unit_unspecified"
        observations.append(record)
    return observations


def canonicalize_source(
    source_phase: str,
    source_file: str,
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Adapt one source file into distinct, provenance-preserving observations."""
    if source_file not in SOURCE_CLASSIFICATIONS:
        raise ValueError(f"No canonical adapter classification for {source_file}")
    observations: list[dict[str, str]] = []
    for source_row, row in enumerate(rows, start=1):
        base = _base_observation(source_phase, source_file, source_row, row)
        if source_file == "phase6_aggregated_summary.csv":
            observations.extend(_aggregate_observations(base, row))
            continue
        for field in METRIC_UNITS:
            if field in row:
                observations.append(_observation(base, field, row[field]))
    return observations


def _mapping_for_column(source_file: str, column: str) -> tuple[str, str, str, str]:
    if column in METRIC_UNITS:
        return "metric_name;metric_value;metric_unit", "metric observation", "yes", ""
    if source_file == "phase6_aggregated_summary.csv" and column == "metric":
        return "metric_name", "aggregate metric label", "yes", "Used with each aggregate statistic."
    if source_file == "phase6_aggregated_summary.csv" and column in {
        "mean", "standard_deviation", "median", "minimum", "maximum"
    }:
        return "metric_value", "aggregate statistic", "yes", "Metric name is derived from metric and statistic."
    if column in CONFIGURATION_FIELDS:
        return CONFIGURATION_FIELDS[column], "explicit configuration", "yes", ""
    if column in {"method", "solver", "criterion", "case_id", "parameter_name", "parameter_value"}:
        destination = {
            "criterion": "parameter_selection_method",
            "case_id": "case_id;experiment_id",
        }.get(column, column)
        return destination, "identity or method", "yes", ""
    if column in {"index", "candidate_index"}:
        return "observation_index", "observation identity", "yes", ""
    if column in {"iteration_count", "iterations"}:
        return "iteration", "iteration identity", "yes", ""
    if column in {"selected_parameter", "parameter", "parameter_type", "damping_parameter_lambda"}:
        return "parameter_name;parameter_value", "parameter", "yes", ""
    return "source_attributes_json", "source-specific preservation", "no", "No justified canonical mapping."


def schema_mapping_audit(
    registry: list[dict[str, str]], metrics_directory: Path
) -> list[dict[str, str]]:
    """Document every source-field mapping decision without discarding fields."""
    audit: list[dict[str, str]] = []
    for entry in registry:
        source_file = Path(entry["relative_file_path"]).name
        if entry["loaded"] != "True":
            audit.append(
                {
                    "source_file": source_file,
                    "source_phase": f"phase{entry['phase_number']}",
                    "original_source_column": "",
                    "canonical_destination": "",
                    "mapping_type": "unavailable source",
                    "mapping_justified": "no",
                    "preserved_as_source_specific": "no",
                    "reason": entry["reason"],
                    "processing_status": "unavailable",
                }
            )
            continue
        fields, _ = _read_csv(metrics_directory / source_file)
        audit.append(
            {
                "source_file": source_file,
                "source_phase": f"phase{entry['phase_number']}",
                "original_source_column": "__source__",
                "canonical_destination": "record_type;analytical_category",
                "mapping_type": "source adapter classification",
                "mapping_justified": "yes",
                "preserved_as_source_specific": "yes",
                "reason": "All original fields are retained in source_attributes_json.",
                "processing_status": "processed",
            }
        )
        for field in fields:
            destination, mapping_type, justified, reason = _mapping_for_column(source_file, field)
            audit.append(
                {
                    "source_file": source_file,
                    "source_phase": f"phase{entry['phase_number']}",
                    "original_source_column": field,
                    "canonical_destination": destination,
                    "mapping_type": mapping_type,
                    "mapping_justified": justified,
                    "preserved_as_source_specific": "yes",
                    "reason": reason,
                    "processing_status": "processed",
                }
            )
    return audit


def run_canonical_schema_stage(
    metrics_directory: Path,
    inventory_path: Path,
    output_directory: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Write only canonical-schema artifacts from the existing source inventory."""
    _, registry = _read_csv(inventory_path)
    observations: list[dict[str, str]] = []
    for entry in registry:
        source_file = Path(entry["relative_file_path"]).name
        if entry["loaded"] != "True":
            continue
        _, rows = _read_csv(metrics_directory / source_file)
        observations.extend(canonicalize_source(f"phase{entry['phase_number']}", source_file, rows))
    identifiers = [record["canonical_record_id"] for record in observations]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Canonical record identifiers must be unique")
    audit = schema_mapping_audit(registry, metrics_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "phase8_canonical_long_form.csv", CANONICAL_FIELDS, observations)
    _write_csv(
        output_directory / "phase8_schema_mapping_audit.csv",
        (
            "source_file",
            "source_phase",
            "original_source_column",
            "canonical_destination",
            "mapping_type",
            "mapping_justified",
            "preserved_as_source_specific",
            "reason",
            "processing_status",
        ),
        audit,
    )
    return observations, audit
