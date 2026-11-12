"""Focused tests for Phase 8 purpose-specific compatibility auditing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis.phase8_compatibility_audit import (
    build_compatibility_audit,
    evaluate_configuration,
    run_compatibility_audit_stage,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE8 = ROOT / "results" / "metrics" / "phase8"
CANONICAL = PHASE8 / "phase8_canonical_long_form.csv"
CLASSIFICATION = PHASE8 / "phase8_record_type_audit.csv"


def _record(identifier: str, source_file: str, attributes: dict[str, str]) -> dict[str, str]:
    return {
        "canonical_record_id": identifier,
        "source_phase": "phase7" if source_file.startswith("phase7") else "phase6",
        "source_file": source_file,
        "source_row": "1",
        "record_type": "iterative_solver_result",
        "analytical_category": "iterative_methods",
        "experiment_id": attributes.get("case_id", ""),
        "case_id": attributes.get("case_id", ""),
        "method": attributes.get("method", "CGLS"),
        "solver": "",
        "parameter_selection_method": attributes.get("criterion", ""),
        "image_shape": attributes.get("image_shape", ""),
        "image_height": "",
        "image_width": "",
        "blur_sigma": attributes.get("sigma", ""),
        "kernel_size": attributes.get("kernel_size", ""),
        "boundary": attributes.get("boundary", ""),
        "noise_standard_deviation": attributes.get("noise_standard_deviation", ""),
        "random_seed": attributes.get("random_seed", ""),
        "metric_name": "residual_norm",
        "metric_value": "1.0",
        "metric_value_status": "observed",
        "metric_unit": "data_l2_norm",
        "observation_index": "",
        "iteration": attributes.get("iteration_count", ""),
        "parameter_name": "",
        "parameter_value": "",
        "source_attributes_json": json.dumps(attributes, sort_keys=True),
    }


def _classification(record: dict[str, str], assigned: str) -> dict[str, str]:
    return {
        "canonical_record_id": record["canonical_record_id"],
        "assigned_record_type": assigned,
        "analytical_category": "iterative_solvers",
    }


def test_exact_mismatch_and_missing_dimensions_have_expected_status() -> None:
    reference = {"image_shape": "64x64", "random_seed": "42"}
    assert evaluate_configuration(reference, dict(reference), tuple(reference))[0] == "compatible"
    assert evaluate_configuration(reference, {"image_shape": "32x32", "random_seed": "42"}, tuple(reference))[0] == "incompatible"
    assert evaluate_configuration(reference, {"image_shape": "64x64", "random_seed": ""}, tuple(reference))[0] == "unknown"


def test_noise_and_iteration_purposes_are_specific() -> None:
    base = {
        "case_id": "case", "image_shape": "64x64", "sigma": "1.5", "kernel_size": "9",
        "boundary": "zero", "noise_standard_deviation": "0.01", "random_seed": "42",
    }
    noise_a = _record("noise-a", "phase6_raw_results.csv", {**base, "sensitivity_axis": "noise", "method": "TSVD", "criterion": "gcv"})
    noise_b = _record("noise-b", "phase6_raw_results.csv", {**base, "sensitivity_axis": "noise", "random_seed": "7", "method": "TSVD", "criterion": "gcv"})
    budget_a = _record("budget-a", "phase7_iterative_results.csv", {**base, "run_type": "iteration_budget", "requested_iteration_limit": "16"})
    budget_b = _record("budget-b", "phase7_iterative_results.csv", {**base, "run_type": "iteration_budget", "requested_iteration_limit": "32"})
    audit = build_compatibility_audit(
        [noise_a, noise_b, budget_a, budget_b],
        [_classification(noise_a, "noise_seed_individual_case_result"), _classification(noise_b, "noise_seed_individual_case_result"), _classification(budget_a, "iteration_budget_solver_observation"), _classification(budget_b, "iteration_budget_solver_observation")],
    )
    outcomes = {row["canonical_record_id"]: row["compatibility_status"] for row in audit}
    assert outcomes["noise-a"] == outcomes["noise-b"] == "compatible"
    assert outcomes["budget-a"] == "compatible"
    assert outcomes["budget-b"] == "incompatible"


def test_full_audit_is_complete_deterministic_and_preserves_canonical_input(tmp_path: Path) -> None:
    before = CANONICAL.read_bytes()
    first = run_compatibility_audit_stage(CANONICAL, CLASSIFICATION, tmp_path / "first.csv")
    second = run_compatibility_audit_stage(CANONICAL, CLASSIFICATION, tmp_path / "second.csv")
    assert CANONICAL.read_bytes() == before
    assert len(first) == len({row["canonical_record_id"] for row in first})
    assert first == second
    assert (tmp_path / "first.csv").read_bytes() == (tmp_path / "second.csv").read_bytes()
    with CANONICAL.open(encoding="utf-8", newline="") as stream:
        assert len(first) == len(list(csv.DictReader(stream)))
    assert all(row["compatibility_status"] in {"compatible", "incompatible", "unknown"} for row in first)
