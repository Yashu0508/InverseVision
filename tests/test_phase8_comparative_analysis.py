"""Focused tests for P8-04 compatibility-aware comparative analysis."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

from analysis.phase8_comparative_analysis import (
    FIGURE_FILENAMES,
    SOURCE_FILES,
    TABLE_SPECS,
    iterative_table,
    robustness_table,
    run_comparative_analysis_stage,
    scope_for_row,
)


ROOT = Path(__file__).resolve().parents[1]


def test_scope_never_promotes_unknown_or_incompatible_records() -> None:
    audit = {
        ("source.csv", "1", "error"): "compatible",
        ("source.csv", "2", "error"): "unknown",
        ("source.csv", "3", "error"): "incompatible",
    }
    assert scope_for_row(audit, "source.csv", 1, ("error",)) == "strict_compatible"
    assert scope_for_row(audit, "source.csv", 2, ("error",)) == "descriptive_unknown_configuration"
    assert scope_for_row(audit, "source.csv", 3, ("error",)) == "excluded_incompatible"


def test_robustness_groups_unique_cases_without_pseudoreplication() -> None:
    sources = {"phase6_raw_results.csv": [
        {"case_id": "case-a", "sensitivity_axis": "noise", "condition_value": "all", "method": "TSVD", "criterion": "gcv", "relative_reconstruction_error": "1", "residual_norm": "2", "unstable_solution": "False"},
        {"case_id": "case-a", "sensitivity_axis": "noise", "condition_value": "all", "method": "TSVD", "criterion": "gcv", "relative_reconstruction_error": "3", "residual_norm": "4", "unstable_solution": "False"},
    ]}
    audit = {
        ("phase6_raw_results.csv", "1", "relative_reconstruction_error"): "compatible",
        ("phase6_raw_results.csv", "1", "residual_norm"): "compatible",
        ("phase6_raw_results.csv", "2", "relative_reconstruction_error"): "compatible",
        ("phase6_raw_results.csv", "2", "residual_norm"): "compatible",
    }
    result = robustness_table(sources, audit)
    assert len(result) == 1
    assert result[0]["n_unique_cases"] == 1
    assert result[0]["mean_relative_reconstruction_error"] == 2.0


def test_iterative_table_excludes_non_budget_and_incompatible_rows() -> None:
    common = {"case_id": "case", "method": "LSQR", "iteration_count": "8", "requested_iteration_limit": "8", "relative_reconstruction_error": "0.2", "residual_norm": "0.1", "relative_residual": "0.01", "execution_time_seconds": "0.02", "relative_tolerance": "1e-8", "termination_status": "limit"}
    sources = {"phase7_iterative_results.csv": [
        {**common, "run_type": "iteration_trajectory"},
        {**common, "run_type": "iteration_budget"},
    ]}
    audit = {
        ("phase7_iterative_results.csv", "1", metric): "compatible"
        for metric in ("relative_reconstruction_error", "residual_norm", "relative_residual", "execution_time_seconds")
    }
    audit.update({
        ("phase7_iterative_results.csv", "2", metric): "incompatible"
        for metric in ("relative_reconstruction_error", "residual_norm", "relative_residual", "execution_time_seconds")
    })
    assert iterative_table(sources, audit) == []


def test_stage_outputs_are_deterministic_and_owned_by_temporary_root(tmp_path: Path) -> None:
    metrics = tmp_path / "results" / "metrics"
    phase8 = metrics / "phase8"
    figures = tmp_path / "results" / "figures" / "phase8"
    metrics.mkdir(parents=True)
    phase8.mkdir()
    for filename in SOURCE_FILES:
        shutil.copy2(ROOT / "results" / "metrics" / filename, metrics / filename)
    for filename in (
        "phase8_canonical_long_form.csv", "phase8_record_type_audit.csv",
        "phase8_compatibility_audit.csv", "phase8_reconciled_metadata.csv",
    ):
        shutil.copy2(ROOT / "results" / "metrics" / "phase8" / filename, phase8 / filename)

    run_comparative_analysis_stage(metrics, phase8, figures)
    first_hash = hashlib.sha256((phase8 / "phase8_table3_parameter_selection.csv").read_bytes()).hexdigest()
    run_comparative_analysis_stage(metrics, phase8, figures)
    second_hash = hashlib.sha256((phase8 / "phase8_table3_parameter_selection.csv").read_bytes()).hexdigest()

    assert first_hash == second_hash
    assert {path.name for path in figures.glob("*.png")} == set(FIGURE_FILENAMES)
    for filename, fields in TABLE_SPECS.items():
        with (phase8 / filename).open(encoding="utf-8", newline="") as stream:
            assert tuple(csv.DictReader(stream).fieldnames or ()) == fields
