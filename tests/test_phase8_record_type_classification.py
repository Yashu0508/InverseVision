"""Focused tests for Phase 8 analytical record-type separation."""

from __future__ import annotations

import csv
from pathlib import Path

from analysis.phase8_record_type_classification import (
    classify_records,
    run_record_type_classification_stage,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results" / "metrics" / "phase8" / "phase8_canonical_long_form.csv"


def _records() -> list[dict[str, str]]:
    with CANONICAL.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _classified() -> list[dict[str, str]]:
    return classify_records(_records())


def test_spectral_and_reconstruction_observations_remain_distinct() -> None:
    audit = _classified()
    assert {
        row["assigned_record_type"]
        for row in audit
        if row["source_file"] == "phase3_svd_spectrum.csv"
    } == {"singular_value_observation"}
    assert {
        row["assigned_record_type"]
        for row in audit
        if row["source_file"] == "phase2_baseline_comparison.csv"
    } == {"unregularized_baseline_reconstruction_result"}


def test_candidates_selections_and_phase6_aggregates_are_separate() -> None:
    audit = _classified()
    by_file = {
        filename: {
            row["assigned_record_type"]
            for row in audit
            if row["source_file"] == filename
        }
        for filename in {
            "phase5_selection_curves.csv",
            "phase5_selection_summary.csv",
            "phase6_aggregated_summary.csv",
            "phase6_raw_results.csv",
        }
    }
    assert by_file["phase5_selection_curves.csv"] == {"parameter_candidate_evaluation"}
    assert by_file["phase5_selection_summary.csv"] == {"selected_parameter_result"}
    assert by_file["phase6_aggregated_summary.csv"] == {"aggregated_sensitivity_summary"}
    assert "noise_seed_individual_case_result" in by_file["phase6_raw_results.csv"]


def test_iteration_protocols_and_metadata_remain_distinct() -> None:
    audit = _classified()
    run_types = {
        row["assigned_record_type"]
        for row in audit
        if row["source_file"] == "phase7_iterative_results.csv"
    }
    assert run_types == {
        "iteration_history_observation",
        "tolerance_stopping_observation",
        "iteration_budget_solver_observation",
    }
    assert {
        row["assigned_record_type"]
        for row in audit
        if row["source_file"] == "phase7_case_metadata.csv"
    } == {"experimental_metadata_diagnostic"}


def test_every_record_is_audited_deterministically(tmp_path: Path) -> None:
    first = run_record_type_classification_stage(CANONICAL, tmp_path / "first.csv")
    second = run_record_type_classification_stage(CANONICAL, tmp_path / "second.csv")
    assert len(first) == len(_records())
    assert {row["canonical_record_id"] for row in first} == {
        row["canonical_record_id"] for row in _records()
    }
    assert first == second
    assert (tmp_path / "first.csv").read_bytes() == (tmp_path / "second.csv").read_bytes()
