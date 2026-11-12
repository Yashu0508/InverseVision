"""Focused validation for the isolated Phase 8 canonical schema stage."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from analysis.phase8_canonical_schema import run_canonical_schema_stage


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
INVENTORY = METRICS / "phase8" / "phase8_source_inventory.csv"


def _run_stage(output_directory: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return run_canonical_schema_stage(METRICS, INVENTORY, output_directory)


def test_all_inventoried_sources_have_processed_adapter_outcomes(tmp_path: Path) -> None:
    _, audit = _run_stage(tmp_path)
    processed = {
        row["source_file"]
        for row in audit
        if row["original_source_column"] == "__source__"
        and row["processing_status"] == "processed"
    }
    with INVENTORY.open(encoding="utf-8", newline="") as stream:
        inventory = list(csv.DictReader(stream))
    expected = {Path(row["relative_file_path"]).name for row in inventory if row["loaded"] == "True"}
    assert processed == expected


def test_canonical_records_have_unique_numeric_traceable_provenance(tmp_path: Path) -> None:
    observations, _ = _run_stage(tmp_path)
    assert len({row["canonical_record_id"] for row in observations}) == len(observations)
    assert all(row["source_phase"] and row["source_file"] and row["source_row"] for row in observations)
    for row in observations:
        assert json.loads(row["source_attributes_json"])
        if row["metric_value_status"] == "observed":
            assert math.isfinite(float(row["metric_value"]))
        else:
            assert row["metric_value_status"] == "missing"
            assert row["metric_value"] == ""


def test_source_rows_preserve_multiple_metrics_and_distinct_record_types(tmp_path: Path) -> None:
    observations, _ = _run_stage(tmp_path)
    phase2_first_row = [
        row for row in observations
        if row["source_file"] == "phase2_baseline_comparison.csv" and row["source_row"] == "1"
    ]
    assert len(phase2_first_row) > 2
    assert len({row["source_attributes_json"] for row in phase2_first_row}) == 1
    record_types = {row["record_type"] for row in observations}
    assert {"spectral_component", "method_result", "parameter_candidate", "iterative_solver_result"} <= record_types


def test_picard_boolean_safe_denominator_is_preserved_as_metadata(tmp_path: Path) -> None:
    observations, audit = _run_stage(tmp_path)
    picard_first_row = [
        row for row in observations
        if row["source_file"] == "phase3_picard_noise.csv" and row["source_row"] == "1"
    ]
    assert "safe_denominator" not in {row["metric_name"] for row in picard_first_row}
    assert json.loads(picard_first_row[0]["source_attributes_json"])["safe_denominator"] == "True"
    audit_row = next(
        row for row in audit
        if row["source_file"] == "phase3_picard_noise.csv"
        and row["original_source_column"] == "safe_denominator"
    )
    assert audit_row["canonical_destination"] == "source_attributes_json"
    assert audit_row["mapping_justified"] == "no"


def test_isolated_stage_is_deterministic_and_preserves_other_files(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    protected = tmp_path / "unrelated_phase8_artifact.txt"
    protected.write_text("preserve me", encoding="utf-8")
    _run_stage(first)
    _run_stage(second)
    assert protected.read_text(encoding="utf-8") == "preserve me"
    assert (first / "phase8_canonical_long_form.csv").read_bytes() == (
        second / "phase8_canonical_long_form.csv"
    ).read_bytes()
    assert (first / "phase8_schema_mapping_audit.csv").read_bytes() == (
        second / "phase8_schema_mapping_audit.csv"
    ).read_bytes()
