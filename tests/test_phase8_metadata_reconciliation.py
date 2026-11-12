"""Focused tests for Phase 8 evidence-backed metadata reconciliation."""

from __future__ import annotations

import csv
from pathlib import Path

from analysis.phase8_metadata_reconciliation import (
    _audit_row,
    add_unmapped_source_field_rows,
    build_metadata_audits,
    run_metadata_reconciliation_stage,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE8 = ROOT / "results" / "metrics" / "phase8"
CANONICAL = PHASE8 / "phase8_canonical_long_form.csv"
MAPPING_AUDIT = PHASE8 / "phase8_schema_mapping_audit.csv"
PHASE6_CONFIG = ROOT / "results" / "metrics" / "phase6_experiment_config.json"
PHASE7_CONFIG = ROOT / "results" / "metrics" / "phase7_experiment_config.json"


def _canonical() -> list[dict[str, str]]:
    with CANONICAL.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_explicit_phase2_and_phase45_metadata_are_reconciled() -> None:
    records = _canonical()
    audit, overlay = build_metadata_audits(records, PHASE6_CONFIG, PHASE7_CONFIG)
    phase2_shape = next(row for row in audit if row["canonical_record_id"].startswith("phase2:") and row["metadata_field"] == "image_dimensions")
    phase45_boundary = next(row for row in audit if row["source_file"] == "phase5_selection_curves.csv" and row["metadata_field"] == "boundary_condition")
    assert phase2_shape["metadata_status"] == "reconciled"
    assert phase2_shape["reconciled_metadata_value"] == "12x12"
    assert phase45_boundary["metadata_status"] == "reconciled"
    assert any(row["canonical_record_id"] == phase2_shape["canonical_record_id"] for row in overlay)


def test_missing_not_applicable_and_unmapped_metadata_are_distinct() -> None:
    audit, _ = build_metadata_audits(_canonical(), PHASE6_CONFIG, PHASE7_CONFIG)
    spectral_noise = next(row for row in audit if row["source_file"] == "phase3_svd_spectrum.csv" and row["metadata_field"] == "noise_seed")
    small_validation = next(row for row in audit if row["source_file"] == "phase7_small_system_validation.csv" and row["metadata_field"] == "image_dimensions")
    assert spectral_noise["metadata_status"] == "not_applicable"
    assert small_validation["metadata_status"] == "missing"


def test_conflicts_and_unmapped_fields_are_never_silently_reconciled(tmp_path: Path) -> None:
    source_record = next(row for row in _canonical() if row["source_file"] == "phase2_baseline_comparison.csv")
    conflicting = {**source_record, "image_shape": "16x16"}
    result = _audit_row(
        conflicting,
        "image_dimensions",
        {"image_dimensions": "12x12", "evidence": "explicit fixture", "location": "fixture"},
    )
    assert result["metadata_status"] == "conflicting"
    assert result["reconciled_metadata_value"] == ""
    assert "canonical=16x16" in result["conflict_details"]
    audit: list[dict[str, str]] = []
    add_unmapped_source_field_rows(audit, MAPPING_AUDIT)
    assert any(row["metadata_status"] == "unmapped_source_field" for row in audit)


def test_stage_is_deterministic_and_does_not_mutate_canonical(tmp_path: Path) -> None:
    before = CANONICAL.read_bytes()
    first_audit, first_overlay = run_metadata_reconciliation_stage(
        CANONICAL, MAPPING_AUDIT, PHASE6_CONFIG, PHASE7_CONFIG,
        tmp_path / "quality-one.csv", tmp_path / "overlay-one.csv",
    )
    second_audit, second_overlay = run_metadata_reconciliation_stage(
        CANONICAL, MAPPING_AUDIT, PHASE6_CONFIG, PHASE7_CONFIG,
        tmp_path / "quality-two.csv", tmp_path / "overlay-two.csv",
    )
    assert CANONICAL.read_bytes() == before
    assert first_audit == second_audit
    assert first_overlay == second_overlay
    assert (tmp_path / "quality-one.csv").read_bytes() == (tmp_path / "quality-two.csv").read_bytes()
    canonical_ids = {row["canonical_record_id"] for row in _canonical()}
    audited_ids = {row["canonical_record_id"] for row in first_audit if row["canonical_record_id"]}
    assert audited_ids == canonical_ids
