"""Independent data-integrity checks for the Phase 8 completion gate."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
PHASE8 = METRICS / "phase8"
FIGURES = ROOT / "results" / "figures" / "phase8"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _source_rows(filename: str) -> list[dict[str, str]]:
    return _rows(METRICS / filename)


def test_source_inventory_and_canonical_provenance_are_complete() -> None:
    inventory = _rows(PHASE8 / "phase8_source_inventory.csv")
    expected = {path.name for path in METRICS.glob("phase[2-7]_*.csv")}
    assert {Path(row["relative_file_path"]).name for row in inventory} == expected
    assert all(row["exists"] == "True" and row["loaded"] == "True" for row in inventory)
    assert all(int(row["row_count"]) == len(_source_rows(Path(row["relative_file_path"]).name)) for row in inventory)

    canonical = _rows(PHASE8 / "phase8_canonical_long_form.csv")
    assert len(canonical) == 8741
    canonical_ids = [row["canonical_record_id"] for row in canonical]
    assert len(canonical_ids) == len(set(canonical_ids))
    assert all(row["metric_name"] != "safe_denominator" for row in canonical)
    assert Counter(row["metric_value_status"] for row in canonical) == {
        "observed": 8134, "missing": 607,
    }
    for row in canonical:
        if row["metric_value_status"] == "observed":
            float(row["metric_value"])


def test_classification_compatibility_and_metadata_cover_every_canonical_id() -> None:
    canonical_ids = {row["canonical_record_id"] for row in _rows(PHASE8 / "phase8_canonical_long_form.csv")}
    classification = _rows(PHASE8 / "phase8_record_type_audit.csv")
    compatibility = _rows(PHASE8 / "phase8_compatibility_audit.csv")
    metadata = _rows(PHASE8 / "phase8_metadata_quality_audit.csv")

    assert len(classification) == len(canonical_ids)
    assert {row["canonical_record_id"] for row in classification} == canonical_ids
    assert all(row["classification_status"] == "classified" for row in classification)
    assert len(compatibility) == len(canonical_ids)
    assert {row["canonical_record_id"] for row in compatibility} == canonical_ids
    assert Counter(row["compatibility_status"] for row in compatibility) == {
        "compatible": 5021, "incompatible": 288, "unknown": 3432,
    }
    metadata_ids = {row["canonical_record_id"] for row in metadata if row["canonical_record_id"]}
    assert metadata_ids == canonical_ids
    counts = Counter(row["metadata_status"] for row in metadata)
    assert counts == {
        "present_verified": 25283,
        "reconciled": 47389,
        "missing": 40938,
        "not_applicable": 8764,
        "unmapped_source_field": 85,
    }
    for row in metadata:
        if row["metadata_status"] == "reconciled":
            assert row["evidence_source"] and row["evidence_location"]


def test_table_values_trace_to_source_data_and_strict_rows_are_compatible() -> None:
    baseline = _rows(PHASE8 / "phase8_table1_baseline_inverse_methods.csv")
    source_baseline = _source_rows("phase2_baseline_comparison.csv")
    for row in baseline:
        source = source_baseline[int(row["source_row"]) - 1]
        assert row["method"] == source["solver"]
        assert float(row["relative_reconstruction_error"]) == float(source["relative_reconstruction_error"])
        assert row["comparison_scope"] == "descriptive_unknown_configuration"

    parameter = _rows(PHASE8 / "phase8_table3_parameter_selection.csv")
    gcv = next(row for row in parameter if row["method"] == "TSVD" and row["criterion"] == "gcv")
    assert float(gcv["selected_parameter"]) == 142.0
    assert float(gcv["relative_reconstruction_error"]) == 17317.138052621427
    assert "noise-dominated failure" in gcv["failure_note"]

    regularization = _rows(PHASE8 / "phase8_table2_regularization_methods.csv")
    lambda_point = next(row for row in regularization if row["method"] == "Tikhonov SVD" and row["parameter_value"] == "0.1")
    assert float(lambda_point["relative_reconstruction_error"]) == 0.18593123130628608
    assert lambda_point["comparison_scope"] == "descriptive_unknown_configuration"

    audit = _rows(PHASE8 / "phase8_compatibility_audit.csv")
    compatibility = {
        (row["source_file"], row["source_row"], row["canonical_record_id"].rsplit(":", 1)[-1]): row["compatibility_status"]
        for row in audit
    }
    iterative = _rows(PHASE8 / "phase8_table5_iterative_solver_comparison.csv")
    assert len(iterative) == 8
    for row in iterative:
        assert row["comparison_scope"] == "strict_compatible"
        for metric in ("relative_reconstruction_error", "residual_norm", "relative_residual", "execution_time_seconds"):
            assert compatibility[(row["source_file"], row["source_row"], metric)] == "compatible"


def test_robustness_aggregation_and_scalability_provenance_recalculate() -> None:
    raw = _source_rows("phase6_raw_results.csv")
    table = _rows(PHASE8 / "phase8_table4_robustness_sensitivity.csv")
    target = next(row for row in table if row["sensitivity_axis"] == "noise" and row["method"] == "normal_equations" and row["criterion"] == "fixed")
    grouped = [row for row in raw if row["sensitivity_axis"] == "noise" and row["method"] == "normal_equations" and row["criterion"] == "fixed"]
    expected = sum(float(row["relative_reconstruction_error"]) for row in grouped) / len(grouped)
    assert len({row["case_id"] for row in grouped}) == int(target["n_unique_cases"])
    assert float(target["mean_relative_reconstruction_error"]) == expected
    assert int(target["unstable_case_count"]) == sum(row["unstable_solution"] == "True" for row in grouped)

    source_iterative = _source_rows("phase7_iterative_results.csv")
    scalability = _rows(PHASE8 / "phase8_table7_scalability.csv")
    for row in scalability:
        matches = [
            source for source in source_iterative
            if source["case_id"] == row["case_id"] and source["method"] == row["method"]
            and source["iteration_count"] == row["iteration_count"]
            and source["run_type"] == "iteration_budget"
        ]
        assert len(matches) == 1
        assert float(row["runtime_seconds"]) == float(matches[0]["execution_time_seconds"])
        assert "not measured" in row["storage_interpretation"]


def test_figures_are_nonempty_pngs_and_comparative_report_has_required_limits() -> None:
    figures = sorted(FIGURES.glob("phase8_figure*.png"))
    assert len(figures) == 12
    for figure in figures:
        with Image.open(figure) as image:
            assert image.format == "PNG"
            assert image.width >= 800 and image.height >= 500
    report = (ROOT / "report" / "phase8_comparative_analysis.md").read_text(encoding="utf-8")
    for phrase in ("strict_compatible", "descriptive_unknown_configuration", "noise-dominated failure", "not peak-RAM"):
        assert phrase in report
