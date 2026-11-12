"""Focused tests for the Phase 8 Phase 2--7 source registry."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "experiments" / "phase8_integrated_analysis.py"
SPEC = importlib.util.spec_from_file_location("phase8_integrated_analysis", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PHASE8 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PHASE8
SPEC.loader.exec_module(PHASE8)


def test_registry_discovers_and_loads_all_available_phase_sources() -> None:
    registry = PHASE8.build_source_registry(ROOT / "results" / "metrics")
    data = PHASE8.load_registered_sources(ROOT / "results" / "metrics", registry)

    expected_files = {
        path.name
        for path in (ROOT / "results" / "metrics").glob("phase[2-7]_*.csv")
    }
    assert {Path(entry["relative_file_path"]).name for entry in registry} == expected_files
    assert all(entry["exists"] and entry["loaded"] for entry in registry)
    assert all(int(entry["row_count"]) == len(data[Path(entry["relative_file_path"]).name]) for entry in registry)


def test_registry_includes_previously_omitted_sources() -> None:
    registry = PHASE8.build_source_registry(ROOT / "results" / "metrics")
    entries = {Path(entry["relative_file_path"]).name: entry for entry in registry}

    for filename in (
        "phase3_structured_vs_full_svd.csv",
        "phase6_selected_parameters.csv",
        "phase7_case_metadata.csv",
        "phase7_small_system_validation.csv",
    ):
        assert entries[filename]["source_classification"] == "included"
        assert entries[filename]["record_type"]


def test_malformed_registered_source_is_reported(tmp_path: Path) -> None:
    malformed = tmp_path / "phase2_baseline_comparison.csv"
    malformed.write_text("\n", encoding="utf-8")
    registry = PHASE8.build_source_registry(tmp_path)

    with pytest.raises(RuntimeError, match="phase2_baseline_comparison.csv"):
        PHASE8.load_registered_sources(tmp_path, registry)
