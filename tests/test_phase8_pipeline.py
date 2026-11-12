"""Architecture tests for safe, independently executable Phase 8 stages."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from analysis.phase8_pipeline import (
    CORE_PIPELINE,
    STAGES,
    available_stage_names,
    execute_core_pipeline,
    execute_stage,
    format_stage_listing,
    validate_stage_inputs,
)


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture\n", encoding="utf-8")


def _prepare_inputs(project_root: Path, stage: str) -> None:
    for relative_path in STAGES[stage].required_inputs:
        target = project_root / relative_path
        if target.suffix:
            _write(target)
        else:
            target.mkdir(parents=True, exist_ok=True)


def test_registered_stage_names_and_listing_are_deterministic() -> None:
    assert available_stage_names() == tuple(STAGES)
    listing = format_stage_listing()
    for stage in STAGES:
        assert stage in listing
    assert "full-core-pipeline" in listing


def test_cli_handler_mapping_preserves_each_isolated_stage() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "phase8_integrated_analysis.py"
    )
    spec = importlib.util.spec_from_file_location("phase8_experiment", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    handlers = module.phase8_stage_handlers()

    assert tuple(handlers) == available_stage_names()
    assert handlers["canonical-schema"].__name__ == "run_canonical_schema_stage_only"
    assert (
        handlers["record-type-classification"].__name__
        == "run_record_type_classification_stage_only"
    )
    assert handlers["compatibility-audit"].__name__ == "run_compatibility_audit_stage_only"
    assert (
        handlers["metadata-reconciliation"].__name__
        == "run_metadata_reconciliation_stage_only"
    )


def test_single_stage_dispatch_invokes_only_its_registered_handler(tmp_path: Path) -> None:
    _prepare_inputs(tmp_path, "canonical-schema")
    calls: list[str] = []
    handlers = {
        name: (lambda name=name: calls.append(name)) for name in STAGES
    }

    execute_stage("canonical-schema", tmp_path, handlers)

    assert calls == ["canonical-schema"]


def test_missing_inputs_name_stage_and_upstream_dependencies(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        validate_stage_inputs("compatibility-audit", tmp_path)

    message = str(error.value)
    assert "compatibility-audit" in message
    assert "canonical-schema" in message
    assert "record-type-classification" in message
    assert "phase8_canonical_long_form.csv" in message


def test_full_pipeline_uses_declared_order_and_only_declared_outputs(
    tmp_path: Path,
) -> None:
    _prepare_inputs(tmp_path, "source-inventory")
    owned_outputs = {output for spec in STAGES.values() for output in spec.outputs}
    for relative_path in STAGES["comparative-analysis"].required_inputs:
        if relative_path not in owned_outputs:
            _write(tmp_path / relative_path)
    _write(tmp_path / "results/metrics/phase6_experiment_config.json")
    _write(tmp_path / "results/metrics/phase7_experiment_config.json")
    calls: list[str] = []

    def handler_for(stage: str):
        def handler() -> None:
            calls.append(stage)
            for relative_path in STAGES[stage].outputs:
                _write(tmp_path / relative_path)

        return handler

    handlers = {stage: handler_for(stage) for stage in STAGES}
    execute_core_pipeline(tmp_path, handlers)

    assert tuple(calls) == CORE_PIPELINE
    declared_outputs = owned_outputs
    fixed_inputs = {
        "results/metrics/phase6_experiment_config.json",
        "results/metrics/phase7_experiment_config.json",
        *(path for path in STAGES["comparative-analysis"].required_inputs if path not in owned_outputs),
    }
    produced_files = {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert declared_outputs.issubset(produced_files)
    assert produced_files - declared_outputs == fixed_inputs


def test_stage_registry_has_no_phase_two_to_seven_output_ownership() -> None:
    for spec in STAGES.values():
        assert all(
            not output.startswith("results/metrics/phase") or "/phase8/" in output
            for output in spec.outputs
        )
