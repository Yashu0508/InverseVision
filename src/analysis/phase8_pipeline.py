"""Safe orchestration metadata for independently executable Phase 8 stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StageSpec:
    """Inputs, outputs, and direct dependencies owned by one Phase 8 stage."""

    name: str
    dependencies: tuple[str, ...]
    required_inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    description: str


STAGES = {
    "source-inventory": StageSpec(
        "source-inventory", (), ("results/metrics",),
        ("results/metrics/phase8/phase8_source_inventory.csv",),
        "Discover and validate Phase 2--7 metrics sources.",
    ),
    "raw-integration": StageSpec(
        "raw-integration", ("source-inventory",),
        ("results/metrics/phase8/phase8_source_inventory.csv",),
        ("results/metrics/phase8/phase8_unified_records.csv",),
        "Create only the raw provenance union; no analytical comparison.",
    ),
    "canonical-schema": StageSpec(
        "canonical-schema", ("source-inventory",),
        ("results/metrics/phase8/phase8_source_inventory.csv",),
        (
            "results/metrics/phase8/phase8_canonical_long_form.csv",
            "results/metrics/phase8/phase8_schema_mapping_audit.csv",
        ),
        "Create canonical long-form observations and mapping audit.",
    ),
    "record-type-classification": StageSpec(
        "record-type-classification", ("canonical-schema",),
        ("results/metrics/phase8/phase8_canonical_long_form.csv",),
        ("results/metrics/phase8/phase8_record_type_audit.csv",),
        "Classify canonical observation roles.",
    ),
    "compatibility-audit": StageSpec(
        "compatibility-audit", ("canonical-schema", "record-type-classification"),
        (
            "results/metrics/phase8/phase8_canonical_long_form.csv",
            "results/metrics/phase8/phase8_record_type_audit.csv",
        ),
        ("results/metrics/phase8/phase8_compatibility_audit.csv",),
        "Audit purpose-specific configuration compatibility.",
    ),
    "metadata-reconciliation": StageSpec(
        "metadata-reconciliation", ("canonical-schema",),
        (
            "results/metrics/phase8/phase8_canonical_long_form.csv",
            "results/metrics/phase8/phase8_schema_mapping_audit.csv",
            "results/metrics/phase6_experiment_config.json",
            "results/metrics/phase7_experiment_config.json",
        ),
        (
            "results/metrics/phase8/phase8_metadata_quality_audit.csv",
            "results/metrics/phase8/phase8_reconciled_metadata.csv",
        ),
        "Audit and overlay evidence-backed metadata reconciliation.",
    ),
    "comparative-analysis": StageSpec(
        "comparative-analysis",
        (
            "canonical-schema", "record-type-classification", "compatibility-audit",
            "metadata-reconciliation",
        ),
        (
            "results/metrics/phase8/phase8_canonical_long_form.csv",
            "results/metrics/phase8/phase8_record_type_audit.csv",
            "results/metrics/phase8/phase8_compatibility_audit.csv",
            "results/metrics/phase8/phase8_reconciled_metadata.csv",
            "results/metrics/phase2_baseline_comparison.csv",
            "results/metrics/phase3_blur_sensitivity.csv",
            "results/metrics/phase3_structured_vs_full_svd.csv",
            "results/metrics/phase3_svd_spectrum.csv",
            "results/metrics/phase3_svd_summary.csv",
            "results/metrics/phase4_regularization_comparison.csv",
            "results/metrics/phase5_selection_curves.csv",
            "results/metrics/phase5_selection_summary.csv",
            "results/metrics/phase6_raw_results.csv",
            "results/metrics/phase7_case_metadata.csv",
            "results/metrics/phase7_iterative_results.csv",
        ),
        (
            "results/metrics/phase8/phase8_table1_baseline_inverse_methods.csv",
            "results/metrics/phase8/phase8_table2_regularization_methods.csv",
            "results/metrics/phase8/phase8_table3_parameter_selection.csv",
            "results/metrics/phase8/phase8_table4_robustness_sensitivity.csv",
            "results/metrics/phase8/phase8_table5_iterative_solver_comparison.csv",
            "results/metrics/phase8/phase8_table6_spectral_conditioning.csv",
            "results/metrics/phase8/phase8_table7_scalability.csv",
            "results/metrics/phase8/phase8_comparative_analysis_audit.csv",
            "results/figures/phase8/phase8_figure01_baseline_inverse_methods.png",
            "results/figures/phase8/phase8_figure02_regularization_accuracy_residual.png",
            "results/figures/phase8/phase8_figure03_lcurve.png",
            "results/figures/phase8/phase8_figure04_gcv_discrepancy_curves.png",
            "results/figures/phase8/phase8_figure05_noise_seed_robustness.png",
            "results/figures/phase8/phase8_figure06_blur_strength_sensitivity.png",
            "results/figures/phase8/phase8_figure07_resolution_sensitivity.png",
            "results/figures/phase8/phase8_figure08_iterative_convergence.png",
            "results/figures/phase8/phase8_figure09_error_vs_iteration.png",
            "results/figures/phase8/phase8_figure10_spectral_decay.png",
            "results/figures/phase8/phase8_figure11_scalability_runtime.png",
            "results/figures/phase8/phase8_figure12_storage_scaling.png",
        ),
        "Create compatibility-aware comparative tables and figures from saved evidence.",
    ),
}

CORE_PIPELINE = (
    "source-inventory", "raw-integration", "canonical-schema",
    "record-type-classification", "compatibility-audit", "metadata-reconciliation",
    "comparative-analysis",
)


def available_stage_names() -> tuple[str, ...]:
    """Return the public isolated stages in deterministic CLI order."""
    return tuple(STAGES)


def format_stage_listing() -> str:
    """Format concise human-readable CLI help from the authoritative registry."""
    lines = ["Available isolated Phase 8 stages:"]
    for name, spec in STAGES.items():
        upstream = ", ".join(spec.dependencies) or "none"
        lines.append(f"  {name}: {spec.description} Upstream: {upstream}.")
    lines.append("  full-core-pipeline: explicitly run all stages in dependency order.")
    return "\n".join(lines)


def validate_stage_inputs(stage: str, project_root: Path) -> None:
    """Fail clearly before a stage reads a missing required artifact."""
    spec = STAGES[stage]
    missing = [path for path in spec.required_inputs if not (project_root / path).exists()]
    if missing:
        raise FileNotFoundError(
            f"Stage '{stage}' requires missing artifact(s): {', '.join(missing)}. "
            f"Run its upstream stage(s): {', '.join(spec.dependencies) or 'none'}."
        )


def execute_stage(
    stage: str, project_root: Path, handlers: Mapping[str, Callable[[], None]]
) -> None:
    """Validate one named stage and invoke only its owning handler."""
    if stage not in STAGES:
        raise ValueError(f"Unknown Phase 8 stage: {stage}")
    if stage not in handlers:
        raise ValueError(f"No implementation handler is registered for stage '{stage}'")
    validate_stage_inputs(stage, project_root)
    handlers[stage]()


def execute_core_pipeline(project_root: Path, handlers: Mapping[str, Callable[[], None]]) -> None:
    """Explicitly execute the documented core dependency order."""
    for stage in CORE_PIPELINE:
        execute_stage(stage, project_root, handlers)
