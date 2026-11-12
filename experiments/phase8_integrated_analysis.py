"""Integrate saved Phase 2–7 metrics without rerunning experiments."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
METRICS = ROOT / "results" / "metrics"
OUT_CSV = METRICS / "phase8"
OUT_PLOTS = ROOT / "results" / "plots" / "phase8"


def run_source_inventory_stage_only() -> None:
    """Discover and load Phase 2--7 CSV evidence, writing only its inventory."""
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    registry = build_source_registry(METRICS)
    load_registered_sources(METRICS, registry)
    write_csv(OUT_CSV / "phase8_source_inventory.csv", registry)


def run_raw_integration_stage_only() -> None:
    """Write only the provenance-preserving raw union from a validated inventory."""
    inventory_path = OUT_CSV / "phase8_source_inventory.csv"
    inventory = read_csv(inventory_path)
    required = {
        "phase_number", "relative_file_path", "exists", "loaded",
        "source_classification", "row_count", "columns",
    }
    if not inventory or not required.issubset(inventory[0]):
        raise ValueError(
            "Source inventory is malformed; run --stage source-inventory first."
        )

    unified: list[dict[str, object]] = []
    for entry in inventory:
        source = Path(entry["relative_file_path"])
        if entry["source_classification"] != "included":
            continue
        if entry["exists"] != "True" or entry["loaded"] != "True":
            raise ValueError(
                f"Source inventory has an unvalidated included source: {source.name}. "
                "Run --stage source-inventory first."
            )
        rows = read_csv(ROOT / source)
        expected_count = int(entry["row_count"])
        actual_columns = "|".join(rows[0].keys()) if rows else ""
        if len(rows) != expected_count or actual_columns != entry["columns"]:
            raise ValueError(
                f"Source inventory provenance does not match {source.name}; "
                "rerun --stage source-inventory before raw integration."
            )
        for source_row, row in enumerate(rows, start=1):
            unified.append(
                {
                    "source_phase": f"phase{entry['phase_number']}",
                    "source_file": source.name,
                    "source_row": source_row,
                    **row,
                }
            )
    if not unified:
        raise ValueError("Validated source inventory contains no included observations.")
    write_csv(OUT_CSV / "phase8_unified_records.csv", unified)


def run_canonical_schema_stage_only() -> None:
    """Generate only the P8-01 canonical schema artifacts."""
    from analysis.phase8_canonical_schema import run_canonical_schema_stage

    run_canonical_schema_stage(
        METRICS,
        OUT_CSV / "phase8_source_inventory.csv",
        OUT_CSV,
    )


def run_record_type_classification_stage_only() -> None:
    """Generate only the P8-07 record-type audit from canonical evidence."""
    from analysis.phase8_record_type_classification import (
        run_record_type_classification_stage,
    )
    run_record_type_classification_stage(
        OUT_CSV / "phase8_canonical_long_form.csv",
        OUT_CSV / "phase8_record_type_audit.csv",
    )


def run_compatibility_audit_stage_only() -> None:
    """Generate only the P8-02 configuration compatibility audit."""
    from analysis.phase8_compatibility_audit import run_compatibility_audit_stage

    run_compatibility_audit_stage(
        OUT_CSV / "phase8_canonical_long_form.csv",
        OUT_CSV / "phase8_record_type_audit.csv",
        OUT_CSV / "phase8_compatibility_audit.csv",
    )


def run_metadata_reconciliation_stage_only() -> None:
    """Generate only the P8-08 metadata quality audit and overlay."""
    from analysis.phase8_metadata_reconciliation import run_metadata_reconciliation_stage

    run_metadata_reconciliation_stage(
        OUT_CSV / "phase8_canonical_long_form.csv",
        OUT_CSV / "phase8_schema_mapping_audit.csv",
        METRICS / "phase6_experiment_config.json",
        METRICS / "phase7_experiment_config.json",
        OUT_CSV / "phase8_metadata_quality_audit.csv",
        OUT_CSV / "phase8_reconciled_metadata.csv",
    )


def run_comparative_analysis_stage_only() -> None:
    """Generate only P8-04 tables and figures from saved Phase 8 evidence."""
    from analysis.phase8_comparative_analysis import run_comparative_analysis_stage

    run_comparative_analysis_stage(METRICS, OUT_CSV, ROOT / "results" / "figures" / "phase8")


def phase8_stage_handlers() -> dict[str, Callable[[], None]]:
    """Return the public stage-to-handler mapping without executing a stage."""
    return {
        "source-inventory": run_source_inventory_stage_only,
        "raw-integration": run_raw_integration_stage_only,
        "canonical-schema": run_canonical_schema_stage_only,
        "record-type-classification": run_record_type_classification_stage_only,
        "compatibility-audit": run_compatibility_audit_stage_only,
        "metadata-reconciliation": run_metadata_reconciliation_stage_only,
        "comparative-analysis": run_comparative_analysis_stage_only,
    }

@dataclass(frozen=True)
class SourceDetails:
    """Analytical metadata for a saved Phase 2--7 metrics CSV."""

    purpose: str
    record_type: str


SOURCE_DETAILS = {
    "phase2_baseline_comparison.csv": SourceDetails(
        "Unregularized baseline reconstruction metrics.", "method result"
    ),
    "phase3_svd_summary.csv": SourceDetails(
        "SVD rank and condition-number summary.", "spectral summary"
    ),
    "phase3_svd_spectrum.csv": SourceDetails(
        "Ordered singular-value spectrum.", "singular-value component"
    ),
    "phase3_structured_vs_full_svd.csv": SourceDetails(
        "Structured versus full SVD agreement and timing.", "validation summary"
    ),
    "phase3_picard_noise.csv": SourceDetails(
        "Picard coefficients and transformed-noise diagnostics.", "spectral component"
    ),
    "phase3_blur_sensitivity.csv": SourceDetails(
        "Blur-strength spectral sensitivity diagnostics.", "blur condition summary"
    ),
    "phase4_tsvd_results.csv": SourceDetails(
        "TSVD parameter sweep results.", "regularization result"
    ),
    "phase4_tikhonov_results.csv": SourceDetails(
        "Tikhonov parameter sweep results.", "regularization result"
    ),
    "phase4_regularization_comparison.csv": SourceDetails(
        "Baseline and explicit-regularization comparison.", "method result"
    ),
    "phase5_selection_curves.csv": SourceDetails(
        "L-curve, GCV, and discrepancy candidate diagnostics.", "parameter candidate"
    ),
    "phase5_selection_summary.csv": SourceDetails(
        "Selected parameter and post-selection evaluation metrics.", "selection result"
    ),
    "phase6_raw_results.csv": SourceDetails(
        "Case-level robustness and sensitivity results.", "case-method result"
    ),
    "phase6_aggregated_summary.csv": SourceDetails(
        "Derived descriptive robustness aggregates.", "aggregate statistic"
    ),
    "phase6_selected_parameters.csv": SourceDetails(
        "Parameter-selection outputs by sensitivity case.", "selection result"
    ),
    "phase7_case_metadata.csv": SourceDetails(
        "Matrix-free case configuration and data-consistency metadata.", "case metadata"
    ),
    "phase7_iterative_results.csv": SourceDetails(
        "Iterative solver trajectories, budgets, tolerances, and timings.", "iterative solver result"
    ),
    "phase7_small_system_validation.csv": SourceDetails(
        "Small-system iterative reference validation.", "validation result"
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(not name for name in reader.fieldnames):
            raise ValueError(f"Malformed CSV header in {path}")
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def phase_number(filename: str) -> int:
    """Extract the validated phase number from a Phase 2--7 metrics name."""
    try:
        value = int(filename.removeprefix("phase").split("_", 1)[0])
    except ValueError as error:
        raise ValueError(f"Cannot determine phase number from {filename}") from error
    if value not in range(2, 8):
        raise ValueError(f"Expected a Phase 2--7 filename, received {filename}")
    return value


def build_source_registry(metrics_directory: Path) -> list[dict[str, object]]:
    """Discover saved Phase 2--7 CSV sources and record their loading state."""
    discovered = {
        path.name for path in metrics_directory.glob("phase[2-7]_*.csv")
    }
    filenames = sorted(discovered | set(SOURCE_DETAILS))
    registry: list[dict[str, object]] = []
    for filename in filenames:
        path = metrics_directory / filename
        details = SOURCE_DETAILS.get(
            filename,
            SourceDetails(
                "Discovered Phase 2--7 metrics CSV; purpose is not yet classified.",
                "unclassified record",
            ),
        )
        exists = path.is_file()
        registry.append(
            {
                "phase_number": phase_number(filename),
                "relative_file_path": f"results/metrics/{filename}",
                "exists": exists,
                "loaded": False,
                "analytical_purpose": details.purpose,
                "record_type": details.record_type,
                "row_count": "",
                "source_classification": "included" if exists else "unavailable",
                "reason": "" if exists else "Expected registry source is missing.",
                "columns": "",
            }
        )
    return registry


def load_registered_sources(
    metrics_directory: Path, registry: list[dict[str, object]]
) -> dict[str, list[dict[str, str]]]:
    """Load every available registry source and expose malformed files clearly."""
    data: dict[str, list[dict[str, str]]] = {}
    for entry in registry:
        filename = Path(str(entry["relative_file_path"])).name
        if not entry["exists"]:
            continue
        try:
            rows = read_csv(metrics_directory / filename)
        except (OSError, csv.Error, ValueError) as error:
            entry["source_classification"] = "unavailable"
            entry["reason"] = f"Could not load source: {error}"
            raise RuntimeError(f"Cannot load registered source {filename}: {error}") from error
        entry["loaded"] = True
        entry["row_count"] = len(rows)
        entry["columns"] = "|".join(rows[0].keys()) if rows else ""
        data[filename] = rows
    return data


def run_legacy_preliminary_analysis() -> None:
    """Legacy coupled synthesis retained for provenance, not CLI-dispatched.

    This function predates P8-06.  The safe public CLI uses only registered
    stages below, so no single-stage command can create its plots or report.
    """
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    registry = build_source_registry(METRICS)
    data = load_registered_sources(METRICS, registry)
    write_csv(OUT_CSV / "phase8_source_inventory.csv", registry)
    unified: list[dict[str, object]] = []
    for entry in registry:
        filename = Path(str(entry["relative_file_path"])).name
        for source_row, row in enumerate(data.get(filename, []), start=1):
            unified.append(
                {
                    "source_phase": f"phase{entry['phase_number']}",
                    "source_file": filename,
                    "source_row": source_row,
                    **row,
                }
            )
    write_csv(OUT_CSV / "phase8_unified_records.csv", unified)

    # The baseline and Phase 4 comparison share the same 8x8 setup; preserve
    # original values and source provenance rather than mixing unlike cases.
    matched = []
    for filename in ("phase2_baseline_comparison.csv", "phase4_regularization_comparison.csv"):
        for source_row, row in enumerate(data[filename], start=1):
            matched.append(
                {
                    "source_phase": f"phase{phase_number(filename)}",
                    "source_file": filename,
                    "source_row": source_row,
                    **row,
                }
            )
    write_csv(OUT_CSV / "phase8_small_system_baseline_regularization.csv", matched)

    p2 = data["phase2_baseline_comparison.csv"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in p2:
        try:
            x = float(row["residual_norm"])
            y = float(row["relative_reconstruction_error"])
        except (KeyError, ValueError):
            continue
        ax.scatter(x, y, label=row.get("solver", "method"))
        ax.annotate(row.get("solver", ""), (x, y), fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="Residual norm", ylabel="Relative reconstruction error", title="Phase 2: fit versus reconstruction error")
    ax.grid(True, which="both", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "phase2_fit_vs_error.png", dpi=160)
    plt.close(fig)

    rows = data["phase7_iterative_results.csv"]
    chosen = [r for r in rows if r.get("run_type") == "iteration_budget" and r.get("case_id") == "resolution_64_sigma_1_5"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sorted({r["method"] for r in chosen}):
        subset = sorted((r for r in chosen if r["method"] == method), key=lambda r: int(r["requested_iteration_limit"]))
        ax.plot([int(r["requested_iteration_limit"]) for r in subset], [float(r["relative_reconstruction_error"]) for r in subset], marker="o", label=method)
    ax.set(xlabel="Iteration budget", ylabel="Relative reconstruction error", title="Phase 7: 64×64, sigma=1.5 early stopping")
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "phase7_early_stopping_error.png", dpi=160)
    plt.close(fig)

    report = f"""# Phase 8 — Integrated Experimental Analysis

## Scope and evidence

This synthesis reads saved Phase 2–7 CSV metrics. It does not rerun prior experiments. The script `experiments/phase8_integrated_analysis.py` writes the provenance-preserving unified records and figures. Each unified row retains its source phase and original source file. The matched subset is limited to the Phase 2/4 small-system comparison; other phases use different resolutions, seeds, or protocols and are not pooled as if identical.

## Unregularized methods and conditioning

Phase 2 provides direct inversion, QR, and normal-equation measurements in `phase2_baseline_comparison.csv`. These records show that a small residual does not guarantee a useful reconstruction: the direct inverse/QR rows report relative reconstruction error around 1.95×10⁴ despite residual near numerical zero. Phase 3 singular-value and Picard/noise-amplification CSVs explain this instability: inverse weighting magnifies components associated with small singular values. Condition estimates depend on the stated numerical-rank threshold and are not interchangeable across matrix sizes.

## Explicit regularization and selection

Phase 4 compares TSVD and Tikhonov through saved reconstruction errors, residuals, and filter outputs. TSVD truncates small-singular-value components; Tikhonov smoothly attenuates them. Neither method dominates independent of blur and chosen parameter. Phase 5 and Phase 6 selected-parameter tables compare L-curve, GCV, and discrepancy selection. The documented TSVD-GCV rank-142 selection is retained as a valid outcome of its criterion/grid and exhibits catastrophic ground-truth error in that case; ground truth is an evaluation metric after selection, never an input to selection.

## Robustness and iterative methods

Phase 6 raw and aggregated data vary noise seed, blur, and resolution; these are interpreted within their sensitivity groups. Repeated rows reporting one shared failed case are not counted as independent failures. Phase 7 compares LSQR, CGLS, and damped LSQR using common observations within each case. Its 64×64 sigma=1.5 trajectories show semiconvergence: CGLS error improves early and later rises while residual continues to fall. Damping and early stopping trade residual fit against reconstruction error. Phase 7's matrix-free operator stores O(n²) one-dimensional factors but its dense factor products cost O(n³) arithmetic per forward or adjoint application for square images; measured solver timings are machine-specific, not complexity estimates.

## Integrated comparison and limitations

The generated `phase8_small_system_baseline_regularization.csv` retains the compatible Phase 2/4 records. Phase 6 and Phase 7 are separate protocols, so their raw values are not treated as a controlled head-to-head ranking. `phase2_fit_vs_error.png` visualizes the distinction between residual and reconstruction error; `phase7_early_stopping_error.png` visualizes iterative semiconvergence. Source CSVs remain the authority for exact metrics and configuration.

There is no single universal best method in this evidence. Explicit regularization offers a tunable spectral filter; parameter-selection rules can fail on particular spectra; iterative methods scale storage well but still incur dense-factor arithmetic and can semi-converge. The study uses limited image sources, noise realizations, and parameter grids. Not every phase exports identical metrics or compatible conditions, so unavailable controlled comparisons are reported as limitations rather than filled with estimates.
"""
    (ROOT / "report" / "phase8_integrated_experimental_analysis.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    from analysis.phase8_pipeline import (
        available_stage_names,
        execute_core_pipeline,
        execute_stage,
        format_stage_listing,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(*available_stage_names(), "full-core-pipeline"),
        help="Run exactly one isolated stage, or the explicit full core pipeline.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List stage responsibilities and upstream dependencies.",
    )
    arguments = parser.parse_args()
    if arguments.list_stages:
        print(format_stage_listing())
    elif arguments.stage is None:
        parser.error("--stage is required unless --list-stages is used.")
    else:
        handlers = phase8_stage_handlers()
        if arguments.stage == "full-core-pipeline":
            execute_core_pipeline(ROOT, handlers)
        else:
            execute_stage(arguments.stage, ROOT, handlers)
