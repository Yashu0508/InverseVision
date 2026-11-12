# Phase 8 P8-03 Regeneration Audit

## Scope and evidence

This is a read-only audit of the P8-03 source-inventory implementation run. No
pipeline, test, plot, or experiment was rerun for this audit. Findings marked
**verified** come from the recorded P8-03 command/output in this conversation,
the current file metadata, and the current source code. Findings about content
before the P8-03 run are limited because this directory is not a Git repository
and no Phase 8 backup or execution log is available.

## 1. Affected-file inventory

The recorded P8-03 command was:

```powershell
python -m pytest tests/test_phase8_source_inventory.py -q
python experiments/phase8_integrated_analysis.py
```

The source-inventory test completed before the script command and does not
write Phase 8 artifacts. The script command called its complete `main()`
function. Current timestamps establish the following artifact changes at
2026-11-12 04:04:45--04:04:46 local time.

| File | P8-03 status | Evidence |
|---|---|---|
| `results/metrics/phase8/phase8_source_inventory.csv` | **Newly created** | Creation time and last-write time are both 04:04:45. |
| `results/metrics/phase8/phase8_unified_records.csv` | **Overwritten** | Creation time is 03:37:36; last-write time is 04:04:45. |
| `results/metrics/phase8/phase8_small_system_baseline_regularization.csv` | **Overwritten** | Creation time is 03:43:02; last-write time is 04:04:45. |
| `results/plots/phase8/phase2_fit_vs_error.png` | **Overwritten** | Creation time is 03:43:02; last-write time is 04:04:45. |
| `results/plots/phase8/phase7_early_stopping_error.png` | **Overwritten** | Creation time is 03:43:02; last-write time is 04:04:46. |
| `report/phase8_integrated_experimental_analysis.md` | **Overwritten** | Creation time is 03:43:02; last-write time is 04:04:46. |

No other Phase 8 output files currently exist. No evidence shows that a
Phase 2--7 result, report, configuration, image, or numerical source file was
written by this run. The script reads only `results/metrics/phase[2-7]_*.csv`
and writes only under `results/metrics/phase8`, `results/plots/phase8`, and
the Phase 8 report path above.

## 2. Verified execution sequence

### Recorded command evidence

**Verified:** The P8-03 command shown above executed successfully after three
focused tests passed. It printed an inventory of 17 existing, loaded sources
and a regenerated `phase8_unified_records.csv` count of 1,399 rows.

**Verified:** `experiments/phase8_integrated_analysis.py` has one `main()`
entry point. The recorded command invokes the file directly, so Python ran the
whole `main()` body, beginning at its output-directory creation and ending at
the report write. There is no command-line argument, stage selector, or early
return that could restrict execution to source inventory.

### Write sequence in the current implementation

The current code establishes this order:

1. `build_source_registry(METRICS)` discovers known and on-disk Phase 2--7
   CSVs.
2. `load_registered_sources(...)` reads every available registry source.
3. `write_csv(...)` writes `phase8_source_inventory.csv`.
4. `write_csv(...)` writes the raw union table.
5. `write_csv(...)` writes the Phase 2/4 subset.
6. Matplotlib `savefig(...)` writes the Phase 2 and Phase 7 figures.
7. `Path.write_text(...)` writes the preliminary integrated report.

The P8-03 invocation occurred after this implementation was patched. The
timestamps follow this source order, supporting the conclusion that all listed
outputs were regenerated during the recorded run. This does not establish the
byte-for-byte content of the pre-run files.

### Independent source-inventory execution

**Verified:** No standalone inventory command or callable command-line stage
exists. `build_source_registry` and `load_registered_sources` are separately
defined functions, but the executable file only calls `main()`. Therefore the
inventory can be called independently by Python code or tests, but not by the
documented script invocation without also triggering later writes.

## 3. Input-to-output dependency mapping and validity

| Regenerated output | Direct inputs in current code | Did newly loaded Phase 3/6/7 sources enter? | Compatibility check | Assessment |
|---|---|---|---|---|
| `phase8_source_inventory.csv` | All discovered Phase 2--7 metrics CSV filenames and headers | Yes: all 17 inventory sources are listed and loaded | Not applicable; this is a registry | Scientifically appropriate as a source inventory. |
| `phase8_unified_records.csv` | All 17 registered Phase 2--7 CSVs | Yes, including Phase 3 structured comparison, Phase 6 selected parameters, and Phase 7 metadata/small validation | None | Provenance-preserving raw union only. It combines heterogeneous record types in columns and must not be treated as an analytical comparison. |
| `phase8_small_system_baseline_regularization.csv` | `phase2_baseline_comparison.csv`; `phase4_regularization_comparison.csv` | No | None; code only appends rows | It is an unverified subset, not a scientifically validated matched comparison. Phase 2/4 setup similarity is asserted in a comment but not encoded or checked. |
| `phase2_fit_vs_error.png` | `phase2_baseline_comparison.csv` | No | Not needed within one source, but no explicit source-status check | A descriptive Phase 2 plot. It does not compare across configurations. It correctly shows residual and error as separate axes, subject to the source metrics. |
| `phase7_early_stopping_error.png` | `phase7_iterative_results.csv`, filtered to `resolution_64_sigma_1_5` and `iteration_budget` | No | Implicit filter only; no explicit metadata audit | A bounded within-case trajectory view. Its filter prevents mixing Phase 7 cases, but it does not validate the case metadata or establish a formal compatibility group. |
| `phase8_integrated_experimental_analysis.md` | Static string in the script; it names some source files | No programmatic use of the newly loaded sources | None | Preliminary narrative only. It does not derive its statements from a validated normalized dataset or the inventory. |

The current union table includes spectral components, candidate sweeps,
aggregate statistics, selection results, case metadata, validation rows, and
solver trajectories. Those units are deliberately not equivalent. The union
CSV includes source phase/file/row provenance but has no record-type field,
configuration signature, analytical mapping, or eligibility rule. It is
useful as a raw integration artifact, not a scientifically justified
comparison dataset.

## 4. Data loss and corruption assessment

### Phase 2--7 sources

**Verified:** The P8-03 script has no write operation targeting Phase 2--7
source paths. The current source inventory contains all 17 on-disk Phase 2--7
metric CSVs and marks each `exists=True`, `loaded=True`. SHA-256 checks were
also recorded during this audit for representative inputs, including the three
previous omissions and the Phase 2 baseline. These checks establish present
readability and a current fingerprint; they cannot prove a historical file was
never changed.

There is no evidence of Phase 2--7 corruption or data loss from P8-03.

### Phase 8 outputs

**Verified:** Five pre-existing Phase 8 artifacts were overwritten. The raw
union table changed from the previously reported 976 rows to 1,399 rows after
the registry added five omitted source files. The Phase 2/4 CSV gained source
row provenance in the current script; its present length is 7,060 bytes,
compared with 6,992 bytes recorded before P8-03. The two image files have the
same present byte sizes that were recorded before P8-03, but their previous
hashes are unavailable, so byte identity cannot be confirmed. The preliminary
report was rewritten by the same static string path.

The pre-P8-03 forms of these Phase 8 artifacts are not recoverable from this
workspace through Git: `git rev-parse`, `git status`, and `git log` all report
that `D:\Image-Deblurring-Inverse-Problem` is not a Git repository. No backup,
archive, or Phase 8 execution log was found in the inspected project paths.
Recovery could still be possible from an external user backup, operating-system
file history, cloud synchronization version history, or a prior exported copy.
This audit does not restore or attempt recovery.

The source experiment data remain intact. The overwritten Phase 8 artifacts
were preliminary, regenerable analysis products. Their prior exact contents
cannot be established from the available evidence.

## 5. Recommended safe execution design

Future single-issue work should make stages explicit and avoid an implicit
full-pipeline call. A small, safe design is:

```text
run_source_inventory()   -> phase8_source_inventory.csv only
run_raw_integration()    -> raw union CSV only
run_comparisons()        -> comparison CSVs only
run_plots()              -> plots only
write_report()           -> report only
```

Expose these stages through a command-line argument such as
`--stage inventory`, with the default requiring an explicit `--stage all`.
Each stage should declare its inputs and output paths, use the source registry
for validation, and return an execution summary. Focused tests can call the
pure registry/loading functions without writing artifacts. This makes P8-03
validation a targeted inventory operation and prevents it from rewriting
comparisons, figures, or narrative reports.

The raw union should remain clearly labelled as a provenance archive. A later
canonical normalization stage must use record-type and configuration metadata
before it writes analytical tables or claims cross-phase comparability.

## Verdict

**Regeneration with scientifically invalid or unverified outputs.** The
regeneration itself appears mechanically safe for Phase 2--7 data: it created
the inventory and rewrote only Phase 8 artifacts. However, the regenerated
raw union and Phase 2/4 subset lack the schema/compatibility safeguards needed
for scientific comparison, and the report is not dynamically validated.
There is no evidence of source-data corruption. There is insufficient evidence
to recover or compare the exact pre-P8-03 Phase 8 artifact contents.
