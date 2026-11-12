# Phase 8 Metadata Reconciliation

## Method and evidence hierarchy

This P8-08 layer audits 14 metadata dimensions for every canonical observation
without editing the canonical dataset, record-type audit, or compatibility
audit. It uses explicit canonical fields first, then attributable experiment
runners, Phase 4/5 protocol documentation, and Phase 6/7 configuration JSON.
No filename, default, or cross-phase similarity is treated as evidence by
itself. The overlay contains only recovered values and their evidence location.

## Status definitions

- `present_verified`: already explicit in the canonical observation.
- `reconciled`: blank canonical field recovered from attributable repository
  evidence; the original canonical value remains unchanged.
- `missing`: relevant to the record but no attributable evidence was found.
- `not_applicable`: irrelevant to that source role, such as a noise seed for a
  noise-free spectral operator analysis.
- `unmapped_source_field`: an original source field retained only in
  `source_attributes_json`; no canonical mapping was justified.
- `conflicting`: reserved for incompatible explicit evidence. None occurred in
  this audit.

## Generated quality summary

The audit contains 122,459 field-level outcomes: 8,741 canonical records times
14 examined fields, plus 85 schema-audit source-field rows. Field counts are
not unique-record counts.

| Status | Fields |
|---|---:|
| present_verified | 25,283 |
| reconciled | 47,389 |
| missing | 40,938 |
| not_applicable | 8,764 |
| unmapped_source_field | 85 |
| conflicting | 0 |

The reconciled overlay has 47,389 rows. Recovered field rows occur for Phase 2
(21 canonical records), Phase 3 (587), Phase 4 (271), Phase 5 (2,451), Phase 6
(1,254), and Phase 7 (1,064). This describes records with at least one overlay
entry, not scientific replication counts.

## Representative recovery

- Phase 2 baseline records recover the 12×12 camera setup, Gaussian blur,
  sigma 1.5, kernel 9, additive Gaussian noise 0.01, and seed 42 from
  `experiments/phase2_baseline_reconstruction.py` and explicit project config
  use in that runner.
- Phase 4/5 records recover the documented shared 12×12 camera protocol,
  zero boundary, noise model, and shared observation identity from the Phase
  4/5 handoff report and Phase 4 protocol.
- Phase 6 case-linked records recover settings from
  `results/metrics/phase6_experiment_config.json`; Phase 7 case-linked records
  recover scalable-case settings from `results/metrics/phase7_experiment_config.json`.

## Remaining gaps and P8-02 impact

The most frequent unresolved fields are configuration identifier (6,423 field
rows), followed by image/blur/noise descriptors for records whose source role
does not carry a case-specific experiment identity. Aggregate summaries and
small-system validation also lack some record-specific metadata. These remain
missing rather than inferred.

P8-02 remains unchanged. The overlay identifies explicit evidence that could
support a later, separately authorized compatibility-audit rerun. It does not
retroactively change any existing `unknown`, `compatible`, or `incompatible`
outcome. The recovered values also do not establish equal observations across
phases without an explicit shared-observation source.

## Limits

No conflicting explicit values were found by the implemented attributable
profiles. This is not proof that every historical setting is consistent: it
means no profile supplied two incompatible values for the same canonical field.
Source image identity, observation identity, stopping configuration, and rank
tolerance remain missing where they were not recorded for that specific role.
