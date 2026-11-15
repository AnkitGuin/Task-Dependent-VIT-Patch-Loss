# Journal revision execution status - 15 November 2026

Current manuscript: **Selecting Frozen Linear Patch Operators: An Empirical Audit under Matched Feature Budgets**, sole author Ankit Guin.

- [x] Add an independent dataset (SVHN).
- [x] Use numerically refined LogME and adaptive H-score in the main comparison.
- [x] Use one corrected, sufficiently broad ridge grid and disclose endpoints.
- [x] Lock and evaluate a CIFAR-developed fixed policy on independent SVHN.
- [x] Evaluate full official test sets with paired image/seed uncertainty.
- [x] Measure full selection time and peak memory on identical hardware.
- [ ] Publish prepared, content-clean code/evidence; public verification follows upload.
- [x] Rewrite the contribution/discussion around the empirical decision audit.

Measured results and exact limitations: [revision summary](../reports/selection_revision/revision_summary.md). Reproduction: [guide](selection_revision_reproducibility.md). Review mapping: [response](reviewer_response_20261115.md).

Final cohort: `runs/selection_revision_final`; 420 heads / 30 units. Partial narrow-grid runs and overlapping cost warm-ups are retained locally and excluded. Source fingerprints, complete pretest artifacts, 420 independent checkpoint replays, all official test predictions and 198 isolated cost measurements support the reported tables. The historical study remains preliminary, never retrospectively confirmatory.

Publication receipt: `reports/selection_revision/public_release.json` is created only after successful anonymous access to the release. The source/evidence archives were built and verified before publication; their original pre-publication execution record is preserved in the tagged version. The current main-branch documentation can subsequently record the public receipt without changing experimental source or tagged results.

No archival DOI, clean-install validation on a second machine, broader-domain generalization or guaranteed acceptance is claimed. Author affiliation/contact details and a journal-specific submission format remain for the author to finalize.
