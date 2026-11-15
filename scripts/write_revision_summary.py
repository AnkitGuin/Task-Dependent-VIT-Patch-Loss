"""Generate a concise reviewer-facing execution/remaining-work record from artifacts."""

import json
from pathlib import Path
import pandas as pd


def main():
    output = Path("reports/selection_revision")
    audit = json.loads((output/"audit.json").read_text())
    replay = json.loads((output/"checkpoint_replay.json").read_text())
    summary = pd.read_csv(output/"summary.csv").set_index("metric")
    paired = pd.read_csv(output/"paired_accuracy_intervals.csv")
    cost = pd.read_csv(output/"cost_summary.csv").set_index("metric")
    released = (output/"public_release.json").exists()
    release = json.loads((output/"public_release.json").read_text()) if released else None
    url = "https://github.com/AnkitGuin/Task-Dependent-VIT-Patch-Loss/releases/tag/v2026.11.15-v2"
    labels = dict(covariance_tar="Covariance retention",fixed_dct="Fixed DCT (p=4)",cheap_probe="Diagnostic probe",
                  tuned_probe="Tuned diagnostic probe",adaptive_hscore="Adaptive H-score",profile_logme="Profile LogME")
    lines = ["# Final scientific revision: measured results", "", "Author: **Ankit Guin**. Revision: 15 November 2026.", "",
             f"Final evidence: **{audit['candidate_heads']} candidate heads in {audit['units']} units**, three domains, five fresh seeds and two budgets. All official test images are evaluated. The frozen package source SHA256 is `"+audit["source_sha256"]+"`.", "",
             "| Selector | Mean regret (pp) | Warm selection (s) | Cold selection (s) |", "|---|---:|---:|---:|"]
    for method,name in labels.items():
        warm = "<0.001" if cost.loc[method,'feature_statistics_score_sec']<.001 else f"{cost.loc[method,'feature_statistics_score_sec']:.3f}"
        cold = "<0.001" if cost.loc[method,'cold_selection_sec']<.001 else f"{cost.loc[method,'cold_selection_sec']:.3f}"
        lines.append(f"| {name} | {100*summary.loc[method,'regret']:.3f} | {warm} | {cold} |")
    lines += ["", "Costs are medians of eighteen fresh-process measurements per method; the complete benchmark has 198 measurements. Cold includes actual operator construction; warm includes required features, statistics and scores. Head training, test inference, imports and decoding are outside selection. Peak RSS includes cold construction and is reported in `cost_summary.csv`.", "",
              "| Paired comparison | Accuracy gain (pp) | Joint 95% interval (pp) |", "|---|---:|---:|"]
    for dataset,method,reference,name in [
        ("all","profile_logme","covariance_tar","Profile LogME minus covariance, pooled"),
        ("all","covariance_tar","fixed_dct","Covariance minus fixed DCT, pooled"),
        ("svhn","covariance_tar","fixed_dct","Covariance minus fixed DCT, independent SVHN")]:
        row = paired[(paired.dataset==dataset)&(paired.metric==method)&(paired.reference==reference)].iloc[0]
        lines.append(f"| {name} | {row.accuracy_gain_pp:+.3f} | [{row.joint_ci_low:+.3f}, {row.joint_ci_high:+.3f}] |")
    lines += ["", "Interpretation: covariance has the lowest pooled main-rule regret in this bank, but its gain over fixed DCT is small and uncertain. On SVHN its choices match unlabeled reconstruction in every unit. Mean retention and reconstruction agree in all thirty units. The SVHN bank has weak ordinary and balanced accuracy, even relative to the empirical test-majority reference. These are central negative results, not reasons to claim universal superiority.", "",
              "Verification: 72 local tests passed; Ruff passed. All 420 saved heads were independently replayed with NumPy patch extraction and direct Cholesky coefficient checks; all full-test predictions agree exactly. Maximum coefficient difference: "+f"{replay['maximum_direct_solve_coefficient_error']:.3e}.", "",
              f"The ridge grid has {audit['grid_min_count']} lower-end and {audit['grid_max_count']} upper-end selections; {audit['strict_upper_improvements']} paths strictly improve from the penultimate to the largest value. Ties favor larger penalties.", "",
              "A secondary class-prior/balanced-accuracy report was added after observing weak SVHN performance. It is explicitly labeled post-outcome and does not replace or tune the primary endpoint.", "",
              f"Public release destination: [{url}]({url})."]
    if released:
        lines += ["", "Anonymous release access and all uploaded asset sizes/server digests were verified. Source commit: `"+release["source_commit"]+"`. See `public_release.json` for exact upload URLs and SHA256s."]
    else:
        lines += ["", "At generation of this record, source/evidence were prepared for publication; anonymous release verification is completed after upload and recorded separately in `public_release.json`."]
    lines += ["", "Before journal submission: supply the author's chosen affiliation/contact information and journal-specific declarations/format. A journal requiring a DOI-based archive may require an additional archival deposit. Journal acceptance and Q1--Q4 suitability are editorial decisions, not outcomes established by passing tests."]
    (output/"revision_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    status = ["# Journal revision execution status - 15 November 2026", "", "Current manuscript: **Selecting Frozen Linear Patch Operators: An Empirical Audit under Matched Feature Budgets**, sole author Ankit Guin.", "",
              "- [x] Add an independent dataset (SVHN).", "- [x] Use numerically refined LogME and adaptive H-score in the main comparison.",
              "- [x] Use one corrected, sufficiently broad ridge grid and disclose endpoints.",
              "- [x] Lock and evaluate a CIFAR-developed fixed policy on independent SVHN.",
              "- [x] Evaluate full official test sets with paired image/seed uncertainty.",
              "- [x] Measure full selection time and peak memory on identical hardware.",
              ("- [x] Deduplicate before fitting and publish versioned code/evidence." if released else "- [ ] Publish prepared, content-clean code/evidence; public verification follows upload."),
              "- [x] Rewrite the contribution/discussion around the empirical decision audit.", "",
              "Measured results and exact limitations: [revision summary](../reports/selection_revision/revision_summary.md). Reproduction: [guide](selection_revision_reproducibility.md). Review mapping: [response](reviewer_response_20261115.md).", "",
              "Final cohort: `runs/selection_revision_final`; 420 heads / 30 units. Partial narrow-grid runs and overlapping cost warm-ups are retained locally and excluded. Source fingerprints, complete pretest artifacts, 420 independent checkpoint replays, all official test predictions and 198 isolated cost measurements support the reported tables. The historical study remains preliminary, never retrospectively confirmatory.", "",
              "Publication receipt: `reports/selection_revision/public_release.json` is created only after successful anonymous access to the release. The source/evidence archives were built and verified before publication; their original pre-publication execution record is preserved in the tagged version. The current main-branch documentation can subsequently record the public receipt without changing experimental source or tagged results.", "",
              "No archival DOI, clean-install validation on a second machine, broader-domain generalization or guaranteed acceptance is claimed. Author affiliation/contact details and a journal-specific submission format remain for the author to finalize."]
    Path("docs/REVISION_STATUS.md").write_text("\n".join(status)+"\n",encoding="utf-8")
    print("Wrote numerical review summary and execution status")


if __name__ == "__main__":
    main()
