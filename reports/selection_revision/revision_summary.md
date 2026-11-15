# Final scientific revision: measured results

Author: **Ankit Guin**. Revision: 15 November 2026.

Final evidence: **420 candidate heads in 30 units**, three domains, five fresh seeds and two budgets. All official test images are evaluated. The frozen package source SHA256 is `84fd13c6b0dd458814edcc25580402b621c345a5b3d80fd49e2f91118075a50e`.

| Selector | Mean regret (pp) | Warm selection (s) | Cold selection (s) |
|---|---:|---:|---:|
| Covariance retention | 0.221 | 0.872 | 1.161 |
| Fixed DCT (p=4) | 0.294 | <0.001 | <0.001 |
| Diagnostic probe | 0.336 | 1.932 | 2.271 |
| Tuned diagnostic probe | 0.502 | 1.926 | 2.262 |
| Adaptive H-score | 0.932 | 1.799 | 2.126 |
| Profile LogME | 0.983 | 3.034 | 3.282 |

Costs are medians of eighteen fresh-process measurements per method; the complete benchmark has 198 measurements. Cold includes actual operator construction; warm includes required features, statistics and scores. Head training, test inference, imports and decoding are outside selection. Peak RSS includes cold construction and is reported in `cost_summary.csv`.

| Paired comparison | Accuracy gain (pp) | Joint 95% interval (pp) |
|---|---:|---:|
| Profile LogME minus covariance, pooled | -0.762 | [-1.196, -0.406] |
| Covariance minus fixed DCT, pooled | +0.073 | [-0.056, +0.203] |
| Covariance minus fixed DCT, independent SVHN | +0.316 | [+0.017, +0.639] |

Interpretation: covariance has the lowest pooled main-rule regret in this bank, but its gain over fixed DCT is small and uncertain. On SVHN its choices match unlabeled reconstruction in every unit. Mean retention and reconstruction agree in all thirty units. The SVHN bank has weak ordinary and balanced accuracy, even relative to the empirical test-majority reference. These are central negative results, not reasons to claim universal superiority.

Verification: 72 local tests passed; Ruff passed. All 420 saved heads were independently replayed with NumPy patch extraction and direct Cholesky coefficient checks; all full-test predictions agree exactly. Maximum coefficient difference: 3.119e-13.

The ridge grid has 0 lower-end and 5 upper-end selections; 0 paths strictly improve from the penultimate to the largest value. Ties favor larger penalties.

A secondary class-prior/balanced-accuracy report was added after observing weak SVHN performance. It is explicitly labeled post-outcome and does not replace or tune the primary endpoint.

Public release destination: [https://github.com/AnkitGuin/Task-Dependent-VIT-Patch-Loss/releases/tag/v2026.11.15-v2](https://github.com/AnkitGuin/Task-Dependent-VIT-Patch-Loss/releases/tag/v2026.11.15-v2).

At generation of this record, source/evidence were prepared for publication; anonymous release verification is completed after upload and recorded separately in `public_release.json`.

Before journal submission: supply the author's chosen affiliation/contact information and journal-specific declarations/format. A journal requiring a DOI-based archive may require an additional archival deposit. Journal acceptance and Q1--Q4 suitability are editorial decisions, not outcomes established by passing tests.
