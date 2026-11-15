# Refined study: evidence and submission scope

The manuscript is now **Selecting Compressed Patch Tokenizers: An Audit of Task-Alignment Scores under Fixed Feature Budgets**. It replaces the earlier primarily illustrative synthetic paper with a focused empirical/methodological study. The previous manuscript is retained under `paper/archive_preselection_20261114/`. No manuscript or new research artifact has been pushed to GitHub.

## Work completed

- Wrote a prospective local primary protocol before running the new study, with separate labeled protocols for later exploratory analyses. This is not external preregistration.
- Audited close prior work on H-score, robust covariance estimation, LogME, cross-validated distances, patchification scaling and pretrained DeiT. The mathematical ingredients are explicitly attributed as established work.
- Implemented an offline selection pipeline with operator fitting, diagnostic estimation, downstream validation and official-test evaluation separated. Saved pretest score choices and hyperparameters before test outcomes were computed.
- Ran CIFAR-10 and CIFAR-100 at native RGB resolution, five seeds each, two scalar budgets, fourteen tokenizer candidates: 20 units and 280 selected heads, each from a five-penalty validation path. Added a raw-feature reference outside the matched-budget bank.
- Compared eight primary selectors, plus separately labeled adaptive-H-score and profile-evidence sensitivity analyses. Added nested diagnostic label budgets of 200, 400 and 1,000, all fixed-policy controls, a validation-only leave-one-dataset-out policy, and restricted candidate banks.
- Audited real ImageNet-trained DeiT-tiny and ViT-B/16 patch matrices, including dtype-aware rank and singular values. The latter is full column rank; broad claims that all standard ViTs have a forced patch-rank bottleneck are inappropriate.
- Verified the isotropic null mean-TAR Beta calibration and an anisotropic Gaussian same-rank/same-Euclidean-retention risk counterexample.
- Trained matched 6/80-epoch-cap relational controls for raw MLP, tiny ViT and CNN, with five seeds and fresh render noise. The longer-budget tiny ViT solves held-out XOR across all five seeds; short-run failure is not a representation impossibility certificate.
- Added actual held-out shifts of equal-area orientation shapes, preserving labels, rather than presenting matched uniform-phase changes as distribution shift.
- Replayed all 280 primary checkpoints, exactly matching predictions, source snapshots and regenerated split hashes. Checked exact-content overlap in the selected official partitions and reported a filtered-test sensitivity result.
- Expanded offline verification to 68 passing tests, with Ruff passing. New tests independently check ridge solutions, integrated Gaussian evidence, adaptive H-score, cross-fitting/partition boundaries, matrix equivalence and immutable pretest artifacts.
- Found that 264/280 original validation paths improved from penalty 100 to 1,000. Ran a separate validation-only expanded grid through 100,000,000, retaining the same operators and score choices: another 280 selected heads. Only two select the new upper boundary. Independently replayed all 280 new heads plus 20 raw references, matching saved predictions and regenerated data hashes.
- Added a stronger diagnostic baseline that jointly tunes candidate and penalty within the same 1,000-label pool. Applied adaptive H-score and profile LogME choices to both head grids, with all outcomes retained. Added five diagnostic-label permutations per unit as a negative control.

## Principal measured findings

Mean selection regret is the best measured bank accuracy minus score-selected accuracy, in **accuracy percentage points**. It is conditional on these datasets and this candidate bank, not a population Bayes-risk or universal recommendation.

| Selector | Original head grid | Expanded head grid |
|---|---:|---:|
| Covariance retention | 0.3375 | 0.3950 |
| Diagnostic held-out ridge probe, fixed penalty | 0.46625 | 0.4275 |
| Cross-fitted retention | 0.4750 | 0.5125 |
| Mean retention | 0.7875 | 0.7425 |
| Reconstruction-energy retention | 0.7875 | 0.7425 |
| LogME, primary short iteration cap | 1.0075 | 0.5900 |
| Rank, uniform tie expectation | 2.6591 | 2.4595 |
| Fixed-shrinkage H-score | 3.8850 | 2.9875 |

The expanded grid is an explicitly post-primary correction to a validation-boundary concern, not independent confirmatory evidence. Each grid has its own empirical test oracle. Its small covariance-retention/probe gap has a paired conditional 95% bootstrap interval containing zero. Matching patch size and output budget changes the ordering: expanded-grid within-patch-bank regret is 0.1825 points for LogME, 0.210625 for covariance retention and 0.64625 for mean retention. Smaller-bank regrets use different oracles and are not numerically interchangeable with whole-bank regret.

The aggregate ordering is not uniform. Mean retention outperforms covariance retention on CIFAR-100 at the larger budget; LogME narrowly leads on CIFAR-10 at the smaller budget. Conditional intervals and per-unit data are in `reports/selection/metric_summary.csv` and `metric_by_dataset_budget.csv`.

The adaptive-H-score exploratory result has 1.50 points of mean regret; profile-evidence LogME has 0.9225 points. A post-hoc audit of all fourteen fixed policies identifies p=4 DCT with observed regret 0.1825 points, lower than the primary selectors. Because this policy is identified after looking at test outcomes, it is **not** a prospectively validated deployment rule. A separate policy chosen solely from the other dataset's validation performance has 0.6625 points of mean regret. These results limit any claim that a complicated new selector is necessary.

With expanded heads, adaptive H-score improves to 1.1675 points and profile-evidence LogME to **0.4225**, close to covariance retention's 0.3950. The stronger evidence baseline therefore materially narrows the apparent gap; the manuscript reports it alongside the original approximate implementation.

Mean retention and reconstruction choose p=16 PCA in all twenty units. Strong average rank correlation also does not guarantee the smallest top-choice regret. These practical distinctions, rather than a claim of a universally superior new metric, form the refined empirical contribution.

The expanded-grid hindsight fixed-policy audit gives p=4 DCT 0.425 points of regret, compared with 0.395 for covariance retention; this policy remains post-hoc. The joint diagnostic-probe grid gives 0.621875 points with original heads and 0.704375 with expanded heads. In 100 unit/permutation cases, shuffling diagnostic labels never changes mean retention's selected candidate, whereas covariance retention's choice changes every time. Shuffled-label covariance regret rises to 0.816 points with original heads and 0.771 with expanded heads. These are descriptive negative controls, not independent tasks or a formal permutation significance test.

## Data quality and provenance

Exact official indices are disjoint, but the content audit finds one exact pixel duplicate between operator/diagnostic subsets and another between diagnostic/test subsets, in different CIFAR-100 seeds. Primary partitions are retained as specified. A separate image-hash-based evaluation removes one affected test example, keeping scores and heads fixed. Near-duplicate independence is not certified.

The real-image study used 1,000 operator-fit, 1,000 diagnostic, 500 validation and 2,000 official-test examples per dataset/seed. Only operator-fit images determine PCA maps; every diagnostic receives the same labeled diagnostic pool. The selected head fits 2,000 images, and test labels do not choose metrics, hyperparameters or primary candidate maxima.

Primary raw evidence is in `runs/selection_primary/`; controls in `runs/selection_controls/`; exploratory label-budget/adaptive analyses in `runs/selection_sensitivity/`; profile-evidence analysis in `runs/selection_logme_profile/`; expanded heads in `runs/selection_regularization/`; diagnostic probe tuning in `runs/selection_probe_grid/`; shuffled-label controls in `runs/selection_permutation/`. Reports, regenerated plots, tables, both checkpoint audits and source manifests are in `reports/selection/`. Downloaded source datasets and full model weights remain in the local cache and are not part of the public GitHub release.

## What this supports before submission

This is substantially stronger evidence for a focused methodological or empirical paper than the original synthetic implementation. It supports collaborator review and venue selection. It does **not** certify novelty, journal acceptance, or broad pretrained-ViT performance. A journal could reasonably request broader domains, stronger practical deployment validation, a larger candidate bank or independent replication.

Before submitting, choose a specific suitable venue and article type; complete an expert full-text novelty review; fill in real authorship, affiliations and contribution disclosures; verify that journal's current AI-assistance, data/code and prior-publication policies; and prepare an archival reproducibility release with dataset/weight licensing considered. These are actual remaining submission steps, not claims that software tests determine publishability.
