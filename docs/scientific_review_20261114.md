# Scientific code review — 14 November 2026

The corrected CPU cohort is `runs/review_20261114/`; its tables, figures, manifest and artifact audit are in `reports/review_20261114/`. The original 25 runs from 7 November are retained as historical records. Their checksums still pass, but that certifies file integrity, not the validity of their original scientific interpretation.

## Findings and corrections

| Finding | Scientific consequence | Correction and evidence |
|---|---|---|
| S1 randomized bases also varied along the label direction | Shifted continuous class supports could overlap; the multi-base generator did not establish its claimed zero raw-image Bayes risk | Center the planted patch on a fixed hyperplane perpendicular to the null vector, then add the signed perturbation. Its sign is now an explicit raw-image oracle; independently tested over many bases. The original fixed-base two-point theorem was already correct. |
| MC reused the analytic row projector and scored raw images | Agreement did not independently verify the likelihood rule from observable tokens; roundoff could introduce a spurious raw-space direction | Compute `Z=WX`, whiten via the token SVD, and score only those features. Tests forbid access to the analytic projector and check redundant, nonorthonormal, zero-rank and scaled operators. |
| MC used balanced labels and a constant null-signal decision | At exact zero signal its error count was fixed, rather than binomial | Use IID equal-prior labels for MC only. Balanced training datasets remain balanced. |
| `theory_in_ci` used `max(SE, 1/(2*sqrt(n)))`, plus an allowance | The saved coverage flag used a wider band than the reported confidence interval | Use the actual endpoints of a pointwise 95% Clopper–Pearson binomial interval for both saved coverage and figures. Zero observed errors still have nonzero upper uncertainty. |
| An absolute singular-value floor and absolute signal/TAR cutoffs | Uniformly scaling a representation or dataset could spuriously erase rank or change the diagnostic | Use a truly relative SVD threshold; remove absolute signal clipping in the token oracle and absolute nonzero-mean-energy clipping in TAR. Test scales from `1e-40` to `1e40`, and a tiny signal with high signal/noise ratio. |
| Gaussian signal builder always required row and null spaces | Valid full-rank alpha=1 and zero-rank alpha=0 controls were rejected | Require only the subspaces used at the requested endpoint. |
| MNIST train/validation subsets were sampled independently from official training data | Leakage was possible despite the README's disjoint-split promise | Partition official training indices once per seed, reserve a fixed validation pool, reject oversized requests and record selected source IDs. Split logic is tested offline; E10 itself remains unrun. |
| XOR translations used slices with separately wrapped endpoints | A marker could disappear when a slice crossed the boundary | Render both markers first, then apply a periodic roll. Test positive, negative and multi-period shifts. |
| E08 silently set width equal to height; overlap could omit borders | Reported shapes or observed pixel coverage could differ from the request | Honor rectangular shapes and reject grids that drop borders. |
| Reference matrices included affine bias in every basis column | The materialized matrix was incorrect for an affine operator | Subtract its zero-input offset and restore training/evaluation mode after inspection. |
| Bad configs could silently truncate samples or substitute tasks | Labels, counts and declared comparisons could change unintentionally | Reject invalid heads, channels, split multiples, shapes, tasks, training settings, ranks and Gaussian grids before creating a run. |
| Nonfinite training could fail indirectly or produce uninterpretable output | Failed optimization was not consistently explicit | Validate data and metric inputs; fail immediately on nonfinite training/validation losses. Save failure records through the runner. |
| Transformer cost proxy omitted depth and major matrix products | Cost comparisons were numerically misleading | Record per-image forward multiply/add proxy `2Nmd + depth*(16Nd² + 4N²d) + 2dK`, total/trainable parameters and optimizer steps. Elementwise operations, backward and optimizer costs remain excluded; CNN FLOPs remain unmeasured. |
| Aggregation did not verify hashes; an empty hash mapping could pass audit | Corrupted or incomplete results could become paper evidence | Share required-artifact, checksum, config fingerprint, identity and finite-value checks across aggregation, audit and paper generation. Audit CLI returns failure status on invalid records. |
| Figures and paper selected latest records using only experiment/seed | Different hyperparameters or implementations could silently replace or pool evidence | Require one explicit configuration/source cohort per experiment; reject mixtures. Paper generation additionally selects current declared configs. Separate CLI run/report output directories preserve historical cohorts. |
| Runs lacked dataset fingerprints and model checkpoints | Reproducing or inspecting a result required unnecessary retraining | Save array hashes, split metadata, package source snapshot/fingerprint, dependency versions, reproduction argument lists and validation-selected state dictionaries. Hash all saved artifacts. |
| Low-pass interpretation went beyond its measurement | Checkerboard annihilation by a linear filter does not prove zero Bayes information for a clipped noisy generator | Test the exact periodic checkerboard operator separately. E07 remains a measured probe degradation result; higher-order information in its stochastic clipped images is not characterized. |
| E05 applied phase offsets to uniformly random-phase gratings in both train and test | It was not a held-out distribution-shift or translation-robustness experiment | Label the matched-distribution phase study explicitly in source rows, figures, README and paper. |
| Noiseless XOR repeats the same four rendered images across splits | This control cannot support unseen-template generalization claims | State that it is a finite-support representation/probe-capacity demonstration. Its population mean symmetry remains valid. |

Python was formatted and linted, including explicit public exports in experiment entry modules. The six shell scripts had pre-existing Git mode changes; their content and modes were not edited by this review.

## Measured corrected results

| Check | Observed result |
|---|---|
| Automated suite | 53 passed |
| Ruff | Passed |
| Full CPU E00–E09 | 14 completed runs, zero failures; 92 source rows |
| Seed coverage | E06 and E09 each use seeds 0, 1, 2; other experiments seed 0 |
| E00 round trip | Maximum absolute error 0 |
| E01 token residual | Maximum pair L2 residual `1.4288057044435644e-15`; paired probe accuracy 0.5 |
| E01 raw oracle | Sign-rule accuracy 1.0; zero risk follows from the corrected hyperplane construction in exact arithmetic |
| E02 independent oracle | 30 conditions; mean absolute discrepancy `0.0038644649606136166`, max `0.011571558163446183`; 0 predictions outside exact pointwise 95% intervals |
| E03 analytic risk | Null-aligned ~0.5; row-aligned `0.10564977366685535`, at sigma 0.4 |
| E04 noiseless XOR | Mean TAR undefined, logistic accuracy 0.5, MLP accuracy 1.0 |
| E06 | Frozen mean 0.4947917; learned mean 0.8242188, three seeds |
| E07 | Signal accuracy 1.0 raw, 0.453125 filtered; nuisance 1.0 in both cases |
| E08 | Rank 32 vs 64; scalar budget 32 vs 72; both probe accuracies 1.0 |
| Artifact audit | 14 corrected runs pass; 25 historical runs independently pass integrity checks |
| Regenerated outputs | 13 PNG/PDF figure sets, five CSV tables, LaTeX macros/tables and source-selection manifest |
| Checkpoint replay | 24 saved E06/E09 models reproduce recorded held-out metrics; regenerated held-out dataset hashes match |
| Paper rebuild | Tectonic 0.17.0 successfully rebuilt 13 pages; zero unresolved-reference markers or text overflows; minor bibliography spacing and nonfatal Fontconfig notices remain |
| Evidence archive | `reports/review_20261114/source_evidence.zip`, 168 files, CRC checked; checksum in adjacent `.sha256` file |

Zero outliers among 30 pointwise intervals is an observation, not a simultaneous-coverage guarantee or proof of a theorem. Model accuracy is not the Bayes oracle outside the controlled Gaussian or collision settings.

The paired E01 probe uses tokens rounded to ten decimal places to suppress floating-point collision residuals; the unrounded probe is also recorded and scores 0.5 in this cohort. Exact-arithmetic token indistinguishability and measured finite-precision probe behavior are separate claims.

## Reproduction

The review used a local `.venv` with access to the preinstalled scientific packages, Python 3.13.3, NumPy 2.2.5 and PyTorch 2.9.1+cpu. On Windows, activate `.venv\Scripts\Activate.ps1`; elsewhere activate the equivalent virtual environment. From the repository root:

```text
python -m pip install -e . --no-build-isolation --no-deps
python -m pytest -q
ruff check .
python -m patchloss.cli test-theory --config configs/smoke.yaml
python -m patchloss.cli run-suite --suite full --device cpu --output runs/review_20261114 --report-output reports/review_20261114
python scripts/generate_paper_results.py --input runs/review_20261114 --output paper
python scripts/verify_saved_models.py --input runs/review_20261114 --output reports/review_20261114/checkpoint_audit.json
python -m patchloss.cli audit --input runs/review_20261114 --output reports/review_20261114/results_audit.md
```

Those Python, test, lint, experiment, macro-generation and audit commands were executed successfully. Initial local installation of pytest/Ruff required network access; the experiments and tests themselves did not download data. A clean dependency installation was not tested; this environment reused existing scientific packages. See `EXECUTION_STATUS.md` for paper compilation and final validation status.

The paper was rebuilt using the official Tectonic 0.17.0 Windows release in `.tools/tectonic/`. Its first compilation downloaded TeX resources to `.tools/tectonic-cache/`; a final `--only-cached` rebuild succeeded offline. The new PDF contains the corrected residual, MC discrepancy and cohort references, with resolved citations. `paper_validation.json` records its SHA256 and structural checks. The package source bytes also match all 14 recorded source snapshots.

## Publication limits and next research

The core rank and Gaussian arguments are standard mathematics, correctly scoped in the manuscript. This review does not establish novelty. Before submission, conduct a full-text prior-art review, choose a substantive new question, add genuinely held-out nuisance/translation controls whose label semantics survive the transformation, expand seed coverage and uncertainty for learned models, and run fair compute/token-budget comparisons. A near-chance linear probe after clipping/filtering is not an information-loss certificate. MNIST training, natural-image transfer and GPU reproducibility remain untested.

Method references checked during review: [Stanford's equal-covariance Gaussian discriminant derivation](https://web.stanford.edu/class/stats202/notes/Classification/LDA.html), [SciPy's exact binomial confidence interval documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html). These support the statistical methods, not a novelty claim for this project.
