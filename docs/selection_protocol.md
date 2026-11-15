# Prospective tokenizer-selection study protocol

Written on 2026-11-14 before running the new experiments. The previous synthetic results are development evidence and will not be represented as prospectively registered. This local file is a timestamped design record, not an independently registered protocol.

## Question and positioning

Can a cheap task-dependent score select useful frozen, compressed linear patch tokenizers under an equal scalar-feature budget? Compare mean TAR with covariance-aware retention, cross-fitted contrast retention, reconstruction-energy retention, shrinkage H-score, LogME, and a small held-out linear probe. This is a controlled comparative study, not a new claim for Fisher discrimination, cross-validated distances, H-score or Bayesian evidence. Novelty remains contingent on the literature audit and actual findings.

Primary setting: ordered frozen linear tokens followed by a regularized linear classifier. No inference about end-to-end pretrained Transformer performance is justified. Standard ViT-B/16 can have an injective patch embedding; explicitly audit real pretrained weights rather than claiming that all ViTs have a patch-rank bottleneck.

## Data and prospective partitions

- Primary datasets: official CIFAR-10 and CIFAR-100, native 32x32 RGB; no resizing or augmentation in the selection study. They are two low-resolution datasets from a related source, not independent broad-domain coverage.
- Explicitly authorize dataset and public pretrained-weight downloads for this user-requested study; all files stay local and hashes are recorded. Tests remain offline.
- Per dataset/seed, draw stratified, non-overlapping official-training subsets: 1,000 unlabeled operator-fit images; 1,000 diagnostic images; 500 downstream validation images. A downstream probe may use the combined first 2,000 images for fitting. Diagnostic methods receive the same 1,000 labeled diagnostic images. Test uses 2,000 stratified official-test images.
- Seeds: 0, 1, 2, 3, 4. Diagnostic examples split into two stratified halves for cross-fitted means and the cheap-probe baseline. Class counts and exact official indices are saved. No diagnostic or hyperparameter selection may use official-test labels.
- Pixel normalization uses fixed ImageNet channel means/stds for every candidate, including non-pretrained ones. All reconstruction and retention scores refer to this same normalized coordinate system.

## Candidate bank and budgets

- Scalar budgets B=256 and 768 at fixed 32x32 resolution. Patch widths p=2,4,8,16 with N=1024/p^2 and d=B/N, all integral. Each full-row-rank tokenizer has global rank B, making rank a deliberately uninformative control within each budget.
- Per width: fixed seeded random orthonormal rows; low-frequency separable DCT rows; PCA rows fit only on the 1,000 unlabeled operator-fit images. PCA patch sampling is fixed in advance and capped at 8,000 patches.
- At p=16: an official ImageNet-pretrained DeiT-tiny patch matrix, truncated by SVD when d<192; compare its native singular scaling with its orthonormal row-space version. This is a patch-operator transfer experiment, not evaluation of the full pretrained DeiT model. External ImageNet pretraining is a separately disclosed data advantage.
- Candidate bank: 14 candidates per budget, 20 dataset/seed/budget units, 280 downstream candidate fits. Full-image raw features are a separate reference, not a budget-matched candidate.
- Dense projection forward proxy is 2*B*(3*p^2) FLOPs per image. Equal B does not equal equal tokenization cost or Transformer attention cost. Report observed transform/runtime and parameter count, plus a separate within-p comparison where dense projection dimensions/cost are matched. Never claim exact compute matching across p.

## Scores and downstream evaluation

- Rank: numerical global rank, with uniform tie expectation. Reconstruction: mean-centered training variance energy captured by the tokenizer row space, estimated on unlabeled operator-fit examples.
- Mean TAR: retained empirical between-class mean energy on the diagnostic set.
- Covariance TAR: ratio of projected to raw common-within-patch-whitened between-class energy. A shared within-class patch covariance uses fixed 0.1 isotropic shrinkage; this is a Gaussian-motivated approximation and omits cross-patch dependence.
- Cross-fitted TAR: replace squared diagnostic class-mean contrasts by cross-inner-products from the two independent diagnostic halves, with class-balanced weights. Do not clip negative values or values above one. A nonpositive denominator is undefined. Its component cross-energies have an unbiasedness argument for a fixed operator independent of diagnostic data; their ratio is not asserted unbiased.
- Shrinkage H-score uses diagnostic feature covariance and between-class means, with fixed 0.1 covariance shrinkage. LogME uses one-hot labels and the published evidence objective/fixed-point procedure, checked against independent direct evidence calculation. Cite their established origin.
- Cheap-probe score: fixed-alpha ridge classifier trained on one diagnostic half and evaluated on the other; no downstream validation or official-test labels.
- Downstream outcome: ridge multiclass least-squares classifier, centered features with a single RMS scale estimated on the combined 2,000 fitting images. This preserves orthogonal-basis invariance, unlike coordinatewise standardization. Choose alpha from {0.1,1,10,100,1000} by the 500-image validation accuracy; ties favor larger alpha. Save selected coefficients/checkpoints and complete validation curves.
- Freeze diagnostic scores, eligible maximizing candidates, downstream alphas and validation metrics in a pretest selection artifact before computing any official-test accuracies. Tied score maxima use uniform expected test accuracy as primary selection performance; also save a deterministic seeded realized selection.

## Endpoints, uncertainty and exclusions

Primary endpoints: test-accuracy selection regret (best measured candidate test accuracy minus score-selected accuracy), Spearman rank agreement within each unit, and diagnostic wall time. Constant-score rank correlations are undefined, not zero. Summaries weight datasets/budgets equally. Bootstrap paired seed blocks within each dataset, preserving budgets, for conditional uncertainty; with only two related datasets, do not claim a confidence interval for arbitrary new domains. Candidate rows and seed repetitions are not independent dataset replications.

Secondary endpoints: per-p compute-dimension-matched selection; selected-candidate identities; preprocessing and label-budget costs; raw-feature reference; native versus orthonormal pretrained row space. Preserve all unsuccessful conditions. Undefined scores must be counted, never silently replaced by an advantageous test-selected fallback. A conservative declared undefined-score policy uses uniform choice over the complete bank and reports the failure count.

There is no hypothesis that the proposed retention variants must win. If established baselines or the cheap validation probe perform better, report that result. No test-driven redesign or hyperparameter changes are allowed in the primary cohort. Any subsequent change requires a labeled exploratory cohort and a written deviation.

## Supporting controls and paper revision

1. Verify the empirical null mean-TAR distribution against its isotropic Gaussian Beta law and show that positive empirical TAR can occur with no label information. State the result as a diagnostic calibration, not inherently new distribution theory.
2. Verify anisotropic Gaussian projected Bayes risk independently; provide a same-rank/same-Euclidean-TAR counterexample with different noise-aligned risks.
3. Audit real pretrained patch-matrix ranks, singular spectra and dtype; include a full-rank ViT-B/16 reference if download/resources permit.
4. Train capacity/optimization controls on fresh-noise relational tasks with validation-selected stopping and several seeds, preserving the original short-training negative results as historical observations.
   Use matched epoch caps of 6 and 80 with identical data, initialization and optimizer for each of raw MLP, tiny ViT and CNN, across five seeds. This supporting comparison is specified before either cap is executed.
5. Add genuinely held-out, label-preserving translations of localized orientation/shape tasks, rather than reinterpreting matched uniform-phase sweeps as distribution shifts.
6. Reframe the manuscript around the selection study and its limits, move elementary proofs and implementation details to supporting material, regenerate every numerical table/figure from audited records, and compile the actual PDF. Keep all unpublished artifacts local.

## Resources and execution

CPU-only reduced study; no paid compute, cloud service or large-scale model training is authorized or needed. Cache data/weights under `data/`. Each unit records immutable configuration, exact splits/data hashes, operator matrices, coefficients, diagnostic scores, validation curves, predictions, source snapshot and checksums. Resume only completed hash-valid units matching configuration and source version. Save explicit failure logs for failed units. Learning/selection scripts expose smoke and full settings; smoke results cannot replace primary results.

Preparation note: the original Toronto archive transfers were too slow, so byte-identical CIFAR archives are downloaded from a mirror using resumable byte ranges, with the official publisher MD5 checksums enforced. DCT channel components also use a fixed orthonormal color DCT so the first low-frequency row mixes RGB equally, rather than selecting only red. These preparation choices were recorded before the primary analysis.
