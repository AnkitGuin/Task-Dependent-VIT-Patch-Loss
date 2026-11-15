# Claim register (evidence classes)

| Claim | Class | Evidence | Qualification |
|---|---|---|---|
| Ordered, non-discarding patchify is an isometric permutation | ESTABLISHED / PROVED_HERE | Theorem A, exact round-trip tests | H/W divisible by p, ordered entries, exact arithmetic |
| Rank of shared patch tokenizer is N times rank of W | ESTABLISHED / PROVED_HERE | Proposition B, block matrix test | Full ordered sequence, linear map |
| A fixed rank-deficient W admits a constructed zero-separability two-point label task | PROVED_HERE | Proposition C, E01 | Worst case, not all natural tasks |
| Equal-prior isotropic Gaussian optimal risk is Phi(-||P_A v||/(2 sigma)) | ESTABLISHED / PROVED_HERE | Theorem D, MC validation E02 | Not a nonlinear or non-Gaussian bound |
| Mean TAR is a retained class-mean-energy diagnostic, not MI | ESTABLISHED / EMPIRICAL | E04 and XOR symmetry | Insufficient for nonlinear relational labels |
| Fixed projection can be outperformed by training W on planted signal | EMPIRICAL | E06 three seeds | Small architecture, particular noise and optimization |
| Pixel-token ViTs or conv stems are novel here | REJECTED | ICML 2025, ICLR 2025, ICCV/NeurIPS 2021 | Established prior art |
| Novelty of a general task-retention predictor | UNVERIFIED | `novelty_matrix.csv` | Not asserted |
| Natural-image transfer | UNVERIFIED | E10 NOT RUN | Optional online data |
| Multi-base S1 has zero raw-image Bayes risk | PROVED_HERE / EMPIRICAL | Fixed-hyperplane construction; raw sign-oracle test and corrected E01 | Corrected 2026-11-14; unrestricted old random bases did not establish this |
| E07 filtered clipped/noisy images contain no label information | UNVERIFIED | Only linear-probe degradation measured | Exact checkerboard annihilation alone does not establish distributional equality |
| E05 phase offsets establish held-out translation robustness | REJECTED | Generator inspection | Matched train/test offsets and uniform random phase preserve the population distribution |
| Noiseless XOR demonstrates unseen-template generalization | REJECTED | Four-point support repeated across splits | It is a representation/probe-capacity control |
| Covariance retention has the lowest primary mean selection regret | EMPIRICAL | 20 prospective units, 280 candidate heads on CIFAR-10/100 | Conditional on this bank/label budget; not best in every dataset/budget cell |
| A positive finite-sample mean-TAR ratio certifies label information | REJECTED | Isotropic null Beta calibration; independent sampling control | Under zero population contrast, the ratio's expected value is the rank fraction |
| Cross-fitted component energies yield an unbiased retention ratio | REJECTED | Independent cross-product reasoning | Components can be unbiased under stated independence; the ratio need not be |
| Real ViT-B/16 must have a rank bottleneck in its patch projection | REJECTED | Actual pretrained matrix audit | Observed full column rank; numerical conditioning and later processing are separate |
| Longer-budget tiny ViT recovers fresh-noise XOR under matched data/initialization | EMPIRICAL | Five paired 6/80-epoch-cap runs | This task and optimizer only; CNN outcomes remain variable |
| A newly proposed diagnostic universally beats simple defaults or established selectors | UNVERIFIED / NOT SUPPORTED | Fixed-policy and adaptive/evidence sensitivity audits | Best fixed policy is post-hoc; no unseen-domain guarantee or new-estimator novelty |
| Covariance retention remains lowest in the expanded-grid whole-bank comparison | EMPIRICAL | Separate validation-only grid, 280 new heads | Post-primary; close to profile LogME and diagnostic probe, with paired intervals containing zero |
| The leading whole-bank selector is also best when projection dimensions are matched | NOT SUPPORTED | Expanded within-patch-bank comparison | LogME leads that aggregate; smaller banks have different empirical oracles |
| Mean TAR choice uses labels effectively in this bank | NOT SUPPORTED BY NEGATIVE CONTROL | Five diagnostic-label shuffles per primary unit leave its choice unchanged | Bank-specific; the formula is not claimed label-independent in general |
