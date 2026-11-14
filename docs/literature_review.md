# Literature review and prior-art boundary

**Review scope (2026-11-07).** These are concise critical reviews based on the published abstracts and official publication records, with the questions tested in this repository distinguished from published results. We did **not** conduct a systematic full-text replication of six papers. Accordingly, fine experimental details not independently verified from full papers are flagged as not assessed. Six primary publication entries were checked against publisher/arXiv records. URLs are primary academic sources.

## Dosovitskiy et al. — *An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ICLR 2021)

The original ViT paper introduced image-to-patch embedding and Transformer-based image classification at scale. The paper's central point is competitive visual recognition through patch-sequence processing after substantial pretraining; it did not introduce a theorem that a reversible rearrangement of pixels destroys frequencies. The published architecture uses a learned linear map from flattened pixels to a token vector, making the **projection dimension** a distinct variable from the patch reshaping step. Source: https://arxiv.org/abs/2010.11929.

Its training regime and transfer setting are different from this repository's intentionally tiny synthetic study. Success on ImageNet-scale transfer cannot settle whether a particular **fixed, rank-deficient linear operator** is sufficient for an arbitrary labeled task, nor whether a small randomly initialized ViT is optimization-limited. We reproduce only its general linear tokenization and small encoder pattern, not its weights or benchmarks; no state-of-the-art claims follow.

## Wang et al. — *Scaling Laws in Patchification: An Image Is Worth 50,176 Tokens And More* (ICML 2025)

This is the closest and most important prior art, not a paper to ignore in novelty claims. It reports extensive smaller-patch scaling across visual architectures and tasks, including very long token sequences, and interprets the trend through compressive patchification. Their empirical observation concerns **complete trained architectures and budgets**, not the claim that lossless pixel rearrangement by itself maps distinct images to identical arrays. Source: https://proceedings.mlr.press/v267/wang25ed.html (PMLR 267, pp. 65278–65290).

Our narrower contribution is an intentionally elementary separation of reversible patch extraction, exact rank loss, task-signal alignment, and supervised adaptation under controlled counterexamples. That is complementary rather than a refutation of the reported scaling laws. We do not re-run their natural-image comparisons and cannot establish whether observed tiny-model trends transfer to large pretrained models. No general law of patch-size accuracy is claimed here.

## Nguyen et al. — *An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels* (ICLR 2025)

This work studies pixels as individual Transformer tokens in multiple vision settings. The publicly available manuscript explicitly says it does not introduce a new method; it shows the potential of standard Transformers at the pixel-token limit despite the impractical computation of very long sequences. The project therefore treats pixel tokenization and increasingly small patch sizes as established context, not a novel architectural contribution. Source: https://arxiv.org/abs/2406.09415.

The paper does not answer the exact finite-dimensional question of which *fixed* shared projection directions are annihilated and what the resulting binary Gaussian Bayes risk must be. We do not reproduce its supervised/unsupervised/generative training suites. The high compute costs of pixel-level attention motivate our rank and frozen-probe controls, which can be run on CPU without conflating computational budget with channel identifiability.

## Qian et al. — *Blending Anti-Aliasing into Vision Transformer* (NeurIPS 2021)

Qian and collaborators study visual aliasing artifacts and propose an Aliasing-Reduction Module, reporting performance and robustness improvements across vision transformer systems. This is explicit prior art for anti-aliasing interventions; we do not claim the broad insight that anti-aliasing can help as new. Source: https://proceedings.neurips.cc/paper/2021/hash/2b3bf3eee2475e03885a110e9acaab61-Abstract.html.

In our more constrained experiment the operator is an explicitly defined 5×5 binomial filter with circular boundaries, followed by patch projection. We compare a family where checkerboard structure is label-defining to one where it is nuisance. The resulting trade-off must not be conflated with the published ARM implementation, for which we have not performed a like-for-like replication. It would be wrong to say that any smoothing step universally retains task information.

## Xiao et al. — *Early Convolutions Help Transformers See Better* (NeurIPS 2021)

Xiao et al. report poor optimization properties of standard stride-p patchify stems relative to lighter early convolutional stems, with improved training stability under a broadly similar compute budget. Their argument is particularly relevant because an input operator may retain information but still be difficult for a specific architecture and optimizer to exploit. Source: https://proceedings.neurips.cc/paper/2021/hash/ff1418e8cc993fe8abcfe3ce2003e5c5-Abstract.html.

This repository separates a token-space Bayes oracle from learned-model outcomes. A weak trained model on a nominally sufficient representation is not proof that its stem irrevocably discarded the label. Our CNN comparator is not their matched-compute conv stem: different parameter counts and inductive biases are disclosed. A true replication would require fixed optimizer, schedule, compute and identical downstream ViT in a larger benchmark.

## Wu et al. — *CvT: Introducing Convolutions to Vision Transformers* (ICCV 2021)

CvT combines convolutional token embedding and convolutional Transformer components within a hierarchical architecture. It establishes that convolutional and overlapping stems have substantial prior use, and the authors report strong generalization and efficiency results on ImageNet-style tasks. Official conference record: https://openaccess.thecvf.com/content/ICCV2021/html/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.html (ICCV 2021, pp. 22–31).

We do not reconstruct CvT: our overlap experiment explicitly materializes a single linear overlapping-window operator on an 8×8 test image. We report actual rank and token budget instead of asserting that overlap always restores invertibility or improves accuracy. Increased scalar budget is a confounder: any overlap gain under that condition is a conditional engineering observation, not a controlled advantage of the architecture alone.

## Mathematical foundations

Bayes decision theory supplies the infimum risk over all measurable classifiers on a given representation. The data-processing inequality for the Markov chain Y→X→Z gives I(Y;Z)≤I(Y;X), but permits equality when the discarded variation is irrelevant to Y. See Cover and Thomas, *Elements of Information Theory*, 2nd edition (Wiley, 2006; https://www.wiley.com/en-in/elements-of-information-theory-2nd-edition-p-9780471241959), and standard multivariate Gaussian discriminant theory. SVD, rank-nullity, and orthogonal projection are elementary linear algebra, not original theorems here. The exact binary equal-prior Gaussian formula follows directly from this classical theory; its use as a diagnostic in the controlled tokenization setting is the pedagogical experiment, not a claim of a new law.

**Unassessed full-text details:** No comparative FLOPs measurement was independently replicated from the cited publications, no full code replication was done, and no exhaustive 2026 novelty search was conducted. `novelty_matrix.csv` marks independent novelty UNVERIFIED.
