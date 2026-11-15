# Hypotheses and falsification register

The experiment mapping and criteria were written before running primary models (`experiment_protocol.md`). Results are linked to actual run records through the manifest. Supported means consistent in this **tested** setting, never a universal generalization.

| Hypothesis | Independent axis | Outcome | Confounders | Falsification criterion | Status |
|---|---|---|---|---|---|
| H1 lossless reshaping | p | exact reconstruction residual | dtype, dimension order | any unexplained mismatch | supported E00 and unit tests |
| H2 Gaussian alignment | alpha, sigma, rank | Bayes MC minus theory | numerical zero direction, Monte Carlo SE | repeat systematic deviation above 95% tolerance | supported E02; see raw per-cell table |
| H3 task dependence | fixed W, varied v | Bayes-risk gap | norm and sigma | inability to construct row/null risks | supported E03 |
| H4 learned recovery | frozen vs trainable W | held-out accuracy and alignment | training and validation | no reproducible difference over three seeds | supported on planted S2 in E06 |
| H5 low-pass tradeoff | label signal vs nuisance | held-out accuracy | smoothing changes rank, phase | absence of predicted two-sided effect | partial: signal hurt; nuisance remained equally easy, not improved |
| H6 overlap | stride and total tokens | rank, accuracy, cost | unequal scalar budget | no rank gain in chosen operator | rank gain; accuracy unchanged in E08 |
| H7 TAR limitations | mean-shift vs XOR | TAR and probe error | random-phase class means | fails to show XOR mean equality | supported limitation; weak TAR predictive generality |

**Negative results are not dropped:** in E09 the six-epoch tiny ViT and CNN did not reliably solve XOR, while a separate nonlinear feature probe solved it. This distinguishes optimization/capacity failure from information-theoretic failure. In E08 increased rank did not improve already saturated accuracy; no matched-budget overlap gain demonstrated. In E07 low-pass did not improve the already easy nuisance task.
