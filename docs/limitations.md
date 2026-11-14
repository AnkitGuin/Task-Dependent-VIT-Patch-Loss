# Limitations

- Synthetic null-space tasks are deliberately constructed for a fixed projection; they do not estimate collision prevalence in natural images.
- The exact Gaussian Bayes formula assumes equal priors, isotropic shared noise and a fixed linear representation.
- Mean TAR measures retained class-mean energy. It is not mutual information and cannot characterize arbitrary nonlinear label relations.
- Tiny-model training outcomes mix representation, architecture, optimization, sample size and compute effects. Weak probe accuracy is not a Bayes-risk certificate.
- The Transformer cost proxy excludes elementwise operations, backward and optimizer work. CNN FLOPs and compute-matched architectural comparisons remain unmeasured.
- Overlap can change the scalar feature budget and rank simultaneously; higher rank alone does not establish a fair accuracy advantage.
- The periodic binomial low-pass control is not a general anti-aliasing architecture. Clipping noisy inputs can leave higher-order information uncharacterized by mean contrast or linear probes.
- Uniform random grating phase has the same population distribution after a fixed offset. The matched phase sweep is not a held-out translation-robustness experiment.
- Noiseless XOR repeats four support images across splits and tests representation/probe capacity rather than unseen-template generalization.
- Numerical rank uses a relative SVD tolerance. Exact collision statements concern exact arithmetic; measured floating-point residuals and conditioning must be considered separately.
- Optional MNIST training, large natural-image validation, GPU reproducibility, comprehensive seed uncertainty and systematic full-text novelty review remain open.
- The manuscript and generated results are intentionally absent from this code-only release. Reproduce the supplied experiments locally before making empirical claims.
