# Scientific code-review corrections

The November 2026 review corrected the following implementation and interpretation issues. Regression coverage is included in `tests/test_scientific_regressions.py`.

- Constrain random collision bases perpendicular to the planted null direction so the raw-image sign rule is a valid zero-risk oracle in exact arithmetic.
- Independently implement the Gaussian Monte Carlo oracle in observable token coordinates, instead of reusing the analytic row projector on raw images.
- Use IID equal-prior Monte Carlo labels and exact pointwise binomial confidence intervals. Compute coverage flags from the actual saved interval endpoints.
- Remove absolute numerical floors that changed rank or task-alignment interpretation under uniform scaling. Support valid Gaussian alignment endpoints.
- Partition MNIST training/validation indices disjointly and record their source IDs.
- Render XOR markers before applying periodic translations so wrapped slices cannot erase them.
- Honor rectangular overlap grids, reject dropped borders, and subtract affine offsets when materializing reference matrices.
- Validate configurations and fail explicitly on nonfinite data, losses or metrics.
- Record a depth-aware Transformer forward-cost proxy, total/trainable parameters and optimizer steps; do not present this proxy as full hardware FLOPs.
- Verify artifacts before aggregation and reject incompatible configuration/source mixtures in figures.
- Save dataset/source fingerprints, selected checkpoints and replayable metadata.

Interpretation limits also matter: exact annihilation of a periodic checkerboard by a linear filter does not establish zero label information after noisy nonlinear clipping. Matched offsets of uniformly random grating phase do not create a held-out translation shift. Noiseless XOR is a finite-support sufficiency control, not an unseen-template benchmark. Finite-precision collision residuals are distinct from exact-arithmetic equalities.

The manuscript and measured experiment artifacts are withheld from this code release. These corrections and tests do not establish novelty or publication readiness.
