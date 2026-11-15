# Post-primary numerical and policy audits

These analyses were specified after inspecting primary results and are explicitly exploratory. Preserve the original primary results and never promote these audits to a prospective experiment.

## LogME numerical sensitivity

The original eleven-step baseline has many classwise fits that do not satisfy its convergence rule. On the same 1,000 diagnostic examples, profile the Gaussian evidence over the prior/noise precision ratio lambda: for fixed lambda, maximize noise precision analytically; evaluate a broad 61-point log grid spanning the Gram spectrum, include the original returned ratio and the null (infinite-lambda) model, and refine the best grid basins with a bounded one-dimensional optimizer. All choices use diagnostic features/labels only. This is a profile-search variant of the same established likelihood objective, not a new transferability measure or a certified global optimizer. Check the objective against direct integrated Gaussian likelihood. Compare outcomes on the fixed primary heads and preserve every baseline result.

## Simple-policy controls

Inspect **all fourteen** fixed tokenizer policies descriptively, not just the best one selected after seeing test data. The lowest observed fixed-policy regret is post-hoc and cannot establish a prospectively selected deployment policy.

Also evaluate a leave-one-dataset-out policy: for each budget, select candidate(s) using mean validation accuracy across five seeds of the other dataset only, then score those frozen candidates on the target dataset. This uses source validation labels and no target test labels. It is a limited two-direction transfer control between related datasets, not independent domain-generalization certification. Uniform tie expectation remains the policy.

Motivation: an apparent improvement over a weak or numerically limited diagnostic is insufficient evidence of a useful new method. These audits may weaken the apparent advantage of a retention score; such outcomes must remain in the paper.
