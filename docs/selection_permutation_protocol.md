# Exploratory diagnostic label-permutation control

Specified after inspecting primary outcomes and the all-fixed-policy audit. This is post-primary and must not be presented as prospective confirmation.

For each of the twenty primary units, keep operators, diagnostic images, downstream heads and true test outcomes fixed. Shuffle only diagnostic labels, preserving their class counts, using five fixed independently derived permutation seeds. Recompute mean retention and covariance retention, freeze their maximizing candidate sets, then evaluate those choices against the unchanged primary target accuracies. Preserve both regret and agreement with the corresponding true-label choice.

This negative control asks whether the apparent practical success of these diagnostics depends on correct diagnostic labels, or whether the bank and covariance/geometry preferences already determine the selected operator. It does not prove that a formula ignores labels, establish a universal null distribution on natural images, or supply a formal permutation significance test. No score is redesigned or tuned on these outcomes.
