# Claim register for the final journal revision

| Claim | Evidence status | Limit |
|---|---|---|
| Frozen linear patch banks were evaluated on three domains with full official test sets | MEASURED, 420 heads / 30 units | Three domains, not thirty independent tasks |
| Every final training partition is exact-content-disjoint from the official test set and other roles | VERIFIED by saved content hashes and reconstruction | Near duplicates and semantic dependence remain |
| Properly refined evidence and adaptive shrinkage are main baselines | IMPLEMENTED / independently numerically checked | Independent preprocessing/implementation variant, not bitwise author code |
| Validation grid extends from 10^-2 to 10^10 uniformly for all heads | VERIFIED; five upper-end selections, zero final-step strict improvements | Finite validation grid, ties favor the larger value |
| Covariance retention has lowest pooled observed regret among locked main rules | MEASURED: 0.221 pp | Pool is conditional on this bank and three domains; sensitivity setting .01 was not a main rule |
| Covariance retention exceeds profile LogME on pooled selected accuracy | MEASURED: +0.762 pp, joint interval +0.406 to +1.196 | Conditional bootstrap without multiplicity/domain-generalization claims |
| Label-aware selection is generally necessary | NOT ESTABLISHED | Overall covariance-minus-fixed gain is small with an interval spanning zero |
| Covariance uniquely enables the observed SVHN gain | REJECTED | It selects the same p16_pca as unlabeled reconstruction and mean retention in every SVHN unit |
| Fixed DCT evaluation is prospective on all domains | REJECTED | Only SVHN is independent of CIFAR policy development |
| The tested bank yields a useful SVHN classifier | NOT ESTABLISHED | Even test-oracle ordinary accuracy is below empirical test majority frequency; balanced accuracy is weak too |
| Shrinkage sensitivity chose the best setting in advance | REJECTED | .1 remains the locked default; .01/.5 are disclosed sensitivity endpoints |
| A bootstrap over repeated seeds establishes generalization to unseen domains | REJECTED | Intervals are conditional and preserve common test images across seeds |
| Scores, heads and predictions are reproducible locally | VERIFIED by 420 independent NumPy/direct-solve replays | Clean installation / another hardware platform not independently validated |
| Experimental evidence is publicly archived | PENDING | Local verified ZIPs are not a public repository or issued DOI |
| A specific journal will accept this manuscript | UNVERIFIED / NOT CLAIMED | Editorial scope, novelty judgment and peer review remain external decisions |

Source of all numbers: `reports/selection_revision/`, generated from immutable `runs/selection_revision_final/`. Resources are reported only from final isolated benchmark records; overlapping warm-ups are excluded. Original Gaussian/rank statements remain established theory and are not promoted as new results.
