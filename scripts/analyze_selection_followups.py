"""Audit expanded-head, permutation and diagnostic-probe-grid follow-up evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from patchloss.reporting.records import read_record
from patchloss.selection.statistics import evaluate_selection, block_bootstrap


def verify_hashes(directory, key="artifact_sha256"):
    md = json.loads((directory / "metadata.json").read_text())
    if md["status"] != "completed":
        raise ValueError("Incomplete follow-up cohort")
    for name, expected in md[key].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Follow-up hash mismatch: {directory / name}")
    return md


def analyze(config, output, paper):
    expanded = Path("runs/selection_regularization")
    probe = Path("runs/selection_probe_grid")
    permutation = Path("runs/selection_permutation")
    expanded_meta = verify_hashes(expanded, "root_artifact_sha256")
    probe_meta = verify_hashes(probe)
    permutation_meta = verify_hashes(permutation)
    sensitivity = Path("runs/selection_sensitivity")
    profile = Path("runs/selection_logme_profile")
    verify_hashes(sensitivity)
    verify_hashes(profile)
    primary_rows = pd.read_csv(output / "selection_by_unit.csv")
    expanded_rows = pd.read_csv(expanded / "selection_by_unit.csv")
    boundary_records, probe_records, within_records, perm_records, raw_records = [], [], [], [], []
    count = 0
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            for budget in config["budgets"]:
                name = f"E11-{dataset}-b{budget}-s{seed}"
                original = Path(config["output"]) / name
                directory = expanded / name
                md = verify_hashes(directory)
                primary_md, primary = read_record(original)
                old_pretest = json.loads((original / "pretest_selection.json").read_text())
                new_pretest = json.loads((directory / "pretest_selection.json").read_text())
                metrics = json.loads((directory / "metrics.json").read_text())
                assert md["primary_source_sha256"] == primary_md["source_sha256"]
                assert (
                    md["primary_metrics_sha256"]
                    == hashlib.sha256((original / "metrics.json").read_bytes()).hexdigest()
                )
                assert new_pretest["selection"] == old_pretest["selection"]
                assert all("test_accuracy" not in row for row in new_pretest["rows"])
                assert md["datasets"] == json.loads((original / "datasets.json").read_text())
                meta = {"dataset": dataset, "seed": seed, "budget": budget}
                lookup = {r["candidate"]: r for r in metrics["rows"]}
                with np.load(directory / "predictions.npz") as saved:
                    for row in new_pretest["rows"]:
                        actual = lookup[row["candidate"]]
                        assert actual["scores"] == row["scores"]
                        assert (
                            actual["scores"]
                            == next(r for r in primary["rows"] if r["candidate"] == row["candidate"])[
                                "scores"
                            ]
                        )
                        assert (
                            float(np.mean(saved[row["candidate"]] == saved["labels"]))
                            == actual["test_accuracy"]
                        )
                        boundary_records.append(
                            {
                                **meta,
                                "candidate": row["candidate"],
                                "selected_alpha": row["selected_alpha"],
                                "upper_grid_plateau": row["upper_grid_accuracy_plateau"],
                            }
                        )
                raw_records.append(
                    {
                        **meta,
                        "primary_accuracy": primary["metrics"]["raw_reference_accuracy"],
                        "expanded_accuracy": md["raw_reference_accuracy"],
                    }
                )
                probe_pretest = json.loads((probe / f"{name}_pretest.json").read_text())
                assert all("test_accuracy" not in r for r in probe_pretest["rows"])
                for grid, outcomes in [("primary", primary["rows"]), ("expanded", metrics["rows"])]:
                    outcome_map = {r["candidate"]: r["test_accuracy"] for r in outcomes}
                    scored = [
                        {**r, "test_accuracy": outcome_map[r["candidate"]]} for r in probe_pretest["rows"]
                    ]
                    result = evaluate_selection(scored, {"cheap_probe": probe_pretest["selection"]})[0]
                    result["metric"] = "probe_grid"
                    probe_records.append({**meta, "head_grid": grid, **result})
                adaptive = json.loads(
                    (sensitivity / f"{dataset}-b{budget}-s{seed}-n1000_pretest.json").read_text()
                )
                evidence = json.loads((profile / f"{name}_pretest.json").read_text())
                extensions = [
                    (
                        "adaptive_hscore",
                        "hscore",
                        [
                            {"candidate": r["candidate"], "scores": {"hscore": r["adaptive_hscore"]}}
                            for r in adaptive["scores"]
                        ],
                        adaptive["adaptive_selection"],
                    ),
                    ("logme_profile", "logme", evidence["scores"], evidence["selection"]),
                ]
                for metric, source_metric, scored_rows, choice in extensions:
                    for grid, outcomes in [("primary", primary["rows"]), ("expanded", metrics["rows"])]:
                        outcome_map = {r["candidate"]: r["test_accuracy"] for r in outcomes}
                        scored = [{**r, "test_accuracy": outcome_map[r["candidate"]]} for r in scored_rows]
                        result = evaluate_selection(scored, {source_metric: choice})[0]
                        result["metric"] = metric
                        probe_records.append({**meta, "head_grid": grid, **result})
                for patch, choices in old_pretest["within_patch_selection"].items():
                    members = [
                        r
                        for r in metrics["rows"]
                        if next(p for p in primary["rows"] if p["candidate"] == r["candidate"])["patch_size"]
                        == int(patch)
                    ]
                    within_records.extend(
                        {**meta, "patch_size": int(patch), **r} for r in evaluate_selection(members, choices)
                    )
                for repetition in json.loads((permutation / f"{name}_pretest.json").read_text()):
                    scored = [
                        {**r, "test_accuracy": lookup[r["candidate"]]["test_accuracy"]}
                        for r in repetition["scored"]
                    ]
                    perm_records.extend(
                        {**meta, "repetition": repetition["repetition"], **r}
                        for r in evaluate_selection(scored, repetition["selection"])
                    )
                count += 1
    assert count == 20
    boundary = pd.DataFrame(boundary_records)
    probe_rows = pd.DataFrame(probe_records)
    combined = pd.concat(
        [primary_rows.assign(head_grid="primary"), expanded_rows.assign(head_grid="expanded"), probe_rows],
        ignore_index=True,
    )
    summary = combined.groupby(["head_grid", "metric"], as_index=False).agg(
        mean_regret=("regret", "mean"), mean_selected_accuracy=("selected_accuracy", "mean")
    )
    pivot = summary.pivot(index="metric", columns="head_grid", values="mean_regret").sort_values("expanded")
    best = summary[summary.head_grid == "expanded"].sort_values("mean_regret").iloc[0]
    intervals = []
    for grid in ["primary", "expanded"]:
        part = combined[combined.head_grid == grid]
        names, samples = block_bootstrap(part, "regret", draws=config.get("bootstrap_draws", 2000))
        for i, metric in enumerate(names):
            intervals.append(
                {
                    "head_grid": grid,
                    "metric": metric,
                    "lower": np.quantile(samples[:, i], 0.025),
                    "upper": np.quantile(samples[:, i], 0.975),
                }
            )
    # Keep paired differences paired across the same resampled seed blocks.
    names, samples = block_bootstrap(
        combined[combined.head_grid == "expanded"], "regret", draws=2000, seed=427
    )
    differences = []
    for comparator in ["mean_tar", "cheap_probe", "logme", "logme_profile", "probe_grid"]:
        delta = samples[:, names.index("covariance_tar")] - samples[:, names.index(comparator)]
        differences.append(
            {
                "first": "covariance_tar",
                "second": comparator,
                "difference": float(
                    pivot.loc["covariance_tar", "expanded"] - pivot.loc[comparator, "expanded"]
                ),
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
            }
        )
    pd.DataFrame(differences).to_csv(output / "expanded_paired_regret_differences.csv", index=False)
    fixed = pd.read_csv(expanded / "candidates.csv")
    fixed["regret"] = (
        fixed.groupby(["dataset", "seed", "budget"]).test_accuracy.transform("max") - fixed.test_accuracy
    )
    fixed.to_csv(output / "expanded_fixed_policy_by_unit.csv", index=False)
    fixed_summary = fixed.groupby("candidate", as_index=False).regret.mean().sort_values("regret")
    fixed_summary.to_csv(output / "expanded_fixed_policy_summary.csv", index=False)
    fixed_best = fixed_summary.iloc[0]
    permutation_rows = pd.read_csv(permutation / "summary.csv")
    expanded_rows.to_csv(output / "expanded_grid_selection.csv", index=False)
    summary.to_csv(output / "head_grid_summary.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "expanded_grid_intervals.csv", index=False)
    probe_rows[probe_rows.metric == "probe_grid"].to_csv(output / "probe_grid_outcomes.csv", index=False)
    probe_rows.to_csv(output / "exploratory_head_grid_outcomes.csv", index=False)
    boundary.to_csv(output / "expanded_grid_boundaries.csv", index=False)
    pd.DataFrame(raw_records).to_csv(output / "raw_reference_grid_comparison.csv", index=False)
    pd.DataFrame(within_records).to_csv(output / "expanded_within_patch_selection.csv", index=False)
    pd.DataFrame(perm_records).to_csv(output / "permutation_expanded_outcomes.csv", index=False)
    permutation_rows.to_csv(output / "permutation_summary.csv", index=False)
    labels = {
        "covariance_tar": "Covariance retention",
        "cheap_probe": "Diagnostic probe (fixed penalty)",
        "crossfit_tar": "Cross-fitted retention",
        "mean_tar": "Mean retention",
        "reconstruction": "Reconstruction energy",
        "logme": "LogME (11 steps)",
        "logme_profile": "LogME (profile; exploratory)",
        "hscore": "H-score (fixed shrinkage)",
        "adaptive_hscore": "H-score (adaptive; exploratory)",
        "probe_grid": "Diagnostic probe (grid; exploratory)",
        "rank": "Rank (uniform ties)",
    }
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(pivot))
    ax.barh(x - 0.18, 100 * pivot.primary, height=0.35, label="Original head grid")
    ax.barh(x + 0.18, 100 * pivot.expanded, height=0.35, label="Expanded head grid")
    ax.set_yticks(x, [labels[m].replace("; exploratory", "") for m in pivot.index])
    ax.set_xlabel("Mean selection regret (accuracy percentage points)")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        fig.savefig(output / "figures" / f"head_grid_sensitivity.{suffix}", dpi=180)
    plt.close(fig)
    at_max = int((boundary.selected_alpha == 1e8).sum())
    plateau = int(boundary.upper_grid_plateau.sum())
    perm_mean = permutation_rows.set_index("metric")
    within_mean = pd.DataFrame(within_records).groupby("metric").regret.mean()
    tex = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Post-primary validation-grid sensitivity (accuracy percentage points). Scores and operator maps are fixed; only downstream penalties are reselected on validation data. Regret uses each head grid's own empirical bank oracle. Diagnostic probe grid, adaptive H-score and profile LogME are exploratory additions.}\label{tab:grid}",
        r"\begin{tabular}{lrr}\toprule Selector & Original-grid regret & Expanded-grid regret\\\midrule",
    ]
    for metric, row in pivot.iterrows():
        tex.append(labels[metric] + f" & {100 * row.primary:.4f} & {100 * row.expanded:.4f}" + r"\\")
    tex.append(r"\bottomrule\end{tabular}\end{table}")
    text = (
        r"The original validation paths improve from penalty 100 to 1,000 in 264 of 280 cases, motivating a separate expanded-grid follow-up rather than a robustness claim from the initial grid. Operators, scores and score-based choices remain fixed; penalties are chosen using validation data only before new test outcomes. The expanded grid reaches $10^8$. "
        + f"{at_max} of 280 heads select that upper value, while {plateau} have identical validation accuracy over the largest three penalties; a finite grid still does not prove global optimality. "
        + r"Table~\ref{tab:grid} reports both outcome definitions, so any changed ordering remains visible. The additional diagnostic-probe grid jointly selects candidate and penalty from the same diagnostic label budget; it does not use official-test labels."
        + f" The lowest observed expanded-grid regret remains covariance retention ({100 * pivot.loc['covariance_tar', 'expanded']:.3f} points), closely followed by the fixed-penalty diagnostic probe ({100 * pivot.loc['cheap_probe', 'expanded']:.3f}). The paired conditional bootstrap interval for their regret difference includes zero; we do not assert statistical superiority."
        + f" Profile-evidence LogME also improves to {100 * pivot.loc['logme_profile', 'expanded']:.4f} points of regret, compared with {100 * pivot.loc['logme_profile', 'primary']:.4f} under the original heads. This stronger evidence baseline further narrows the observed difference from covariance retention."
        + f" A hindsight audit of all fixed candidates gives {fixed_best.candidate.replace('_', ' ')} a mean expanded-grid regret of {100 * fixed_best.regret:.3f} points; this is still not a prospectively chosen rule."
        + "\n\n"
        + r"Matching patch size as well as feature budget changes the comparison: expanded-grid within-patch-bank regrets average "
        + f"{100 * within_mean['logme']:.3f} for LogME, {100 * within_mean['covariance_tar']:.3f} for covariance retention, and {100 * within_mean['mean_tar']:.3f} for mean retention. "
        + r"These smaller banks have their own oracles and cannot be compared numerically with global-bank regret as though they were the same task. The change in ordering further limits any universal winner claim."
        + "\n\n"
        + r"In the diagnostic-label shuffle control, mean retention selects the same candidate in "
        + f"{100 * perm_mean.loc['mean_tar', 'same_choice_fraction']:.0f}"
        + r"\% of the 100 unit/permutation cases. Under the original heads, shuffled-label covariance retention has "
        + f"{100 * perm_mean.loc['covariance_tar', 'mean_regret']:.2f}"
        + r" points of mean regret, with its candidate choice changing in all cases. This is a bank-specific negative control: it does not imply that the mean-retention formula is label-independent in general or constitute a formal permutation significance test."
        + "\n"
    )
    (paper / "selection_followups.tex").write_text(text + "\n".join(tex) + "\n", encoding="utf-8")
    macros = {
        "ExpandedBestMetric": {"covariance_tar": "covariance retention"}.get(
            best.metric, best.metric.replace("_", " ")
        ),
        "ExpandedBestRegret": f"{100 * best.mean_regret:.3f}",
        "ExpandedBoundaryCount": str(at_max),
        "ExpandedPlateauCount": str(plateau),
        "ExpandedFixedRegret": f"{100 * fixed_best.regret:.3f}",
    }
    (paper / "selection_followup_results.tex").write_text(
        "% AUTOGENERATED follow-up values.\n"
        + "\n".join(chr(92) + "newcommand{" + chr(92) + k + "}{" + v + "}" for k, v in macros.items())
        + "\n"
    )
    result = {
        "expanded_best_metric": best.metric,
        "expanded_best_regret": float(best.mean_regret),
        "upper_boundary_count": at_max,
        "upper_plateau_count": plateau,
        "expanded_metadata": expanded_meta,
        "probe_metadata": probe_meta,
        "permutation_metadata": permutation_meta,
    }
    (output / "followup_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(
        f"AUDITED expanded grid: {count} units, 280 heads; fixed scores and expanded-grid predictions checked"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_study.yaml")
    parser.add_argument("--output", type=Path, default=Path("reports/selection"))
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    args = parser.parse_args()
    analyze(yaml.safe_load(Path(args.config).read_text()), args.output, args.paper)
