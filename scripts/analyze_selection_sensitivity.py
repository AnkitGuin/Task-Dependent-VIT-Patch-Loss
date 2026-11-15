"""Audit and report exploratory label-budget, numerical, fixed-policy and dedup checks."""

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
from patchloss.selection.data import load_cifar
from patchloss.selection.statistics import evaluate_selection, block_bootstrap, freeze_selection


def audit_cohort(root):
    md = json.loads((root / "metadata.json").read_text())
    if md["status"] != "completed":
        raise ValueError(f"Incomplete cohort: {root}")
    for name, expected in md["artifact_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {root / name}")
    return md


def analyze(config, output, paper):
    sensitivity = Path("runs/selection_sensitivity")
    profile = Path("runs/selection_logme_profile")
    sensitivity_meta = audit_cohort(sensitivity)
    profile_meta = audit_cohort(profile)
    sensitivity_rows = pd.read_csv(sensitivity / "selection_by_unit.csv")
    profile_rows = pd.read_csv(profile / "selection_by_unit.csv")
    primary = pd.read_csv(output / "selection_by_unit.csv")
    cells = pd.read_csv(output / "candidates.csv")
    for metric in primary.metric.unique():
        old = primary[primary.metric == metric].sort_values(["dataset", "seed", "budget"]).regret.to_numpy()
        new = (
            sensitivity_rows[
                (sensitivity_rows.metric == metric) & (sensitivity_rows.diagnostic_samples == 1000)
            ]
            .sort_values(["dataset", "seed", "budget"])
            .regret.to_numpy()
        )
        np.testing.assert_allclose(old, new, atol=1e-12)
    sensitivity_summary = sensitivity_rows.groupby(["diagnostic_samples", "metric"], as_index=False).agg(
        mean_regret=("regret", "mean"),
        mean_selected_accuracy=("selected_accuracy", "mean"),
        mean_spearman=("spearman", "mean"),
        undefined_units=("undefined_fallback", "sum"),
    )
    sensitivity_summary.to_csv(output / "label_budget_summary.csv", index=False)
    profile_rows.to_csv(output / "profile_logme_selection.csv", index=False)
    all_full = pd.concat(
        [
            primary,
            sensitivity_rows[
                (sensitivity_rows.diagnostic_samples == 1000) & (sensitivity_rows.metric == "adaptive_hscore")
            ],
            profile_rows,
        ],
        ignore_index=True,
    )
    metrics, bootstrap = block_bootstrap(all_full, "regret", 2000, seed=394)
    extended = []
    for j, name in enumerate(metrics):
        part = all_full[all_full.metric == name]
        extended.append(
            {
                "metric": name,
                "mean_regret": part.regret.mean(),
                "ci_low": float(np.quantile(bootstrap[:, j], 0.025)),
                "ci_high": float(np.quantile(bootstrap[:, j], 0.975)),
                "mean_spearman": part.spearman.mean(),
            }
        )
    pd.DataFrame(extended).sort_values("mean_regret").to_csv(
        output / "exploratory_extended_comparison.csv", index=False
    )
    # All fixed policies are shown, so the best post-hoc one is not hidden as a
    # prospectively chosen baseline. No target outcome chooses the LODO policy.
    cells["oracle_accuracy"] = cells.groupby(["dataset", "seed", "budget"]).test_accuracy.transform("max")
    cells["regret"] = cells.oracle_accuracy - cells.test_accuracy
    fixed = cells.groupby("candidate", as_index=False).agg(mean_regret=("regret", "mean"))
    fixed.sort_values("mean_regret").to_csv(output / "all_fixed_policies_posthoc.csv", index=False)
    validations = []
    bank_rows = []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            for budget in config["budgets"]:
                directory = Path(config["output"]) / f"E11-{dataset}-b{budget}-s{seed}"
                _, record = read_record(directory)
                for bank_name in ["local_only", "orthonormal_only"]:
                    subset = [
                        r
                        for r in record["rows"]
                        if (
                            not r["family"].startswith("deit")
                            if bank_name == "local_only"
                            else r["family"] != "deit_native_svd"
                        )
                    ]
                    outcomes = evaluate_selection(subset, freeze_selection(subset))
                    bank_rows.extend(
                        {"dataset": dataset, "seed": seed, "budget": budget, "bank": bank_name, **r}
                        for r in outcomes
                    )
                history = pd.read_csv(directory / "history.csv")
                for candidate, part in history.groupby("candidate"):
                    validations.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "budget": budget,
                            "candidate": candidate,
                            "validation_accuracy": part.validation_accuracy.max(),
                        }
                    )
    validations = pd.DataFrame(validations)
    pd.DataFrame(bank_rows).to_csv(output / "restricted_bank_selection.csv", index=False)
    pd.DataFrame(bank_rows).groupby(["bank", "metric"], as_index=False).regret.mean().to_csv(
        output / "restricted_bank_summary.csv", index=False
    )
    policies = []
    for target in config["datasets"]:
        source = next(name for name in config["datasets"] if name != target)
        for budget in config["budgets"]:
            values = (
                validations[(validations.dataset == source) & (validations.budget == budget)]
                .groupby("candidate")
                .validation_accuracy.mean()
            )
            eligible = list(values[np.isclose(values, values.max(), rtol=1e-8, atol=1e-12)].index)
            for seed in config["seeds"]:
                target_cells = cells[
                    (cells.dataset == target) & (cells.budget == budget) & (cells.seed == seed)
                ]
                chosen = target_cells[target_cells.candidate.isin(eligible)].test_accuracy.mean()
                policies.append(
                    {
                        "dataset": target,
                        "source_dataset": source,
                        "seed": seed,
                        "budget": budget,
                        "eligible_candidates": eligible,
                        "selected_accuracy": chosen,
                        "regret": target_cells.test_accuracy.max() - chosen,
                    }
                )
    pd.DataFrame(policies).to_csv(output / "leave_one_dataset_out_policy.csv", index=False)
    # Pixel-hash exclusion depends only on images, preserving the primary decisions.
    dedup = []
    exclusions = []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            data, _ = load_cifar(dataset, seed, config)
            train_hashes = {
                hashlib.sha256(image.tobytes()).hexdigest()
                for key, (images, _) in data.items()
                if key != "test"
                for image in images
            }
            keep = np.array(
                [hashlib.sha256(image.tobytes()).hexdigest() not in train_hashes for image in data["test"][0]]
            )
            exclusions.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "test_examples_removed": int(np.sum(~keep)),
                    "remaining_test_examples": int(np.sum(keep)),
                }
            )
            for budget in config["budgets"]:
                directory = Path(config["output"]) / f"E11-{dataset}-b{budget}-s{seed}"
                _, data_record = read_record(directory)
                pretest = json.loads((directory / "pretest_selection.json").read_text())
                with np.load(directory / "predictions.npz") as predictions:
                    rows = []
                    for row in data_record["rows"]:
                        rows.append(
                            {
                                **row,
                                "test_accuracy": float(
                                    np.mean(
                                        predictions[row["candidate"]][keep] == predictions["labels"][keep]
                                    )
                                ),
                            }
                        )
                    outcomes = evaluate_selection(rows, pretest["selection"])
                    dedup.extend({"dataset": dataset, "seed": seed, "budget": budget, **r} for r in outcomes)
    pd.DataFrame(dedup).to_csv(output / "deduplicated_test_selection.csv", index=False)
    pd.DataFrame(exclusions).to_csv(output / "deduplicated_test_exclusions.csv", index=False)
    # Figure: all methods, not only whichever improves the claim.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for metric, part in sensitivity_summary.groupby("metric"):
        ax.plot(part.diagnostic_samples, 100 * part.mean_regret, "o-", label=metric.replace("_", " "))
    ax.set_xlabel("Diagnostic labels (fixed downstream fitting/validation data)")
    ax.set_ylabel("Mean selection regret (accuracy percentage points)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output / "figures/label_budget_sensitivity.pdf")
    plt.savefig(output / "figures/label_budget_sensitivity.png", dpi=180)
    plt.close()
    original = float(primary[primary.metric == "logme"].regret.mean())
    profiled = float(profile_rows.regret.mean())
    adaptive = float(
        sensitivity_rows[
            (sensitivity_rows.metric == "adaptive_hscore") & (sensitivity_rows.diagnostic_samples == 1000)
        ].regret.mean()
    )
    best_fixed = fixed.sort_values("mean_regret").iloc[0]
    lodo = float(pd.DataFrame(policies).regret.mean())
    text = (
        r"At 1,000 diagnostic labels, the exploratory adaptive shrinkage H-score has mean regret "
        + f"{100 * adaptive:.2f}"
        + r" percentage points. The profiled-evidence LogME variant has "
        + f"{100 * profiled:.2f}"
        + r" points, compared with "
        + f"{100 * original:.2f}"
        + r" for the primary short iteration cap. These variants use diagnostic data only, but their evaluation is post-primary and does not establish a prospectively selected new method."
        + "\n\n"
        + r"An audit of all fourteen fixed policies finds a lowest observed post-hoc regret of "
        + f"{100 * best_fixed.mean_regret:.2f}"
        + r" points for "
        + r"\texttt{"
        + best_fixed.candidate.replace("_", r"\_")
        + r"}. "
        + r"This fixed policy is identified after observing test outcomes and is not a validated deployment choice. The separate leave-one-dataset-out policy, selected using only the other dataset\textquotesingle s validation accuracies, has mean regret "
        + f"{100 * lodo:.2f}"
        + r" points. These controls limit claims that a more elaborate score is necessary."
        + "\n\n"
        + r"The content audit finds one exact image shared by operator/diagnostic subsets and one shared by diagnostic/test subsets, in separate CIFAR-100 seed draws. Official indices remain disjoint. Primary partitions are retained, while an exploratory evaluation removes test images matching any fitting/diagnostic/validation image by exact pixel hash. The exclusion count and resulting selection outcomes are reported with the artifacts; near-duplicate independence is not certified."
        + "\n"
    )
    (paper / "selection_sensitivity.tex").write_text(text, encoding="utf-8")
    macro_values = {
        "BestFixedRegret": f"{100 * best_fixed.mean_regret:.2f}",
        "ProfileLogmeRegret": f"{100 * profiled:.2f}",
        "AdaptiveHscoreRegret": f"{100 * adaptive:.2f}",
        "LodoRegret": f"{100 * lodo:.2f}",
        "LogmeNonconverged": str(int(cells.logme_nonconverged_classes.sum())),
    }
    (paper / "selection_secondary_results.tex").write_text(
        "% AUTOGENERATED exploratory values.\n"
        + "\n".join(
            chr(92) + "newcommand{" + chr(92) + key + "}{" + value + "}"
            for key, value in macro_values.items()
        )
        + "\n"
    )
    table = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Exploratory diagnostic-label sensitivity. Values are mean selection regret in accuracy percentage points, conditional on the unchanged primary bank and downstream heads. Adaptive H-score is an additional established baseline; this analysis was not in the primary protocol.}\label{tab:labels}",
        r"\begin{tabular}{lrrr}\toprule Score & 200 labels & 400 labels & 1,000 labels\\\midrule",
    ]
    for metric, part in sensitivity_summary.groupby("metric"):
        values = part.set_index("diagnostic_samples").mean_regret
        label = metric.replace("_", " ")
        table.append(label + " & " + " & ".join(f"{100 * values[n]:.2f}" for n in [200, 400, 1000]) + r"\\")
    table.append(r"profiled LogME & -- & -- & " + f"{100 * profiled:.2f}" + r"\\")
    table.append(r"\bottomrule\end{tabular}\end{table}")
    with (paper / "selection_sensitivity.tex").open("a", encoding="utf-8") as f:
        f.write("\n".join(table) + "\n")
    summary = {
        "adaptive_hscore_regret": adaptive,
        "profile_logme_regret": profiled,
        "primary_logme_regret": original,
        "best_fixed_policy_posthoc": best_fixed.candidate,
        "best_fixed_regret_posthoc": float(best_fixed.mean_regret),
        "lodo_policy_regret": lodo,
        "dedup_exclusions": exclusions,
        "sensitivity_metadata": sensitivity_meta,
        "profile_metadata": profile_meta,
        "interpretation": "All sensitivity and policy comparisons are exploratory; no unseen-domain guarantee",
    }
    (output / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"sensitivity_metadata", "profile_metadata", "dedup_exclusions"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_study.yaml")
    parser.add_argument("--output", type=Path, default=Path("reports/selection"))
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    args = parser.parse_args()
    analyze(yaml.safe_load(Path(args.config).read_text()), args.output, args.paper)
