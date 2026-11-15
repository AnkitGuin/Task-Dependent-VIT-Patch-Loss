"""Regenerate every selection-study table, plot and manuscript macro from audited runs."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_distribution
import yaml
from patchloss.reporting.records import read_record
from patchloss.selection.statistics import block_bootstrap

LABELS = {
    "rank": "Rank / uniform tie",
    "reconstruction": "Reconstruction",
    "mean_tar": "Mean TAR",
    "covariance_tar": "Covariance TAR",
    "crossfit_tar": "Cross-fitted TAR",
    "hscore": "Fixed-shrink H-score",
    "logme": "LogME (11 steps)",
    "cheap_probe": "Diagnostic probe",
}


def analyze(config, controls, output, paper):
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    records = []
    candidates = []
    units = []
    within = []
    manifests = []
    expected = {
        (dataset, seed, budget)
        for dataset in config["datasets"]
        for seed in config["seeds"]
        for budget in config["budgets"]
    }
    found = set()
    sources = set()
    for directory in sorted(Path(config["output"]).iterdir()):
        if not (directory / "metadata.json").exists():
            continue
        md, data = read_record(directory)
        key = (md["dataset"], md["seed"], md["budget"])
        if key not in expected or key in found:
            raise ValueError("Unexpected or duplicate study unit")
        found.add(key)
        sources.add(md["source_sha256"])
        meta = {"dataset": key[0], "seed": key[1], "budget": key[2], "run_id": md["run_id"]}
        pretest_path = directory / "pretest_selection.json"
        if hashlib.sha256(pretest_path.read_bytes()).hexdigest() != data["metrics"]["pretest_sha256"]:
            raise ValueError("Pretest selection fingerprint mismatch")
        pretest = json.loads(pretest_path.read_text())
        if any("test_accuracy" in row for row in pretest["rows"]):
            raise ValueError("Pretest artifact contains test outcomes")
        for row in data["rows"]:
            flat = {
                **meta,
                **{k: v for k, v in row.items() if k not in {"scores", "score_seconds", "details"}},
            }
            flat.update({f"score_{k}": v for k, v in row["scores"].items()})
            flat.update({f"seconds_{k}": v for k, v in row["score_seconds"].items()})
            flat["logme_nonconverged_classes"] = sum(
                not p["converged"] for p in row["details"]["logme_parameters"]
            )
            candidates.append(flat)
        records.extend({**meta, **r} for r in data["metrics"]["selection"])
        within.extend({**meta, **r} for r in data["metrics"]["within_patch_selection"])
        units.append(
            {
                **meta,
                **{
                    k: v
                    for k, v in data["metrics"].items()
                    if k not in {"selection", "within_patch_selection"}
                },
            }
        )
        manifests.append(
            {
                "directory": str(directory.resolve()),
                "metadata": md,
                "metrics_sha256": hashlib.sha256((directory / "metrics.json").read_bytes()).hexdigest(),
            }
        )
    if found != expected or len(sources) != 1:
        raise ValueError(
            f"Incomplete or mixed-source primary study: missing={expected - found}, source_versions={len(sources)}"
        )
    frame = pd.DataFrame(records)
    cells = pd.DataFrame(candidates)
    unit_frame = pd.DataFrame(units)
    frame.to_csv(output / "selection_by_unit.csv", index=False)
    cells.to_csv(output / "candidates.csv", index=False)
    unit_frame.to_csv(output / "units.csv", index=False)
    pd.DataFrame(within).to_csv(output / "within_patch_selection.csv", index=False)
    metrics, bootstrap = block_bootstrap(frame, "regret", config["bootstrap_samples"], seed=2718)
    summary = []
    for j, metric in enumerate(metrics):
        part = frame[frame.metric == metric]
        summary.append(
            {
                "metric": metric,
                "mean_regret": part.regret.mean(),
                "regret_ci_low": np.quantile(bootstrap[:, j], 0.025),
                "regret_ci_high": np.quantile(bootstrap[:, j], 0.975),
                "mean_selected_accuracy": part.selected_accuracy.mean(),
                "mean_spearman": part.spearman.mean() if part.spearman.notna().any() else None,
                "undefined_units": int(part.undefined_fallback.sum()),
                "mean_tie_count": part.tie_count.mean(),
                "mean_marginal_score_sec": cells[f"seconds_{metric}"].mean(),
            }
        )
    summary = pd.DataFrame(summary).sort_values("mean_regret")
    summary.to_csv(output / "metric_summary.csv", index=False)
    dataset_table = frame.groupby(["dataset", "budget", "metric"], as_index=False).agg(
        mean_regret=("regret", "mean"),
        mean_selected_accuracy=("selected_accuracy", "mean"),
        mean_spearman=("spearman", "mean"),
    )
    dataset_table.to_csv(output / "metric_by_dataset_budget.csv", index=False)
    differences = []
    for j, metric in enumerate(metrics):
        if metric == "mean_tar":
            continue
        delta = bootstrap[:, j] - bootstrap[:, metrics.index("mean_tar")]
        differences.append(
            {
                "metric": metric,
                "minus": "mean_tar",
                "mean_regret_difference": float(delta.mean()),
                "conditional_ci_low": float(np.quantile(delta, 0.025)),
                "conditional_ci_high": float(np.quantile(delta, 0.975)),
            }
        )
    pd.DataFrame(differences).to_csv(output / "paired_regret_differences.csv", index=False)
    # Controls use a separate checksum convention because they are a multi-part cohort.
    control_meta = json.loads((controls / "metadata.json").read_text())
    if control_meta["status"] != "completed":
        raise ValueError("Controls incomplete")
    for name, expected_hash in control_meta["artifact_sha256"].items():
        if hashlib.sha256((controls / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Control checksum mismatch: {name}")
    control = json.loads((controls / "metrics.json").read_text())
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "singular_values"} for r in control["pretrained_audit"]]
    ).to_csv(output / "pretrained_geometry.csv", index=False)

    def save(name):
        plt.tight_layout()
        plt.savefig(figure_dir / (name + ".pdf"))
        plt.savefig(figure_dir / (name + ".png"), dpi=180)
        plt.close()

    plot = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(
        [LABELS[m] for m in plot.metric],
        100 * plot.mean_regret,
        xerr=100 * np.array([plot.mean_regret - plot.regret_ci_low, plot.regret_ci_high - plot.mean_regret]),
        capsize=3,
    )
    ax.set_xlabel("Selection regret (accuracy percentage points; lower is better)")
    save("selection_regret")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dataset in zip(axs, config["datasets"]):
        for p, part in cells[(cells.dataset == dataset) & (cells.budget == 768)].groupby("patch_size"):
            ax.scatter(part.score_mean_tar, part.test_accuracy, label=f"p={p}", s=24)
        ax.set_xlabel("Empirical mean TAR")
        ax.set_title(dataset)
    axs[0].set_ylabel("Held-out ridge accuracy")
    axs[-1].legend(fontsize=8)
    save("tar_vs_accuracy")
    fig, ax = plt.subplots(figsize=(6, 4))
    values = np.load(controls / "null_tar_n20.npy")
    ax.hist(values, bins=35, density=True, alpha=0.6, label="4000 measured null tasks")
    x = np.linspace(0.001, 0.65, 300)
    ax.plot(x, beta_distribution(8, 24).pdf(x), label="Analytic Beta(8,24)")
    ax.axvline(0.25, color="black", linestyle="--", label="Rank fraction 16/64")
    ax.set_xlabel("Empirical mean TAR with no class signal")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    save("null_tar_calibration")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    anis = pd.DataFrame(control["anisotropic"])
    axs[0].bar(["coordinate 0", "coordinate 1"], anis.euclidean_tar)
    axs[0].set_ylabel("Euclidean mean retention")
    axs[0].set_ylim(0, 0.6)
    axs[1].bar(["coordinate 0", "coordinate 1"], anis.theory_bayes_error, label="Analytic")
    axs[1].plot([0, 1], anis.mc_error, "ko", label="Independent token MC")
    axs[1].set_ylabel("Gaussian Bayes error")
    axs[1].legend(fontsize=8)
    save("anisotropic_counterexample")
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in control["pretrained_audit"]:
        s = np.asarray(row["singular_values"])
        ax.semilogy(
            np.arange(1, len(s) + 1),
            s / s[0],
            label=f"{row['model']}: rank {row['rank_rtol_1e10']}/{row['patch_dim']}",
        )
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Relative singular value")
    ax.legend(fontsize=8)
    save("pretrained_spectra")
    capacity = pd.DataFrame(control["capacity"])
    if not capacity.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (name, part) in enumerate(capacity.groupby("model")):
            for seed, sub in part.groupby("seed"):
                sub = sub.sort_values("epoch_cap")
                ax.plot([i - 0.14, i + 0.14], sub.accuracy, "o-", alpha=0.55)
        ax.set_xticks(range(capacity.model.nunique()), sorted(capacity.model.unique()))
        ax.set_ylabel("Held-out fresh-noise XOR accuracy")
        ax.set_ylim(0.4, 1.05)
        ax.set_title("Paired seeds: 6-epoch cap (left), 80-epoch cap (right)")
        save("xor_training_budget")
    translation = pd.DataFrame(control["translations"])
    fig, ax = plt.subplots(figsize=(6, 4))
    for p, part in translation.groupby("patch_size"):
        group = part.groupby("test_shift").accuracy.agg(["mean", "std"])
        ax.errorbar(group.index, group["mean"], yerr=group["std"], fmt="o-", capsize=3, label=f"p={p}")
    ax.set_xlabel("Held-out translation (pixels); training shift = 0")
    ax.set_ylabel("Held-out probe accuracy; bars = seed SD")
    ax.legend()
    save("heldout_translation")
    fig, ax = plt.subplots(figsize=(6, 4))
    for p, part in cells[cells.budget == 768].groupby("patch_size"):
        ax.scatter(part.projection_flops_proxy, part.test_accuracy, s=20, label=f"p={p}")
    ax.set_xscale("log")
    ax.set_xlabel("Dense projection FLOPs proxy (fixed B=768)")
    ax.set_ylabel("Held-out ridge accuracy")
    ax.legend(fontsize=8)
    save("fixed_budget_cost")
    # All manuscript numbers are generated from the records above.
    paper.mkdir(parents=True, exist_ok=True)
    macros = {
        "SelectionUnits": str(len(units)),
        "CandidateFits": str(len(cells)),
        "BestMetric": LABELS[summary.iloc[0].metric],
        "BestRegret": f"{100 * summary.iloc[0].mean_regret:.2f}",
        "MeanTarRegret": f"{100 * summary[summary.metric == 'mean_tar'].iloc[0].mean_regret:.2f}",
        "CheapProbeRegret": f"{100 * summary[summary.metric == 'cheap_probe'].iloc[0].mean_regret:.2f}",
        "CovarianceRegret": f"{100 * summary[summary.metric == 'covariance_tar'].iloc[0].mean_regret:.2f}",
        "CrossfitRegret": f"{100 * summary[summary.metric == 'crossfit_tar'].iloc[0].mean_regret:.2f}",
        "LogmeRegret": f"{100 * summary[summary.metric == 'logme'].iloc[0].mean_regret:.2f}",
        "PrimaryRuntime": f"{unit_frame.wallclock_sec.sum():.1f}",
        "ControlsRuntime": f"{control_meta['wallclock_sec']:.1f}",
        "UndefinedCrossfit": str(int(summary[summary.metric == "crossfit_tar"].iloc[0].undefined_units)),
    }
    (paper / "selection_results.tex").write_text(
        "% AUTOGENERATED; source manifest in reports/selection.\n"
        + "\n".join(
            chr(92) + "newcommand{" + chr(92) + key + "}{" + value + "}" for key, value in macros.items()
        )
        + "\n"
    )
    tex = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Primary tokenizer selection across two CIFAR datasets, two scalar budgets and five seeds. Regret is in accuracy percentage points. Intervals resample paired seed blocks within each fixed dataset, preserving budgets; they do not quantify generalization to new domains. Correlations exclude undefined constant-score cases.}\label{tab:selection}",
        r"\begin{tabular}{lrrr}\toprule Score & Mean regret [95\% CI] & Mean $\rho$ & Undefined units\\\midrule",
    ]
    for row in summary.itertuples():
        rho = "--" if pd.isna(row.mean_spearman) else f"{row.mean_spearman:.3f}"
        tex.append(
            f"{LABELS[row.metric]} & {100 * row.mean_regret:.2f} [{100 * row.regret_ci_low:.2f}, {100 * row.regret_ci_high:.2f}] & {rho} & {row.undefined_units}"
            + r"\\"
        )
    tex.append(r"\bottomrule\end{tabular}\end{table}")
    tex.extend(
        [
            r"\begin{table}[t]",
            r"\centering\small",
            r"\caption{Observed pretrained patch-matrix geometry. Numerical ranks use a relative $10^{-10}$ threshold on float64 SVD of stored float32 weights. This audits only the patch projection, not the full encoder.}\label{tab:pretrained}",
            r"\begin{tabular}{lrrrr}\toprule Patch map & $m$ & $d$ & Rank & Condition\\\midrule",
        ]
    )
    for row in control["pretrained_audit"]:
        model_name = row["model"].replace("_", chr(92) + "_")
        tex.append(
            f"{model_name} & {row['patch_dim']} & {row['embedding_dim']} & {row['rank_rtol_1e10']} & {row['condition_nonzero']:.1f}"
            + r"\\"
        )
    tex.append(r"\bottomrule\end{tabular}\end{table}")
    (paper / "selection_tables.tex").write_text("\n".join(tex) + "\n")
    manifest = {
        "config": config,
        "source_versions": sorted(sources),
        "units": manifests,
        "controls_directory": str(controls.resolve()),
        "controls_metadata": control_meta,
        "interpretation": "descriptive, conditional on these datasets and the predeclared bank",
        "generated_artifact_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in output.iterdir()
            if f.is_file() and f.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary.to_json(output / "summary.json", orient="records", indent=2)
    print(summary.to_string(index=False))
    print(f"AUDITED {len(units)} units, {len(cells)} candidates; controls and pretest artifacts verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selection_study.yaml")
    parser.add_argument("--controls", type=Path, default=Path("runs/selection_controls"))
    parser.add_argument("--output", type=Path, default=Path("reports/selection"))
    parser.add_argument("--paper", type=Path, default=Path("paper"))
    args = parser.parse_args()
    analyze(yaml.safe_load(Path(args.config).read_text()), args.controls, args.output, args.paper)
