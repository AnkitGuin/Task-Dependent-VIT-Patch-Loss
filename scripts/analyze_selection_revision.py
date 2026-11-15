"""Verify revision evidence and regenerate all revision tables, statistics and figures."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from patchloss.selection.revision import METHODS, write_json
from patchloss.selection.revision_statistics import paired_bootstrap
from patchloss.selection.statistics import block_bootstrap

LABELS = dict(rank="Rank / uniform ties", reconstruction="Reconstruction", mean_tar="Mean retention",
              covariance_tar="Covariance retention", crossfit_tar="Cross-fitted retention", hscore="Fixed H-score",
              adaptive_hscore="Adaptive H-score", profile_logme="Profile LogME", cheap_probe="Diagnostic probe",
              tuned_probe="Tuned diagnostic probe", fixed_dct="Fixed DCT (p=4)",
              covariance_lambda_0_01="Covariance, lambda=.01", covariance_lambda_0_5="Covariance, lambda=.5")
DOMAINS = dict(cifar10="CIFAR-10", cifar100="CIFAR-100", svhn="SVHN")


def label(metric):
    return LABELS.get(metric, metric.replace("_", " "))


def tex_table(filename, caption, tag, headings, rows, alignment=None):
    alignment = alignment or "l" + "r"*(len(headings)-1)
    text = [r"\begin{table}[tb]\centering\small", r"\caption{"+caption+r"}\label{"+tag+"}",
            r"\begin{tabular}{"+alignment+r"}\toprule", " & ".join(headings)+r"\\\midrule"]
    text += [" & ".join(map(str, row))+r"\\" for row in rows]
    text += [r"\bottomrule\end{tabular}", r"\end{table}"]
    Path("paper", filename).write_text("\n".join(text)+"\n", encoding="utf-8")


def analyze():
    config = yaml.safe_load(Path("configs/selection_revision.yaml").read_text())
    root = Path(config["output"])
    output = Path("reports/selection_revision")
    output.mkdir(parents=True, exist_ok=True)
    selections, candidates, within, frequencies, audits, alphas, contexts = [], [], [], [], [], [], []
    correctness, source_hashes, test_hashes = {}, set(), {}
    for domain in config["datasets"]:
        per_seed = []
        for seed in config["seeds"]:
            per_budget = []
            for budget in config["budgets"]:
                path = root / f"{domain}-b{budget}-s{seed}"
                md = json.loads((path / "metadata.json").read_text())
                assert md["status"] == "completed"
                source_hashes.add(md["source_sha256"])
                for name, digest in md["artifact_sha256"].items():
                    assert hashlib.sha256((path/name).read_bytes()).hexdigest() == digest, (path, name)
                assert hashlib.sha256((path/"pretest.json").read_bytes()).hexdigest() == md["pretest_sha256"]
                data_md = json.loads((path/"datasets.json").read_text())
                digest = data_md["partitions"]["test"]["raw_images_sha256"]
                assert domain not in test_hashes or test_hashes[domain] == digest
                test_hashes[domain] = digest
                if seed == config["seeds"][0] and budget == config["budgets"][0]:
                    audits.append(dict(dataset=domain, **data_md["deduplication"]))
                result = json.loads((path/"results.json").read_text())
                pretest = json.loads((path/"pretest.json").read_text())
                predictions = np.load(path/"predictions.npz")
                y = predictions["labels"]
                classes, test_counts = np.unique(y, return_counts=True)
                if seed == config["seeds"][0] and budget == config["budgets"][0]:
                    contexts.append(dict(dataset=domain, test_majority_class=int(classes[test_counts.argmax()]),
                                         test_majority_fraction=float(test_counts.max()/len(y)),
                                         balanced_uniform_reference=1/len(classes),
                                         balanced_training_constant_class=int(classes[0]),
                                         balanced_training_constant_accuracy=float(np.mean(y==classes[0]))))
                common = dict(dataset=domain, seed=seed, budget=budget)
                corr = []
                for row in result["rows"]:
                    row["balanced_test_accuracy"] = float(np.mean([np.mean(predictions[row["candidate"]][y==c]==c) for c in classes]))
                    np.testing.assert_allclose(row["test_accuracy"], np.mean(predictions[row["candidate"]] == y), atol=1e-15)
                    assert "test_accuracy" not in next(r for r in pretest["rows"] if r["candidate"] == row["candidate"])
                    candidates.append(common | {k: v for k, v in row.items() if k not in {"scores", "details"}})
                    curve = [r for r in pretest["validation_curves"] if r["candidate"] == row["candidate"]]
                    alphas.append(common | dict(candidate=row["candidate"], alpha=row["selected_alpha"],
                                                upper_improves_over_previous=curve[-1]["validation_accuracy"] > curve[-2]["validation_accuracy"]))
                methods = list(pretest["selection"])
                for method in methods:
                    ids = pretest["selection"][method]["eligible_candidates"]
                    corr.append(np.mean([predictions[c] == y for c in ids], axis=0))
                    balanced = float(np.mean([np.mean(corr[-1][y==c]) for c in classes]))
                    next(r for r in result["selection"] if r["metric"]==method)["balanced_test_accuracy"] = balanced
                    frequencies.extend(common | dict(metric=method, candidate=c, weight=1/len(ids)) for c in ids)
                per_budget.append(np.stack(corr))
                selections.extend(common | r for r in result["selection"])
                within.extend(common | r for r in result["within_patch"])
            per_seed.append(np.mean(per_budget, axis=0))
        correctness[domain] = np.stack(per_seed)
    assert len(source_hashes) == 1, "Mixed source cohorts"
    frame, bank, within_frame = pd.DataFrame(selections), pd.DataFrame(candidates), pd.DataFrame(within)
    frame.to_csv(output/"selection.csv", index=False)
    bank.to_csv(output/"candidates.csv", index=False)
    within_frame.to_csv(output/"within_patch.csv", index=False)
    pd.DataFrame(audits).to_csv(output/"deduplication.csv", index=False)
    pd.DataFrame(contexts).to_csv(output/"class_prior_context.csv", index=False)
    frequency = pd.DataFrame(frequencies)
    frequency.to_csv(output/"selection_frequencies_by_unit.csv", index=False)
    frequency.groupby(["dataset", "budget", "metric", "candidate"]).weight.sum().reset_index().to_csv(output/"selection_frequencies.csv", index=False)
    alpha = pd.DataFrame(alphas)
    alpha.to_csv(output/"ridge_penalties.csv", index=False)
    summary = frame.groupby("metric").agg(regret=("regret", "mean"), accuracy=("selected_accuracy", "mean"),
                                          spearman=("spearman", "mean"), undefined=("undefined_fallback", "sum"))
    summary.to_csv(output/"summary.csv")
    per_domain = frame.groupby(["dataset", "metric"])[["regret", "selected_accuracy", "spearman"]].mean()
    per_domain.to_csv(output/"by_domain.csv")
    frame.groupby(["dataset", "budget", "metric"])[["regret", "selected_accuracy", "spearman"]].mean().to_csv(output/"by_domain_budget.csv")
    balanced_means = frame.groupby(["dataset","budget","metric"]).balanced_test_accuracy.mean()
    frame.groupby(["dataset","budget","metric"]).balanced_test_accuracy.mean().to_csv(output/"balanced_accuracy_secondary.csv")
    balance_rows = []
    for domain in config["datasets"]:
        prior = next(r for r in contexts if r["dataset"]==domain)
        for budget in config["budgets"]:
            oracle = bank[(bank.dataset==domain)&(bank.budget==budget)].groupby("seed").balanced_test_accuracy.max().mean()
            balance_rows.append([DOMAINS[domain], budget, f"{100*prior['test_majority_fraction']:.2f}",f"{100*oracle:.2f}"]+
                                 [f"{100*balanced_means.loc[(domain,budget,m)]:.2f}" for m in ["covariance_tar","profile_logme","fixed_dct"]])
    tex_table("revision_balance.tex", "Secondary, post-outcome class-prior context. Majority is the observed test majority-class frequency (a test-informed descriptive constant-class reference, not a prospectively selected rule). Remaining columns are macro-average class recall (balanced accuracy, percent); the oracle is defined on that secondary endpoint. Primary selection and regret remain ordinary full-test accuracy.",
              "tab:balance", ["Domain","$B$","Majority","Bal. oracle","Bal. cov.","Bal. LogME","Bal. DCT"],balance_rows)
    intervals, bootstraps = [], {}
    for domain in config["datasets"]:
        print(f"BOOTSTRAP {domain}", flush=True)
        seed_boot, joint_boot = paired_bootstrap(correctness[domain], config["bootstrap_samples"], seed=71+config["datasets"].index(domain))
        bootstraps[domain] = seed_boot, joint_boot
    bootstraps["all"] = tuple(np.mean([bootstraps[d][i] for d in config["datasets"]], axis=0) for i in [0, 1])
    for domain, (seed_boot, joint_boot) in bootstraps.items():
        point = correctness[domain].mean((0, 2)) if domain != "all" else np.mean([c.mean((0, 2)) for c in correctness.values()], axis=0)
        for anchor in ["covariance_tar", "fixed_dct"]:
            ai = methods.index(anchor)
            for j, method in enumerate(methods):
                seed_ci = np.quantile((seed_boot[:,j]-seed_boot[:,ai])*100, [.025,.975])
                joint_ci = np.quantile((joint_boot[:,j]-joint_boot[:,ai])*100, [.025,.975])
                intervals.append(dict(dataset=domain, metric=method, reference=anchor,
                                      accuracy_gain_pp=(point[j]-point[ai])*100,
                                      seed_ci_low=seed_ci[0], seed_ci_high=seed_ci[1],
                                      joint_ci_low=joint_ci[0], joint_ci_high=joint_ci[1]))
    paired = pd.DataFrame(intervals)
    paired.to_csv(output/"paired_accuracy_intervals.csv", index=False)
    order, regret_boot = block_bootstrap(frame, "regret", config["bootstrap_samples"], seed=28)
    pd.DataFrame(dict(metric=order, low=np.quantile(regret_boot,.025,axis=0)*100,
                       high=np.quantile(regret_boot,.975,axis=0)*100)).to_csv(output/"regret_seed_intervals.csv", index=False)
    # Domain means get equal weight; seeds are repetitions within the same task.
    main_rows = []
    for method in METHODS:
        r = summary.loc[method]
        main_rows.append([label(method)] + [f"{100*per_domain.loc[(d,method),'regret']:.3f}" for d in config["datasets"]]
                         + [f"{100*r.regret:.3f}", "--" if pd.isna(r.spearman) or method=="fixed_dct" else f"{r.spearman:.3f}"])
    tex_table("revision_comparison.tex", "Revision selection regret (percentage points), averaged over seeds and budgets. The three observed domains receive equal weight. Rank uses uniform tie expectations; correlations are within candidate banks, then averaged. A fixed policy supplies no meaningful full-bank ranking, so its correlation is omitted.",
              "tab:revision", ["Selector", "CIFAR-10", "CIFAR-100", "SVHN", "Mean", r"$\rho$"], main_rows)
    absolute = []
    for (domain,budget), part in frame.groupby(["dataset","budget"]):
        means = part.groupby("metric").selected_accuracy.mean()
        absolute.append([DOMAINS[domain], budget, f"{100*part.oracle_accuracy.mean():.2f}"] +
                        [f"{100*means[m]:.2f}" for m in ["covariance_tar","profile_logme","tuned_probe","fixed_dct"]])
    tex_table("revision_absolute.tex", "Full-test absolute accuracy (percent). Oracle selects by test outcomes only as a descriptive upper reference. Cov., LogME, Probe and DCT denote covariance retention, profile LogME, tuned diagnostic probe and locked fixed DCT.",
              "tab:absolute", ["Domain", "$B$", "Oracle", "Cov.", "LogME", "Probe", "DCT"], absolute)
    def ci_cell(row):
        return f"{row.accuracy_gain_pp:+.3f} [{row.joint_ci_low:+.3f}, {row.joint_ci_high:+.3f}]"
    paired_rows = []
    for method in ["covariance_tar", "profile_logme", "adaptive_hscore", "cheap_probe", "tuned_probe"]:
        a = paired[(paired.dataset=="all") & (paired.metric==method) & (paired.reference=="fixed_dct")].iloc[0]
        b = paired[(paired.dataset=="svhn") & (paired.metric==method) & (paired.reference=="fixed_dct")].iloc[0]
        paired_rows.append([label(method), ci_cell(a), ci_cell(b)])
    tex_table("revision_paired.tex", "Paired accuracy gain over fixed DCT, in percentage points, with 95\\% joint seed/image bootstrap intervals. Only SVHN provides independent prospective evaluation of the CIFAR-developed fixed policy. The pooled column is descriptive. Full seed-only and all pairwise-reference results accompany the code.",
              "tab:paired", ["Selector", "Three-domain gain [CI]", "SVHN gain [CI]"], paired_rows, "lrr")
    freq_rows = []
    for method in ["mean_tar","covariance_tar","profile_logme","adaptive_hscore","tuned_probe","fixed_dct"]:
        counts = frequency[frequency.metric==method].groupby("candidate").weight.sum().sort_values(ascending=False)
        first = counts.index[0]
        svhn = frequency[(frequency.dataset=="svhn") & (frequency.metric==method)].groupby("candidate").weight.sum().sort_values(ascending=False)
        freq_rows.append([label(method), first.replace("_",r"\_"), f"{counts.iloc[0]:g}/30",
                          svhn.index[0].replace("_",r"\_"), f"{svhn.iloc[0]:g}/10"])
    tex_table("revision_frequencies.tex", "Most frequently selected candidate, with fractional counts for tied selections. All candidate frequencies, including remaining mass, are supplied as CSV. SVHN contains ten dataset/seed/budget units, not ten independent domains.",
              "tab:frequencies", ["Selector", "Modal candidate", "Count", "SVHN modal candidate", "Count"], freq_rows, "llrlr")
    sensitivity = []
    for method in ["covariance_lambda_0.01","covariance_tar","covariance_lambda_0.5"]:
        sensitivity.append(["0.1" if method=="covariance_tar" else method.split("_")[-1]] +
                           [f"{100*per_domain.loc[(d,method),'regret']:.3f}" for d in config["datasets"]] +
                           [f"{100*summary.loc[method,'regret']:.3f}"])
    tex_table("revision_shrinkage.tex", "Predetermined covariance-shrinkage sensitivity: mean regret in percentage points. Lambda=0.1 is the locked default; no setting is chosen retrospectively for the main comparison.",
              "tab:shrinkage", [r"$\lambda$","CIFAR-10","CIFAR-100","SVHN","Mean"], sensitivity)
    within_means = within_frame.groupby(["patch_size","metric"]).regret.mean()
    tex_table("revision_matched.tex", "Regret within matched patch size and feature budget, in percentage points, averaged over the three domains and five seeds. The fixed p=4 policy has no within-p analogue at other patch sizes and is omitted.",
              "tab:matched", ["Patch size", "Covariance", "LogME", "Adaptive H", "Tuned probe"],
              [[p]+[f"{100*within_means.loc[(p,m)]:.3f}" for m in ["covariance_tar","profile_logme","adaptive_hscore","tuned_probe"]] for p in config["patch_sizes"]])
    cost_path = output/"cost_measurements.csv"
    if cost_path.exists():
        costs = pd.read_csv(cost_path)
        assert len(costs) == len(config["datasets"])*len(config["budgets"])*len(METHODS)*3
        medians = costs.groupby("metric")[["feature_statistics_score_sec","cold_selection_sec","peak_rss_mib","incremental_peak_rss_mib"]].median()
        medians.to_csv(output/"cost_summary.csv")
        tex_table("revision_cost.tex", "Selection costs on identical CPU hardware (two numerical threads): median of three fresh-process repeats for each domain and budget, at seed 10. Warm time includes all required extraction, statistics and scoring; cold includes bank fitting once. Peak RSS includes cold operator construction; incremental RSS is relative to the loaded-input baseline (10 ms sampling). Input decoding, imports, downstream head fitting and test inference are excluded. Regret repeats the full-cohort mean from Table~\\ref{tab:revision}.",
                  "tab:cost", ["Selector", "Warm (s)", "Cold (s)", "RSS (MiB)", r"$\Delta$RSS", "Regret"],
                  [[label(m), ("$<0.001$" if medians.loc[m,'feature_statistics_score_sec']<.001 else f"{medians.loc[m,'feature_statistics_score_sec']:.3f}"),
                    ("$<0.001$" if medians.loc[m,'cold_selection_sec']<.001 else f"{medians.loc[m,'cold_selection_sec']:.3f}"),
                    f"{medians.loc[m,'peak_rss_mib']:.0f}", ("$<1$" if medians.loc[m,'incremental_peak_rss_mib']<1 else f"{medians.loc[m,'incremental_peak_rss_mib']:.0f}"),f"{100*summary.loc[m,'regret']:.3f}"] for m in METHODS])
    # Historical test-exclusion deltas, holding historical heads and choices fixed.
    old = pd.read_csv("reports/selection/selection_by_unit.csv")
    excluded = pd.read_csv("reports/selection/deduplicated_test_selection.csv")
    delta = old.merge(excluded,on=["dataset","seed","budget","metric"], suffixes=("_original","_excluded"))
    delta["regret_change_pp"] = 100*(delta.regret_excluded-delta.regret_original)
    delta["accuracy_change_pp"] = 100*(delta.selected_accuracy_excluded-delta.selected_accuracy_original)
    delta.to_csv(output/"historical_test_exclusion_deltas.csv",index=False)
    # H-score family/conditioning audit is descriptive, not a causal explanation.
    score_rows = []
    for domain in config["datasets"]:
        for seed in config["seeds"]:
            for budget in config["budgets"]:
                rows = json.loads((root/f"{domain}-b{budget}-s{seed}"/"results.json").read_text())["rows"]
                for r in rows:
                    score_rows.append(dict(dataset=domain, seed=seed, budget=budget, family=r["family"],
                                           accuracy=r["test_accuracy"], fixed_h=r["scores"]["hscore"],
                                           adaptive_h=r["scores"]["adaptive_hscore"],
                                           adaptive_lambda=r["details"]["adaptive_hscore"]["shrinkage"]))
    pd.DataFrame(score_rows).groupby(["dataset","budget","family"]).mean(numeric_only=True).to_csv(output/"hscore_family_audit.csv")
    # Large labels, one readable axis; supporting old panels are expanded in the appendix.
    fig, ax = plt.subplots(figsize=(8.7,5.6))
    y = np.arange(len(METHODS))
    for offset,(domain,color) in enumerate(zip(config["datasets"],["#0072B2","#D55E00","#009E73"])):
        means = [100*per_domain.loc[(domain,m),"regret"] for m in METHODS]
        ax.scatter(means,y+(offset-1)*.18,label=DOMAINS[domain],color=color,s=36)
    ax.set_yticks(y,[label(m) for m in METHODS],fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean selection regret (percentage points)",fontsize=11)
    ax.grid(axis="x",alpha=.25)
    ax.legend(loc="lower right",frameon=False)
    fig.tight_layout()
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    for ext in ["pdf","png"]:
        fig.savefig(figures/f"regret_by_domain.{ext}",dpi=190)
    plt.close(fig)
    evidence = dict(units=len(frame[frame.metric=="fixed_dct"]),candidate_heads=len(bank),
                    domains=config["datasets"],source_sha256=next(iter(source_hashes)),
                    test_set_hashes=test_hashes,hashes_and_prediction_accuracy_verified=True,
                    grid_min_count=int((alpha.alpha==min(config["ridge_alphas"])).sum()),
                    grid_max_count=int((alpha.alpha==max(config["ridge_alphas"])).sum()),
                    strict_upper_improvements=int(alpha.upper_improves_over_previous.sum()),
                    lower_half_grid_selected=int((alpha.alpha<=1000).sum()),
                    bank_fits_sec=float(bank.head_fit_sec.sum()),
                    covariance_vs_profile=paired[(paired.dataset=="all") & (paired.metric=="covariance_tar") & (paired.reference=="fixed_dct")].to_dict("records"),
                    historical_max_accuracy_delta_pp=float(delta.accuracy_change_pp.abs().max()),
                    historical_mean_covariance_regret_delta_pp=float(delta[delta.metric=="covariance_tar"].regret_change_pp.mean()))
    write_json(output/"audit.json",evidence)
    macros = {}
    for name,method in [("Cov","covariance_tar"),("Logme","profile_logme"),("Tuned","tuned_probe"),("Fixed","fixed_dct")]:
        macros[f"Rev{name}Regret"] = f"{100*summary.loc[method,'regret']:.3f}"
    diff = paired[(paired.dataset=="all") & (paired.metric=="profile_logme") & (paired.reference=="covariance_tar")].iloc[0]
    macros["RevCovLogmeGain"] = f"{-diff.accuracy_gain_pp:+.3f}"
    macros["RevCovLogmeCI"] = f"$[{-diff.joint_ci_high:+.3f}, {-diff.joint_ci_low:+.3f}]$"
    diff = paired[(paired.dataset=="all") & (paired.metric=="covariance_tar") & (paired.reference=="fixed_dct")].iloc[0]
    macros["RevCovFixedGain"] = f"{diff.accuracy_gain_pp:+.3f}"
    macros["RevCovFixedCI"] = f"$[{diff.joint_ci_low:+.3f}, {diff.joint_ci_high:+.3f}]$"
    for name,method in [("Cov","covariance_tar"),("Logme","profile_logme")]:
        diff = paired[(paired.dataset=="svhn") & (paired.metric==method) & (paired.reference=="fixed_dct")].iloc[0]
        macros[f"RevSvhn{name}Gain"] = f"{diff.accuracy_gain_pp:+.3f}"
        macros[f"RevSvhn{name}CI"] = f"$[{diff.joint_ci_low:+.3f}, {diff.joint_ci_high:+.3f}]$"
    hardware = json.loads((output/"hardware.json").read_text())
    macros.update(RevUpperGridCount=str(evidence["grid_max_count"]),
                  RevHardware=hardware["cpu"].replace("_",r"\_"),
                  RevCrossfitUndefined=str(int(summary.loc["crossfit_tar","undefined"])),
                  RevLowerGridCount=str(evidence["grid_min_count"]),
                  RevUpperImprovementCount=str(evidence["strict_upper_improvements"]),
                  RevHscoreRho=f"{summary.loc['hscore','spearman']:.3f}",
                  RevAdaptiveRho=f"{summary.loc['adaptive_hscore','spearman']:.3f}",
                  RevHeadSeconds=f"{bank.head_fit_sec.median():.3f}",
                  RevHistoricalMaxDelta=f"{evidence['historical_max_accuracy_delta_pp']:.4f}",
                  RevHistoricalCovDelta=f"{evidence['historical_mean_covariance_regret_delta_pp']:+.7f}",
                  RevSvhnMajority=f"{100*next(r for r in contexts if r['dataset']=='svhn')['test_majority_fraction']:.2f}")
    Path("paper/revision_macros.tex").write_text("% Generated from verified revision artifacts.\n"+
        "\n".join(chr(92)+"newcommand{"+chr(92)+name+"}{"+value+"}" for name,value in macros.items())+"\n",encoding="utf-8")
    print(summary.to_string(),flush=True)
    print(json.dumps(evidence,indent=2),flush=True)


if __name__ == "__main__":
    with threadpool_limits(2):
        analyze()
