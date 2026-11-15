"""Generate observed experiment figures; structural panels explicitly analytic."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from .records import select_latest


def _oracle_summary(part):
    rows = []
    for alpha, group in part.groupby("alpha"):
        n = int(group.num_test_samples.sum())
        errors = int(np.rint(group.mc_bayes_error * group.num_test_samples).sum())
        ci = binomtest(errors, n).proportion_ci(method="exact")
        rows.append(
            {
                "alpha": alpha,
                "theory_bayes_error": group.theory_bayes_error.mean(),
                "mc_bayes_error": errors / n,
                "ci95_low": ci.low,
                "ci95_high": ci.high,
            }
        )
    return pd.DataFrame(rows)


def _errorbars(ax, part, **kwargs):
    ax.errorbar(
        part.alpha,
        part.mc_bayes_error,
        yerr=np.array([part.mc_bayes_error - part.ci95_low, part.ci95_high - part.mc_bayes_error]),
        **kwargs,
    )


def figures(input_dir="reports", output_dir="reports/figures"):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    src = Path(input_dir) / "experiments.csv"
    if not src.exists():
        raise FileNotFoundError("Run aggregate first")
    df = pd.read_csv(src)
    # Present one most recent run for each independent experiment/seed in figures.
    # Older repeated seed-0 verification runs remain available in raw aggregates.
    df = select_latest(df)
    written = []

    def save(name):
        plt.tight_layout()
        plt.savefig(folder / (name + ".png"), dpi=150)
        plt.savefig(folder / (name + ".pdf"))
        plt.close()
        written.append(name)

    # F1 and F2 are analytic schematics, not experimental performance data.
    fig, ax = plt.subplots(figsize=(9, 2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for x, t in [
        (0.4, "Image x"),
        (3.2, "Ordered patches Qx\nLOSSLESS"),
        (6.6, "Projected tokens Ax\nRANK MAY DROP"),
    ]:
        ax.text(x, 1, t, ha="left", va="center", bbox={"boxstyle": "round", "fc": "white"})
    for x in [2.35, 5.7]:
        ax.annotate("", xy=(x + 0.6, 1), xytext=(x, 1), arrowprops={"arrowstyle": "->"})
    save("F1_operator_schematic")
    ps = [2, 4, 8, 16]
    ds = [8, 16, 32, 64]
    phase = np.array([[min(1.0, d / (p * p)) for p in ps] for d in ds])
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    im = ax.imshow(phase, origin="lower", vmin=0, vmax=1)
    ax.set_xticks(range(4), ps)
    ax.set_yticks(range(4), ds)
    ax.set_xlabel("patch width p (grayscale; m=p²)")
    ax.set_ylabel("embedding width d")
    for i, d in enumerate(ds):
        for j, p in enumerate(ps):
            ax.text(j, i, f"{min(d, p * p)}/{p * p}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="structural rank fraction (analytic)")
    save("F2_rank_phase_analytic")
    for eid, name in [("E02", "F4_gaussian"), ("E03", "F4b_task_dependence")]:
        part = df[df.experiment_id == eid] if not df.empty else pd.DataFrame()
        if len(part):
            fig, ax = plt.subplots(figsize=(6, 4))
            if eid == "E02":
                rank = part["rank"].dropna().min()
                part = part[part["rank"] == rank]
                for sigma in sorted(part.sigma.dropna().unique()):
                    ss = _oracle_summary(part[part.sigma == sigma])
                    ax.plot(ss.alpha, ss.theory_bayes_error, label=f"analytic σ={sigma}")
                    _errorbars(ax, ss, fmt="o", capsize=2, label=f"MC σ={sigma}")
            else:
                part = _oracle_summary(part)
                ax.plot(part.alpha, part.theory_bayes_error, label="analytic")
                _errorbars(ax, part, fmt="o", label="MC")
            ax.set_xlabel("row-space alignment α")
            ax.set_ylabel("classification error")
            ax.legend(fontsize=8)
            save(name)
    for eid, name in [
        ("E05", "F5_frequency_phase"),
        ("E07", "F7_lowpass_tradeoff"),
        ("E04", "F8_mean_TAR_failure"),
    ]:
        part = df[df.experiment_id == eid] if not df.empty else pd.DataFrame()
        if part.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        if eid == "E05":
            for p, group in part[part.task == "frequency"].groupby("patch_size"):
                gp = group.groupby("frequency", as_index=False).mean(numeric_only=True)
                ax.plot(gp.frequency, gp.error, "o-", label=f"p={int(p)}")
            ax.set_xlabel("grating cycles")
            ax.set_ylabel("probe held-out error")
        elif eid == "E07":
            part = part.groupby(["family", "lowpass"], as_index=False)[["accuracy", "error"]].mean()
            labels = [f"{r.family}\n" + ("smoothed" if bool(r.lowpass) else "raw") for r in part.itertuples()]
            ax.bar(labels, part.error)
            ax.set_ylabel("held-out error")
        else:
            part = part.groupby("task", as_index=False)[["logistic_accuracy", "mlp_accuracy"]].mean()
            labels = part.task.values
            ax.bar(np.arange(len(labels)) - 0.17, part.logistic_accuracy, 0.34, label="linear probe")
            ax.bar(np.arange(len(labels)) + 0.17, part.mlp_accuracy, 0.34, label="MLP probe")
            ax.set_xticks(np.arange(len(labels)), labels)
            ax.set_ylabel("held-out accuracy")
        ax.legend(fontsize=8) if eid != "E07" else None
        save(name)
    part = df[df.experiment_id == "E01"] if not df.empty else pd.DataFrame()
    if len(part):
        source = (
            Path(part.iloc[0]["source_dir"])
            if "source_dir" in part
            else Path(input_dir).parent / "runs" / str(part.iloc[0]["run_id"])
        ) / "collision_images.npz"
        if source.exists():
            obj = np.load(source)
            fig, axs = plt.subplots(1, 3, figsize=(8, 3))
            for ax, key in zip(axs, ["x_plus", "x_minus", "difference"]):
                ax.imshow(obj[key][0, 0], cmap="gray")
                ax.set_title(key)
                ax.axis("off")
            save("F3_exact_collision")
    part = df[df.experiment_id == "E05"] if not df.empty else pd.DataFrame()
    if len(part):
        fig, ax = plt.subplots(figsize=(6, 4))
        sm = part[part.task == "marker"]
        for p, group in sm.groupby("patch_size"):
            group = group.groupby("marker_size", as_index=False).mean(numeric_only=True)
            ax.plot(group.marker_size, group.error, "o-", label=f"p={int(p)}")
        ax.set_xlabel("marker side length in pixels")
        ax.set_ylabel("held-out error")
        ax.legend(fontsize=8)
        save("F5b_marker_size")
        fig, ax = plt.subplots(figsize=(6, 4))
        ss = part[part.task == "frequency"]
        for p, group in ss.groupby("patch_size"):
            vals = group.groupby("phase_shift").error.mean()
            ax.plot(vals.index, vals.values, "o-", label=f"p={int(p)}")
        ax.set_xlabel("matched random-phase offset (pixels)")
        ax.set_title("Uniform phase: same population distribution")
        ax.set_ylabel("mean held-out error")
        ax.legend(fontsize=8)
        save("F5c_phase_shift")
    part = df[df.experiment_id == "E06"] if not df.empty else pd.DataFrame()
    if len(part):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["frozen", "learned"], part.groupby("frozen")["accuracy"].mean().reindex([True, False]).values)
        ax.set_ylabel("held-out accuracy")
        save("F6_frozen_vs_learned")
        fig, ax = plt.subplots(figsize=(6, 4))
        has_history = False
        # Use the most recent (same-seed) E06 source history, never fabricated epoch values.
        for seed, grp in part.groupby("seed"):
            rid = grp.sort_values("run_id").iloc[-1]["run_id"]
            history = (
                Path(grp.iloc[0]["source_dir"])
                if "source_dir" in grp
                else Path(input_dir).parent / "runs" / str(rid)
            ) / "history.csv"
            if history.exists():
                hh = pd.read_csv(history)
                if "row_alignment" in hh:
                    for label, data in hh.groupby("model"):
                        ax.plot(data.epoch, data.row_alignment, "o-", label=f"{label} seed {int(seed)}")
                        has_history = True
        if has_history:
            ax.set_xlabel("epoch")
            ax.set_ylabel("row-space alignment")
            ax.legend(fontsize=7)
            save("F6b_alignment_trajectory")
        else:
            plt.close(fig)
    part = df[df.experiment_id == "E09"] if not df.empty else pd.DataFrame()
    if len(part):
        fig, ax = plt.subplots(figsize=(6, 4))
        for model, group in part.groupby("model"):
            ax.scatter(group.model_params, group.accuracy, label=model)
        ax.set_xlabel("parameter count")
        ax.set_ylabel("held-out accuracy")
        ax.legend(fontsize=7)
        save("F9_accuracy_parameter_frontier")
    return written
