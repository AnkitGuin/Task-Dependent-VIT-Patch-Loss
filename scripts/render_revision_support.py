"""Redraw measured supporting controls with readable fonts and larger panels."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path("runs/selection_controls")
    control = json.loads((root/"metrics.json").read_text())
    output = Path("reports/selection_revision/figures")
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":12,"axes.labelsize":12,"xtick.labelsize":11,"ytick.labelsize":11})
    def save(fig,name):
        fig.tight_layout()
        for ext in ["pdf","png"]:
            fig.savefig(output/f"{name}.{ext}",dpi=190)
        plt.close(fig)
    fig,ax = plt.subplots(figsize=(6.2,4.2))
    ax.hist(np.load(root/"null_tar_n20.npy"),bins=35,density=True,alpha=.6,label="4,000 measured null tasks")
    x = np.linspace(.001,.65,300)
    ax.plot(x,beta(8,24).pdf(x),label="Analytic Beta(8,24)")
    ax.axvline(.25,color="black",linestyle="--",label="Rank fraction 16/64")
    ax.set_xlabel("Empirical mean retention without class signal")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    save(fig,"null_tar_calibration")
    anis = pd.DataFrame(control["anisotropic"])
    fig,axs = plt.subplots(2,1,figsize=(6.2,6.6))
    axs[0].bar(["Coordinate 0","Coordinate 1"],anis.euclidean_tar,color="#0072B2")
    axs[0].set_ylabel("Euclidean mean retention")
    axs[0].set_ylim(0,.65)
    axs[1].bar(["Coordinate 0","Coordinate 1"],anis.theory_bayes_error,color="#D55E00",label="Analytic")
    axs[1].plot([0,1],anis.mc_error,"ko",label="Independent token Monte Carlo")
    axs[1].set_ylabel("Gaussian Bayes error")
    axs[1].legend(fontsize=10)
    save(fig,"anisotropic_counterexample")
    fig,ax = plt.subplots(figsize=(6.2,4.2))
    for row in control["pretrained_audit"]:
        s = np.asarray(row["singular_values"])
        ax.semilogy(np.arange(1,len(s)+1),s/s[0],label=f"{row['model']}: rank {row['rank_rtol_1e10']}/{row['patch_dim']}")
    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Relative singular value")
    ax.legend(fontsize=10)
    save(fig,"pretrained_spectra")
    capacity = pd.DataFrame(control["capacity"])
    fig,ax = plt.subplots(figsize=(6.2,4.4))
    for i,(name,part) in enumerate(capacity.groupby("model")):
        for seed,sub in part.groupby("seed"):
            sub = sub.sort_values("epoch_cap")
            ax.plot([i-.14,i+.14],sub.accuracy,"o-",alpha=.6)
    ax.set_xticks(range(capacity.model.nunique()),sorted(capacity.model.unique()))
    ax.set_ylabel("Fresh-noise XOR test accuracy")
    ax.set_ylim(.4,1.05)
    ax.set_title("Paired seeds: 6 epochs (left), 80 epochs (right)",fontsize=11)
    save(fig,"xor_training_budget")
    print("Rendered four supporting figures from recorded control observations")


if __name__ == "__main__":
    main()
