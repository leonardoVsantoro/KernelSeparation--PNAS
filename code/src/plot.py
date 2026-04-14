import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.serif": "Palatino",
})


def res_plot(df):
    """
    Plot Monte Carlo results: standardized violin plots + empirical power curve.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: ["Hypothesis", "Value", "Statistic", "Sample Size"]

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    Ns = sorted(df["Sample Size"].unique())
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 2, 1])
    axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    ax_right = fig.add_subplot(gs[:, 2])
    palette = {"Null": "#4C72B0", "Alternative": "#DD8452"}

    titles = {
        "MMD": r"$\mathrm{MMD}^2(\mathbb{P}_n, \mathbb{Q}_n)$",
        "KLR": (r"$\mathrm{D}_{\mathrm{KL}}"
                r"({\mathcal{P}_N}_{\#}\mathcal{N}_{\mathbb{P}_n},"
                r"\,{\mathcal{P}_N}_{\#}\mathcal{N}_{\mathbb{Q}_n})$"),
    }

    for ax, stat in zip(axes, ["MMD", "KLR"]):
        sub = df[df["Statistic"] == stat].copy()
        sub["Std Value"] = sub.groupby("Sample Size")["Value"].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        sns.violinplot(
            data=sub, y="Std Value", x="Sample Size", hue="Hypothesis",
            split=True, palette=palette, cut=10, inner=None,
            width=0.8, legend=True, ax=ax,
        )
        ax.set_title(titles[stat], fontsize=14)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.legend_.remove()

    axes[-1].set_xlabel("Sample Size", fontsize=11)
    axes[0].set_xticklabels([])

    # Power curve
    linestyles = {"KLR": "dashdot", "MMD": (0, (1, 1))}
    linewidths  = {"KLR": 2,         "MMD": 1.5}

    for stat in ["KLR", "MMD"]:
        power_means, power_stds = [], []
        for N in Ns:
            sub_N   = df[(df["Sample Size"] == N) & (df["Statistic"] == stat)]
            null_v  = sub_N[sub_N["Hypothesis"] == "Null"]["Value"]
            alt_v   = sub_N[sub_N["Hypothesis"] == "Alternative"]["Value"]
            thr     = np.percentile(null_v, 95)
            power   = np.mean(alt_v > thr)
            power_means.append(power)
            power_stds.append(np.sqrt(power * (1 - power) / len(alt_v)))
        ax_right.errorbar(
            Ns, power_means, yerr=power_stds,
            marker="o", capsize=5, capthick=2,
            linewidth=linewidths[stat], markersize=4,
            label=stat, alpha=0.6, color="k",
            linestyle=linestyles[stat],
        )

    ax_right.set_title("Monte-Carlo Power", fontsize=12)
    ax_right.set_xlabel("Sample Size", fontsize=11)
    ax_right.set_ylabel("Power", fontsize=11)
    ax_right.set_ylim(0, 1.05)
    ax_right.grid(True, alpha=0.3)

    # Shared legend below violin plots
    label_map = {"MMD": "MMD", "KLR": r"$\mathrm{D}_{\mathrm{KL}}$"}
    hyp_handles, hyp_labels = axes[0].get_legend_handles_labels()
    hyp_labels = [label_map.get(l, l) for l in hyp_labels]
    axes[-1].legend(hyp_handles, hyp_labels, loc="upper center",
                    bbox_to_anchor=(0.5, -0.4), ncol=2)

    # Legend for power panel
    stat_handles = [
        mlines.Line2D([], [], color="k", linestyle=linestyles[s],
                      linewidth=linewidths[s], label=label_map[s])
        for s in ["KLR", "MMD"]
    ]
    ax_right.legend(handles=stat_handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.17), ncol=2)

    return fig
