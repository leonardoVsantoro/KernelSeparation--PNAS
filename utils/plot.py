
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

plt.rcParams.update({'mathtext.fontset': 'stix',
                     'font.family': 'serif',
                     'font.serif':'Palatino'})


def res_plot(df):
    '''
    Plot results from MC test values DataFrame
    -----------------------------------------------
    Inputs:
        df: pandas DataFrame with columns ["Hypothesis", "Value", "Statistic", "Sample Size"]
    Outputs:
        fig: matplotlib Figure object
    -----------------------------------------------
    '''
    Ns = sorted(df["Sample Size"].unique())
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 2, 1])
    axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    ax_right = fig.add_subplot(gs[:, 2])
    palette = {"Null": "#4C72B0", "Alternative": "#DD8452"}
    for j, (stat, title )in enumerate(
                zip(
                    ["MMD", "KLR"], 
                    [r'$\mathrm{MMD}^2(\mathbb{P}_n, \mathbb{Q}_n)$',
                    r'$\mathrm{D}_{\mathrm{KL}}({\mathcal{P}_N}_{\#}\mathcal{N}_{\mathbb{P}_n}, {\mathcal{P}_N}_{\#}\mathcal{N}_{\mathbb{Q}_n})$'
                        ])):
        ax = axes[j]
        sub = df[df["Statistic"] == stat].copy()
        sub['Std Value'] = sub.groupby(['Sample Size'])['Value'].transform(lambda x: (x - x.mean()) / x.std() )
        sns.violinplot(
            data=sub,
            y="Std Value",
            x="Sample Size",
            hue="Hypothesis",
            split=True,
            palette=palette,
            cut=10,
            inner=None,
            width=0.8,
            legend=True,
            ax=ax
        )
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(""); ax.set_xlabel("")
    for ax in axes:
        ax.legend(title='', fontsize=10)
        ax.legend_.remove()
        ax.set_xlabel("Sample Size", fontsize=11)
        if ax != axes[-1]:
            ax.set_xlabel(None)
            ax.set_xticklabels([])

    # Power curve on the right
    for stat in ['KLR', 'MMD']:
        power_means = []
        power_stds = []
        for N in Ns:
            sub_N = df[(df["Sample Size"] == N) & (df["Statistic"] == stat)]
            null_vals = sub_N[sub_N["Hypothesis"] == "Null"]["Value"]
            alt_vals = sub_N[sub_N["Hypothesis"] == "Alternative"]["Value"]
            threshold = np.percentile(null_vals, 95)
            power = np.mean(alt_vals > threshold)
            power_means.append(power)
            n_samples = len(alt_vals)
            power_std = np.sqrt(power * (1 - power) / n_samples) # Standard error for binomial proportion
            power_stds.append(power_std)
        linestyles = {"KLR": "dashdot", "MMD": (0, (1,1))}
        linewidths = {"KLR": 2, "MMD": 1.5}
        ax_right.errorbar(
            Ns, 
            power_means, 
            yerr=power_stds, 
            marker='o', 
            capsize=5,
            capthick=2,
            linewidth=linewidths[stat],
            markersize=4,
            label=f"{stat}",
            alpha=0.6,
            color='k',
            linestyle=linestyles[stat]
        )
    
    ax_right.set_title("Monte-Carlo Power", fontsize=12)
    ax_right.set_xlabel("Sample Size", fontsize=11)
    ax_right.set_ylabel("Power", fontsize=11)
    ax_right.set_ylim(0, 1.05)
    ax_right.legend()
    ax_right.grid(True, alpha=0.3)
    ax_right.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    label_map = {"MMD": "MMD", "KLR": r"$\mathrm{D}_{\mathrm{KL}}$"}
    labels = [label_map.get(l, l) for l in labels]
    axes[-1].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=2)

    handles, labels = ax_right.get_legend_handles_labels()
    labels = [label_map.get(l, l) for l in labels]
    new_handles = [
        plt.Line2D([0], [0], color='k', linestyle=linestyles.get(orig, '-'), linewidth=1.5)
        for orig in ax_right.get_legend_handles_labels()[1]
    ]
    ax_right.legend(new_handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
    plt.show()
    return fig
