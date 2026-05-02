import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# Mean KS (numeric fidelity) per model per dataset [D1, D2, D3, D4, D5]
ks = {
    'ARF':            [0.102, 0.104, 0.037, 0.143, 0.087],
    'GenForest':      [0.063, 0.080, 0.012, 0.095, 0.071],
    'ForestFlow':     [0.100, 0.290, 0.028, 0.072, 0.064],
    'GaussianCopula': [0.149, 0.195, 0.147, 0.234, 0.053],
    'CTGAN':          [0.180, 0.204, 0.081, 0.168, 0.313],
    'TabDDPM':        [0.462, 0.480, 0.134, 0.025, 0.543],
    'TVAE':           [0.144, 0.118, 0.095, 0.270, 0.370],
}

# Mean TV (categorical fidelity) per model per dataset
tv_cat = {
    'ARF':            [0.014, 0.034, 0.005, 0.018, 0.031],
    'GenForest':      [0.035, 0.063, 0.006, 0.117, 0.089],
    'ForestFlow':     [0.151, 0.284, 0.013, 0.190, 0.213],
    'GaussianCopula': [0.013, 0.034, 0.008, 0.015, 0.032],
    'CTGAN':          [0.097, 0.065, 0.079, 0.110, 0.062],
    'TabDDPM':        [0.207, 0.318, 0.034, 0.043, 0.449],
    'TVAE':           [0.201, 0.138, 0.037, 0.167, 0.233],
}

# Mean TV (ordinal fidelity) per model per dataset
# None = no ordinal features in that dataset (Dataset 4)
tv_ord = {
    'ARF':            [0.018, 0.013, 0.007, None, 0.028],
    'GenForest':      [0.035, 0.009, 0.010, None, 0.062],
    'ForestFlow':     [0.092, 0.106, 0.018, None, 0.115],
    'GaussianCopula': [0.021, 0.011, 0.004, None, 0.033],
    'CTGAN':          [0.109, 0.041, 0.074, None, 0.069],
    'TabDDPM':        [0.179, 0.069, 0.021, None, 0.198],
    'TVAE':           [0.259, 0.030, 0.052, None, 0.469],
}

datasets = ['D1', 'D2', 'D3', 'D4', 'D5']

# Model order: tree-based, benchmark, deep learning
models = ['ARF', 'GenForest', 'ForestFlow', 'GaussianCopula', 'CTGAN', 'TabDDPM', 'TVAE']

# ── Style ─────────────────────────────────────────────────────────────────────
colors = {
    'ARF':            '#2C6E49',
    'GenForest':      '#52B788',
    'ForestFlow':     '#95D5B2',
    'GaussianCopula': '#6B6B6B',
    'CTGAN':          '#C1440E',
    'TabDDPM':        '#E07A5F',
    'TVAE':           '#F2B880',
}

markers = {
    'ARF':            'o',
    'GenForest':      's',
    'ForestFlow':     '^',
    'GaussianCopula': 'D',
    'CTGAN':          'o',
    'TabDDPM':        's',
    'TVAE':           '^',
}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
fig.patch.set_alpha(0)

metric_data   = [ks,    tv_cat,    tv_ord]
metric_titles = [
    'Numeric Fidelity\n(Mean KS, lower is better)',
    'Categorical Fidelity\n(Mean TV, lower is better)',
    'Ordinal Fidelity\n(Mean TV, lower is better)',
]
metric_ylims  = [(0, 0.65), (0, 0.52), (0, 0.52)]

x = np.arange(len(datasets))

for ax, data, title, ylim in zip(axes, metric_data, metric_titles, metric_ylims):
    ax.set_facecolor('none')
    ax.yaxis.grid(True, color='#DDDDDD', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    for m in models:
        vals   = data[m]
        x_plot = [x[i] for i, v in enumerate(vals) if v is not None]
        y_plot = [v     for v        in vals          if v is not None]

        ax.plot(
            x_plot, y_plot,
            color=colors[m],
            marker=markers[m],
            linewidth=2.2 if m == 'GaussianCopula' else 1.6,
            linestyle='--' if m == 'GaussianCopula' else '-',
            markersize=6,
            alpha=0.85,
            zorder=3,
            label=m,
        )

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=8, color='#222222')
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')

# ── Legends ───────────────────────────────────────────────────────────────────
# Per-model line legend (bottom)
handles = [
    plt.Line2D(
        [0], [0],
        color=colors[m],
        marker=markers[m],
        linewidth=1.5,
        markersize=5,
        label=m,
        linestyle='--' if m == 'GaussianCopula' else '-',
    )
    for m in models
]
fig.legend(
    handles=handles,
    loc='lower center',
    ncol=7,
    fontsize=8,
    frameon=False,
    bbox_to_anchor=(0.5, -0.04),
)

# Paradigm legend (top)
paradigm_patches = [
    mpatches.Patch(color='#2C6E49', label='Tree-based (ARF, GenForest, ForestFlow)'),
    mpatches.Patch(color='#6B6B6B', label='Benchmark (GaussianCopula)'),
    mpatches.Patch(color='#C1440E', label='Deep learning (CTGAN, TabDDPM, TVAE)'),
]
fig.legend(
    handles=paradigm_patches,
    loc='upper center',
    ncol=3,
    fontsize=8,
    frameon=False,
    bbox_to_anchor=(0.5, 1.03),
)

plt.tight_layout(rect=[0, 0.06, 1, 1])

# ── Save ──────────────────────────────────────────────────────────────────────
output_pdf = 'figures/marginal_fidelity_plot.pdf'
output_png = 'figures/marginal_fidelity_plot.png'

import os
os.makedirs('figures', exist_ok=True)

plt.savefig(output_pdf, bbox_inches='tight', dpi=200, transparent=True)
plt.savefig(output_png, bbox_inches='tight', dpi=200, transparent=True)
print(f"Saved to {output_pdf} and {output_png}")