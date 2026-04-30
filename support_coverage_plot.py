import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

datasets = ['D1', 'D2', 'D3', 'D4', 'D5']
models   = ['ARF', 'GenForest', 'ForestFlow', 'GaussianCopula', 'CTGAN', 'TabDDPM', 'TVAE']

alpha = {
    'ARF':            [0.986, 0.964, 0.998, 0.945, 1.000],
    'GenForest':      [0.999, 0.973, 1.000, 0.980, 1.000],
    'ForestFlow':     [0.999, 0.682, 0.998, 0.799, 1.000],
    'GaussianCopula': [0.996, 0.982, 0.953, 0.792, 0.997],
    'CTGAN':          [0.996, 0.956, 0.991, 0.848, 1.000],
    'TabDDPM':        [0.937, 0.474, 0.999, 0.897, 0.744],
    'TVAE':           [0.996, 0.989, 0.999, 0.972, 1.000],
}
beta = {
    'ARF':            [0.945, 0.726, 0.992, 0.940, 0.893],
    'GenForest':      [0.970, 0.715, 0.999, 0.876, 0.670],
    'ForestFlow':     [0.919, 0.073, 0.980, 0.707, 0.958],
    'GaussianCopula': [0.920, 0.245, 0.723, 0.772, 0.720],
    'CTGAN':          [0.863, 0.179, 0.807, 0.808, 0.626],
    'TabDDPM':        [0.503, 0.282, 0.998, 0.949, 0.009],
    'TVAE':           [0.687, 0.529, 0.838, 0.736, 0.788],
}

colors = {
    'ARF': '#2ca02c', 'GenForest': '#52c752', 'ForestFlow': '#98df8a',
    'GaussianCopula': '#7f7f7f', 'CTGAN': '#d62728',
    'TabDDPM': '#ff7f0e', 'TVAE': '#e8a090',
}
linestyles = {m: '--' if m == 'GaussianCopula' else '-' for m in models}
markers = {
    'ARF': 'o', 'GenForest': 's', 'ForestFlow': '^',
    'GaussianCopula': 'D', 'CTGAN': 'o', 'TabDDPM': 's', 'TVAE': '^',
}

x = np.arange(len(datasets))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.patch.set_alpha(0.0)

for ax, data, ylabel, title in [
    (axes[0], alpha, r'$\alpha$-precision', r'$\alpha$-Precision (higher is better)'),
    (axes[1], beta,  r'$\beta$-recall',     r'$\beta$-Recall (higher is better)'),
]:
    ax.set_facecolor('none')
    for m in models:
        ax.plot(x, data[m], color=colors[m], linestyle=linestyles[m],
                marker=markers[m], linewidth=1.6, markersize=5, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles = [plt.Line2D([0],[0], color=colors[m], linestyle=linestyles[m],
           marker=markers[m], markersize=5, label=m) for m in models]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.08))

tree_patch  = mpatches.Patch(color='#2ca02c', label='Tree-based (ARF, GenForest, ForestFlow)')
bench_patch = mpatches.Patch(color='#7f7f7f', label='Benchmark (GaussianCopula)')
dl_patch    = mpatches.Patch(color='#d62728', label='Deep learning (CTGAN, TabDDPM, TVAE)')
fig.legend(handles=[tree_patch, bench_patch, dl_patch], loc='lower center', ncol=3,
           fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.18), handlelength=1.2)

plt.tight_layout()
plt.savefig('support_coverage.png', dpi=180, bbox_inches='tight',
            facecolor='none', transparent=True)