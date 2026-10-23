"""Draws the paper's data figures from experiments/results/*.csv into paper/figures/.

Usage: python experiments/make_paper_figures.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.ndimage import maximum_filter

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, 'results')
OUT = os.path.join(HERE, '..', 'paper', 'figures')
os.makedirs(OUT, exist_ok=True)

dj = pd.read_csv(os.path.join(FINAL, 'phase3_disjoint_runs.csv'))
HIGHLIGHT, MUTED, INK, GRID = '#2a78d6', '#8a8984', '#0b0b0b', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.edgecolor': '#52514e',
                     'axes.linewidth': 0.6, 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.labelcolor': INK, 'pdf.fonttype': 42})
NAMES = {'plain': 'Plain CNN', 'se_gamma': 'SE', 'eca_gamma': 'ECA', 'cbam': 'CBAM',
         'gct': 'GCT', 'srm': 'SRM', 'snsm_gamma': r'SNSM ($\gamma$)', 'snsm': r'SNSM ($\gamma$+$\beta$)'}
BASES = ['plain', 'se_gamma', 'eca_gamma', 'cbam', 'gct', 'srm', 'snsm_gamma']


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figure: paired same-seed AA differences SNSM (gamma+beta) - baseline on the patch-disjoint splits
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for ax, (ds, title) in zip(axes, [('PaviaU', 'Pavia University (patch-disjoint)'), ('KSC', 'KSC (patch-disjoint)')]):
    g = dj[dj.dataset == ds]
    ref = g[g.model == 'snsm'].set_index('seed').AA
    for row, m in enumerate(BASES):
        d = (ref - g[g.model == m].set_index('seed').AA).dropna().to_numpy()
        y = len(BASES) - 1 - row
        ax.scatter(d, np.full(len(d), y), s=12, color=MUTED, edgecolor='white', linewidth=0.5, zorder=3)
        ax.plot([d.mean()] * 2, [y - 0.3, y + 0.3], color=HIGHLIGHT, linewidth=2, zorder=4)
    ax.axvline(0, color='#52514e', linewidth=0.6)
    ax.grid(axis='x', color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=8, color=INK)
    ax.set_xlabel(r'$\Delta$AA (points)')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(BASES)))
axes[0].set_yticklabels([NAMES[m] for m in BASES][::-1])
save(fig, 'disjoint_paired')


# Figure: how the disjoint split is built (Pavia University, seed 42).
# Same rule as split_disjoint() in phase3_disjoint_gct_srm.ipynb; the assert below checks they agree.
WINDOW, RATIO, SEED = 15, 0.05, 42
gt = np.load(os.path.join(FINAL, '_gt_PaviaU.npy'))
rng = np.random.RandomState(SEED)
train = np.zeros(gt.shape, dtype=bool)
for c in np.unique(gt[gt > 0]):
    idx = np.argwhere(gt == c)
    n = max(1, int(len(idx) * RATIO))
    centre = idx[rng.randint(len(idx))]
    sel = idx[np.argsort(((idx - centre) ** 2).sum(1), kind='stable')[:n]]
    train[sel[:, 0], sel[:, 1]] = True
near = maximum_filter(train, size=2 * (WINDOW - 1) + 1)
test = (gt > 0) & ~near
expected = dj[(dj.dataset == 'PaviaU') & (dj.seed == SEED)].n_test.iloc[0]
assert test.sum() == expected, (test.sum(), expected)

state = np.zeros(gt.shape, dtype=int)            # 0 unlabelled, 1 excluded, 2 test, 3 train
state[(gt > 0) & near] = 1
state[test] = 2
state[train] = 3
fig, ax = plt.subplots(figsize=(2.6, 4.4))
ax.imshow(state, cmap=ListedColormap(['#ffffff', '#c9c8c3', '#8a8984', HIGHLIGHT]), vmin=0, vmax=3,
          interpolation='nearest')
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color('#52514e'); s.set_linewidth(0.6)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [HIGHLIGHT, '#8a8984', '#c9c8c3']]
ax.legend(handles, ['training', 'test', 'excluded (patch overlap)'], loc='upper center',
          bbox_to_anchor=(0.5, -0.01), ncol=1, frameon=False, fontsize=7)
save(fig, 'disjoint_split_pu')

print('Wrote', sorted(os.listdir(OUT)))
