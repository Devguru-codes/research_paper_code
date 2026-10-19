"""Draws the paper's data figures from experiments/results/*.csv into paper/figures/.

Usage: python experiments/make_paper_figures.py
"""
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, 'results')
OUT = os.path.join(HERE, '..', 'paper', 'figures')
os.makedirs(OUT, exist_ok=True)
shutil.copy(os.path.join(FINAL, 'classification_map_IndianPines.pdf'), OUT)  # keeps paper/ self-contained

runs = pd.read_csv(os.path.join(FINAL, 'runs.csv'))
HIGHLIGHT, MUTED, INK, GRID = '#2a78d6', '#8a8984', '#0b0b0b', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.edgecolor': '#52514e',
                     'axes.linewidth': 0.6, 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.labelcolor': INK, 'pdf.fonttype': 42})
NAMES = {'plain': 'Plain CNN', 'se_gamma': r'SE ($\gamma$)', 'eca_gamma': r'ECA ($\gamma$)', 'cbam': 'CBAM',
         'fc_gamma_beta': r'FC ($\gamma$+$\beta$)', 'snsm_gamma': r'SNSM ($\gamma$)',
         'snsm': r'SNSM ($\gamma$+$\beta$)', 'hybridsn': 'HybridSN', 'ssftt': 'SSFTT'}
SCENES = [('IndianPines', 'Indian Pines'), ('KSC', 'Kennedy Space Center')]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# Figure: paired same-seed AA differences (dots = seeds, bar = mean)
PAIRS = [('snsm', 'snsm_gamma', r'SNSM: $\gamma$+$\beta$ $-$ $\gamma$ only'),
         ('snsm_gamma', 'se_gamma', r'SNSM ($\gamma$) $-$ SE ($\gamma$)'),
         ('snsm_gamma', 'eca_gamma', r'SNSM ($\gamma$) $-$ ECA ($\gamma$)'),
         ('snsm', 'cbam', r'SNSM ($\gamma$+$\beta$) $-$ CBAM'),
         ('snsm', 'plain', r'SNSM ($\gamma$+$\beta$) $-$ Plain CNN')]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
for ax, (ds, title) in zip(axes, SCENES):
    for row, (a, b, _) in enumerate(PAIRS):
        A = runs[(runs.dataset == ds) & (runs.model == a)].set_index('seed').AA
        B = runs[(runs.dataset == ds) & (runs.model == b)].set_index('seed').AA
        d = (A - B).dropna().to_numpy()
        y = len(PAIRS) - 1 - row
        ax.scatter(d, np.full(len(d), y), s=16, color=MUTED, edgecolor='white', linewidth=0.6, zorder=3)
        ax.plot([d.mean()] * 2, [y - 0.28, y + 0.28], color=HIGHLIGHT, linewidth=2, zorder=4)
    ax.axvline(0, color='#52514e', linewidth=0.6)
    ax.grid(axis='x', color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=8, color=INK)
    ax.set_xlabel(r'$\Delta$AA (percentage points)')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(PAIRS)))
axes[0].set_yticklabels([p[2] for p in PAIRS][::-1])
save(fig, 'paired_differences')

# Figure: AA mean +- std over seeds per model (rows), SNSM highlighted
ORDER = ['plain', 'se_gamma', 'eca_gamma', 'cbam', 'fc_gamma_beta', 'snsm_gamma', 'snsm', 'hybridsn', 'ssftt']
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
for ax, (ds, title) in zip(axes, SCENES):
    for row, m in enumerate(ORDER):
        v = runs[(runs.dataset == ds) & (runs.model == m)].AA
        if not len(v):
            continue
        y = len(ORDER) - 1 - row
        color = HIGHLIGHT if m in ('snsm', 'snsm_gamma') else MUTED
        ax.errorbar(v.mean(), y, xerr=v.std(ddof=1), fmt='o', ms=4.5, color=color, ecolor=color,
                    elinewidth=1, capsize=2, mec='white', mew=0.6, zorder=3)
    ax.grid(axis='x', color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=8, color=INK)
    ax.set_xlabel('AA (%), mean $\pm$ std over five seeds')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(ORDER)))
axes[0].set_yticklabels([NAMES[m] for m in ORDER][::-1])
save(fig, 'aa_by_model')

print('Wrote', sorted(os.listdir(OUT)))
