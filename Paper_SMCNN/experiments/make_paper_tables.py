"""Writes the LaTeX tables of the RSL paper straight from experiments/results/*.csv.

Every number in the paper's tables comes from here; nothing is typed by hand.
Usage: python experiments/make_paper_tables.py
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, 'results')
OUT = os.path.join(HERE, '..', 'paper', 'tables')

LABELS = {'plain': 'Plain CNN', 'se_gamma': r'SE ($\gamma$)', 'eca_gamma': r'ECA ($\gamma$)', 'cbam': 'CBAM', 'fc_gamma_beta': r'FC ($\gamma{+}\beta$)',
          'snsm_gamma': r'SNSM ($\gamma$)', 'snsm': r'SNSM ($\gamma{+}\beta$)',
          'hybridsn': 'HybridSN', 'ssftt': 'SSFTT'}
SHORT = {'IndianPines': 'IP', 'KSC': 'KSC', 'PaviaU': 'PU', 'Botswana': 'BW'}
CORE = ['IndianPines', 'KSC', 'PaviaU', 'Botswana']

runs = pd.read_csv(os.path.join(FINAL, 'runs.csv'))
eff = pd.read_csv(os.path.join(FINAL, 'efficiency.csv'))
os.makedirs(OUT, exist_ok=True)


def stats(ds, m, col):
    v = runs[(runs.dataset == ds) & (runs.model == m)][col].to_numpy(float)
    return (v.mean(), v.std(ddof=1)) if len(v) else None


def cell(ds, m, col, best):
    s = stats(ds, m, col)
    if s is None:
        return '--'
    txt = f'{s[0]:.2f}$\\pm${s[1]:.2f}'
    return f'\\textbf{{{txt}}}' if np.isclose(s[0], best) else txt


def write(name, lines):
    with open(os.path.join(OUT, name), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# Table: OA / AA on the four core datasets, best mean per column in bold
lines = [r'\begin{tabular}{l' + 'cc' * len(CORE) + '}', r'\toprule',
         'Method & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{SHORT[d]}}}' for d in CORE) + r' \\',
         ' & ' + ' & '.join(['OA & AA'] * len(CORE)) + r' \\', r'\midrule']
best = {(d, c): max(s[0] for m in LABELS if (s := stats(d, m, c))) for d in CORE for c in ['OA', 'AA']}
for m in LABELS:
    if m == 'hybridsn':
        lines.append(r'\midrule')
    cells = [cell(d, m, c, best[(d, c)]) for d in CORE for c in ['OA', 'AA']]
    lines.append(LABELS[m] + ' & ' + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
write('main.tex', lines)

# Table: paired same-seed differences in AA (mean, wins out of 5)
pairs = [('snsm', 'snsm_gamma', r'SNSM ($\gamma{+}\beta$) $-$ SNSM ($\gamma$)'),
         ('fc_gamma_beta', 'se_gamma', r'FC ($\gamma{+}\beta$) $-$ SE ($\gamma$)'),
         ('snsm_gamma', 'se_gamma', r'SNSM ($\gamma$) $-$ SE ($\gamma$)'),
         ('snsm_gamma', 'eca_gamma', r'SNSM ($\gamma$) $-$ ECA ($\gamma$)'),
         ('snsm', 'cbam', r'SNSM ($\gamma{+}\beta$) $-$ CBAM'),
         ('snsm', 'plain', r'SNSM ($\gamma{+}\beta$) $-$ Plain CNN')]
lines = [r'\begin{tabular}{l' + 'c' * len(CORE) + '}', r'\toprule',
         r'$\Delta$AA (wins/5) & ' + ' & '.join(SHORT[d] for d in CORE) + r' \\', r'\midrule']
for a, b, label in pairs:
    row = []
    for d in CORE:
        A = runs[(runs.dataset == d) & (runs.model == a)].set_index('seed').AA
        B = runs[(runs.dataset == d) & (runs.model == b)].set_index('seed').AA
        diff = (A - B).dropna()
        row.append(f'{diff.mean():+.2f} ({int((diff > 0).sum())}/{len(diff)})')
    lines.append(label + ' & ' + ' & '.join(row) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
write('paired.tex', lines)

# Table: efficiency (IP input: 60 features, 15x15 patch) and mean training time
lines = [r'\begin{tabular}{lrrrrr}', r'\toprule',
         r'Method & Params (M) & MMACs$^{a}$ & Latency (ms)$^{b}$ & Train IP (s) & Train PU (s) \\', r'\midrule']
for m in LABELS:
    e = eff[(eff.model == m) & (eff.dataset == 'IndianPines')].iloc[0]
    t = {d: runs[(runs.dataset == d) & (runs.model == m)].train_time_s.mean() for d in ['IndianPines', 'PaviaU']}
    tpu = '--' if np.isnan(t['PaviaU']) else f"{t['PaviaU']:.1f}"
    lines.append(f"{LABELS[m]} & {e.params / 1e6:.3f} & {e.macs_convlinear / 1e6:.1f} & "
                 f"{e.latency_ms:.2f} & {t['IndianPines']:.1f} & {tpu} \\\\")
lines += [r'\bottomrule', r'\end{tabular}']
write('efficiency.tex', lines)

# Table: datasets. Sensor, band count and ground sample distance are the published
# scene properties; size, classes and labelled pixels are read from the ground truth.
SCENES = [('IndianPines', 'Indian Pines (IP)', 'AVIRIS', 200, '20'),
          ('PaviaU', 'Pavia University (PU)', 'ROSIS', 103, '1.3'),
          ('KSC', 'Kennedy Space Center (KSC)', 'AVIRIS', 176, '18'),
          ('Botswana', 'Botswana (BW)', 'Hyperion', 145, '30'),
          ('Salinas', 'Salinas', 'AVIRIS', 204, '3.7'),
          ('WHU_Hi_HanChuan', 'WHU-Hi HanChuan', 'Headwall Nano-Hyperspec', 274, '0.109')]
lines = [r'\begin{tabular}{llrrrrrr}', r'\toprule',
         r'Scene & Sensor & Size (pixels) & Bands & GSD (m) & Classes & Labelled & Train/test \\', r'\midrule']
for key, name, sensor, bands, gsd in SCENES:
    gt = np.load(os.path.join(FINAL, f'_gt_{key}.npy'))
    n = runs[runs.dataset == key].iloc[0]
    lines.append(f'{name} & {sensor} & ${gt.shape[0]}\\times{gt.shape[1]}$ & {bands} & {gsd} & '
                 f'{int(gt.max())} & {int((gt > 0).sum()):,} & {int(n.n_train):,}/{int(n.n_test):,} \\\\')
lines += [r'\bottomrule', r'\end{tabular}']
write('datasets.tex', lines)

# Table: per-class accuracy on Indian Pines (mean over seeds), best per row in bold
PC_MODELS = ['plain', 'se_gamma', 'eca_gamma', 'cbam', 'snsm_gamma', 'snsm', 'hybridsn', 'ssftt']
names = pd.read_csv(os.path.join(FINAL, 'table_per_class_IndianPines.csv'))[['class', 'labelled_pixels']]
pcm = {}
for m in PC_MODELS:
    sub = runs[(runs.dataset == 'IndianPines') & (runs.model == m)]
    arr = np.array([[np.nan if a is None else a for a in json.loads(p)] for p in sub.per_class], dtype=float)
    pcm[m] = np.nanmean(arr, axis=0)
lines = [r'\begin{tabular}{lr' + 'r' * len(PC_MODELS) + '}', r'\toprule',
         'Class & $n$ & ' + ' & '.join(LABELS[m] for m in PC_MODELS) + r' \\', r'\midrule']
for i, row in names.iterrows():
    vals = [pcm[m][i] for m in PC_MODELS]
    top = max(vals)
    cells = [f'\\textbf{{{v:.1f}}}' if np.isclose(v, top) else f'{v:.1f}' for v in vals]
    lines.append(f"{row['class']} & {row['labelled_pixels']} & " + ' & '.join(cells) + r' \\')
lines += [r'\midrule', 'AA & & ' + ' & '.join(f'{np.mean(pcm[m]):.1f}' for m in PC_MODELS) + r' \\',
          r'\bottomrule', r'\end{tabular}']
write('perclass_ip.tex', lines)

# Inline numbers used in the text (as \newcommand macros)
def mac(name, value):
    return f'\\newcommand{{\\{name}}}{{{value}}}'

macros = []
for d in ['IndianPines', 'KSC', 'PaviaU', 'Botswana', 'Salinas', 'WHU_Hi_HanChuan']:
    key = d.replace('_', '').replace('HiHanChuan', '')
    for c in ['OA', 'AA']:
        s = stats(d, 'snsm', c)
        macros.append(mac(f'snsm{key}{c}', f'{s[0]:.2f}$\\pm${s[1]:.2f}'))
    n = runs[runs.dataset == d].iloc[0]
    macros.append(mac(f'ntrain{key}', int(n.n_train)))
    macros.append(mac(f'ntest{key}', int(n.n_test)))
hyb = runs[(runs.dataset == 'Botswana') & (runs.model == 'hybridsn')].OA
macros.append(mac('hybridBWmin', f'{hyb.min():.1f}'))
write('numbers.tex', macros)

print('Wrote', sorted(os.listdir(OUT)))
