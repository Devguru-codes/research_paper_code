"""Writes the LaTeX tables and inline numbers of the RSL paper straight from experiments/results/*.csv.

Every number in the paper's tables comes from here; nothing is typed by hand.
Usage: python experiments/make_paper_tables.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(HERE, 'results')
OUT = os.path.join(HERE, '..', 'paper', 'tables')
os.makedirs(OUT, exist_ok=True)

LABELS = {'plain': 'Plain CNN', 'se_gamma': r'SE ($\gamma$)', 'eca_gamma': r'ECA ($\gamma$)', 'cbam': 'CBAM',
          'gct': 'GCT', 'srm': 'SRM', 'snsm_gamma': r'SNSM ($\gamma$)', 'snsm': r'SNSM ($\gamma{+}\beta$)',
          'fc_gamma_beta': r'FC ($\gamma{+}\beta$)', 'hybridsn': 'HybridSN', 'ssftt': 'SSFTT'}
BLOCKS = ['plain', 'se_gamma', 'eca_gamma', 'cbam', 'gct', 'srm', 'snsm_gamma', 'snsm']
BASES = BLOCKS[:-1]
SHORT = {'IndianPines': 'IP', 'KSC': 'KSC', 'PaviaU': 'PU', 'Botswana': 'BW'}
CORE = ['IndianPines', 'KSC', 'PaviaU', 'Botswana']

runs = pd.read_csv(os.path.join(FINAL, 'runs.csv'))                      # random 5% splits
eff = pd.read_csv(os.path.join(FINAL, 'efficiency.csv'))
abl = pd.read_csv(os.path.join(FINAL, 'phase2_runs.csv')).query("study == 'ablation'")
dj = pd.read_csv(os.path.join(FINAL, 'phase3_disjoint_runs.csv'))         # patch-disjoint 5% splits
std = pd.read_csv(os.path.join(FINAL, 'phase5_std_runs.csv')).replace({'dataset': {'Indian': 'IndianPines'}})

# Held-out protocol per scene: the standard fixed split for IP (region-separated, but every test patch overlaps a
# training patch at 15x15), the patch-disjoint split for PU and KSC (no overlap).
LF = pd.concat([std[std.dataset == 'IndianPines'], dj[dj.dataset.isin(['PaviaU', 'KSC'])]], ignore_index=True)
LF_SETS = ['IndianPines', 'PaviaU', 'KSC']


def vals(df, ds, m, col):
    return df[(df.dataset == ds) & (df.model == m)].set_index('seed')[col]


def fmt(v, best=None):
    txt = f'{v.mean():.2f}$\\pm${v.std(ddof=1):.2f}'
    return f'\\textbf{{{txt}}}' if best is not None and np.isclose(v.mean(), best) else txt


def write(name, lines):
    with open(os.path.join(OUT, name), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def holm(p):
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def paired(df, ds, col):
    # SNSM (gamma+beta) minus each baseline on shared seeds: mean, wins, Holm-adjusted Wilcoxon p.
    ref = vals(df, ds, 'snsm', col)
    diffs = [(ref - vals(df, ds, m, col)).dropna() for m in BASES]
    p = holm(np.array([wilcoxon(d).pvalue for d in diffs]))
    return {m: (d.mean(), int((d > 0).sum()), len(d), pv) for m, d, pv in zip(BASES, diffs, p)}


# Table: random 5% splits, OA / AA on the four core scenes (best mean per column in bold)
lines = [r'\begin{tabular}{l' + 'cc' * len(CORE) + '}', r'\toprule',
         'Method & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{SHORT[d]}}}' for d in CORE) + r' \\',
         ' & ' + ' & '.join(['OA & AA'] * len(CORE)) + r' \\', r'\midrule']
best = {(d, c): max(vals(runs, d, m, c).mean() for m in LABELS if len(vals(runs, d, m, c))) for d in CORE for c in ['OA', 'AA']}
for m in LABELS:
    if m == 'fc_gamma_beta':
        lines.append(r'\midrule')
    cells = [fmt(v, best[(d, c)]) if len(v := vals(runs, d, m, c)) else '--' for d in CORE for c in ['OA', 'AA']]
    lines.append(LABELS[m] + ' & ' + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
write('main.tex', lines)

# Table: protocol comparison. Random-split AA next to leakage-free OA / AA for every block.
lines = [r'\begin{tabular}{l' + 'ccc' * len(LF_SETS) + '}', r'\toprule',
         'Block & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{SHORT[d]}}}' for d in LF_SETS) + r' \\',
         ' & ' + ' & '.join(['AA$_\\mathrm{R}$ & OA$_\\mathrm{H}$ & AA$_\\mathrm{H}$'] * len(LF_SETS)) + r' \\', r'\midrule']
cols = [(runs, 'AA'), (LF, 'OA'), (LF, 'AA')]
top = {(d, i): max(vals(df, d, m, c).mean() for m in BLOCKS) for d in LF_SETS for i, (df, c) in enumerate(cols)}
for m in BLOCKS:
    cells = [fmt(vals(df, d, m, c), top[(d, i)]) for d in LF_SETS for i, (df, c) in enumerate(cols)]
    lines.append(LABELS[m] + ' & ' + ' & '.join(cells) + r' \\')
spread = {(d, i): np.ptp([vals(df, d, m, c).mean() for m in BLOCKS]) for d in LF_SETS for i, (df, c) in enumerate(cols)}
lines += [r'\midrule', 'Spread of block means & ' + ' & '.join(f'{spread[(d, i)]:.2f}' for d in LF_SETS for i in range(3)) + r' \\',
          r'\bottomrule', r'\end{tabular}']
write('protocol.tex', lines)

# Table: paired SNSM (gamma+beta) - baseline AA differences, random vs leakage-free
P_COLS = [('IP, random', runs, 'IndianPines'), ('IP, standard', LF, 'IndianPines'),
          ('PU, disjoint', LF, 'PaviaU'), ('KSC, disjoint', LF, 'KSC')]
res = {name: paired(df, d, 'AA') for name, df, d in P_COLS}
lines = [r'\begin{tabular}{l' + 'c' * len(P_COLS) + '}', r'\toprule',
         r'SNSM ($\gamma{+}\beta$) $-$ & ' + ' & '.join(name for name, _, _ in P_COLS) + r' \\', r'\midrule']
for m in BASES:
    cells = []
    for name, _, _ in P_COLS:
        mu, w, n, pv = res[name][m]
        star = '$^{*}$' if pv < 0.05 else ''
        cells.append(f'{mu:+.2f}{star} ({w}/{n})')
    lines.append(LABELS[m] + ' & ' + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
write('paired.tex', lines)
n_sig = {name: sum(r[3] < 0.05 for r in res[name].values()) for name in res}

# Table: SNSM descriptor / initialisation ablation on IP (random 5%)
ABL = [('snsm', 'mean, var, grad (SNSM)'), ('snsm_mean', 'mean'), ('snsm_mean_var', 'mean, var'),
       ('snsm_mean_grad', 'mean, grad'), ('snsm_noinit', 'mean, var, grad; default init.')]
ref = vals(abl, 'IndianPines', 'snsm', 'AA')
lines = [r'\begin{tabular}{lcc}', r'\toprule', r'Descriptors & AA & $\Delta$AA (wins/5) \\', r'\midrule']
for m, label in ABL:
    v = vals(abl, 'IndianPines', m, 'AA')
    d = (v - ref).dropna()
    lines.append(f'{label} & {fmt(v)} & ' + ('--' if m == 'snsm' else f'{d.mean():+.2f} ({int((d > 0).sum())}/{len(d)})') + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
write('ablation.tex', lines)

# Table: efficiency (IP input: 60 features, 15x15 patch) and mean training time on the random splits
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

# Table: datasets. Sensor, band count and ground sample distance are the published scene properties;
# size, classes and labelled pixels are read from the ground truth.
SCENES = [('IndianPines', 'Indian Pines (IP)', 'AVIRIS', 200, '20'),
          ('PaviaU', 'Pavia University (PU)', 'ROSIS', 103, '1.3'),
          ('KSC', 'Kennedy Space Center (KSC)', 'AVIRIS', 176, '18'),
          ('Botswana', 'Botswana (BW)', 'Hyperion', 145, '30')]
lines = [r'\begin{tabular}{llrrrrrr}', r'\toprule',
         r'Scene & Sensor & Size (pixels) & Bands & GSD (m) & Classes & Labelled & Train/test (random) \\', r'\midrule']
for key, name, sensor, bands, gsd in SCENES:
    gt = np.load(os.path.join(FINAL, f'_gt_{key}.npy'))
    n = runs[runs.dataset == key].iloc[0]
    lines.append(f'{name} & {sensor} & ${gt.shape[0]}\\times{gt.shape[1]}$ & {bands} & {gsd} & '
                 f'{int(gt.max())} & {int((gt > 0).sum()):,} & {int(n.n_train):,}/{int(n.n_test):,} \\\\')
lines += [r'\bottomrule', r'\end{tabular}']
write('datasets.tex', lines)


# Inline numbers used in the text (as \newcommand macros)
def mac(name, value):
    return f'\\newcommand{{\\{name}}}{{{value}}}'


def thousands(x):
    return f'{int(x):,}'.replace(',', '{,}')


macros = []
for d in CORE:
    n = runs[runs.dataset == d].iloc[0]
    macros += [mac(f'ntrain{SHORT[d]}', thousands(n.n_train)), mac(f'ntest{SHORT[d]}', thousands(n.n_test))]
s = std[(std.dataset == 'IndianPines') & (std.model == 'snsm')].iloc[0]
macros += [mac('ntrainIPstd', thousands(s.n_train)), mac('ntestIPstd', thousands(s.n_test))]
for d in ['PaviaU', 'KSC']:
    t = dj[(dj.dataset == d) & (dj.model == 'snsm')]
    macros.append(mac(f'djtest{SHORT[d]}', f'{thousands(t.n_test.min())}--{thousands(t.n_test.max())}'))
    macros.append(mac(f'djclasses{SHORT[d]}', int(t.n_test_classes.min())))
for d in LF_SETS:   # mean AA drop from random to leakage-free, averaged over blocks
    drop = np.mean([vals(runs, d, m, 'AA').mean() - vals(LF, d, m, 'AA').mean() for m in BLOCKS])
    macros.append(mac(f'drop{SHORT[d]}', f'{drop:.1f}'))
    macros.append(mac(f'spreadRand{SHORT[d]}', f'{spread[(d, 0)]:.2f}'))
    macros.append(mac(f'spreadLF{SHORT[d]}', f'{spread[(d, 2)]:.2f}'))
for name, key in zip(res, ['IPrand', 'IPstd', 'PUdj', 'KSCdj']):
    macros.append(mac(f'nsig{key}', n_sig[name]))
ip_oa = [vals(runs, 'IndianPines', m, 'OA').mean() for m in BLOCKS]
macros.append(mac('ipRandOAspread', f'{np.ptp(ip_oa):.2f}'))
macros.append(mac('nseedsIP', vals(runs, 'IndianPines', 'snsm', 'AA').size))
write('numbers.tex', macros)

print('Wrote', sorted(os.listdir(OUT)))
