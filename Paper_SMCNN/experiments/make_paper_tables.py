"""Writes the LaTeX tables and inline numbers of the RSL paper straight from experiments/results/*.csv.

Every number in the paper's tables comes from here; nothing is typed by hand.
Usage: python experiments/make_paper_tables.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
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
SHORT = {'IndianPines': 'IP', 'KSC': 'KSC', 'PaviaU': 'PU', 'Botswana': 'BW', 'WHU_Hi_HanChuan': 'WHU'}
CORE = ['IndianPines', 'KSC', 'PaviaU', 'Botswana']

runs = pd.read_csv(os.path.join(FINAL, 'runs.csv'))                      # random 5% splits
eff = pd.read_csv(os.path.join(FINAL, 'efficiency.csv'))
abl = pd.read_csv(os.path.join(FINAL, 'phase2_runs.csv')).query("study == 'ablation'")
dj = pd.read_csv(os.path.join(FINAL, 'phase3_disjoint_runs.csv'))         # patch-disjoint 5% splits
std = pd.read_csv(os.path.join(FINAL, 'phase5_std_runs.csv')).replace({'dataset': {'Indian': 'IndianPines'}})
cnt = pd.read_csv(os.path.join(FINAL, 'phase6_counts_runs.csv'))    # random IP split, standard class counts
p8 = pd.read_csv(os.path.join(FINAL, 'phase8_runs.csv'))              # Phase 8 robustness studies
p8['model'] = p8.model.str.replace('deep_', '', regex=False)           # deeper host: same block names


def p8s(study):
    return p8[p8.study == study]


DJ_SETS = ['PaviaU', 'KSC', 'WHU_Hi_HanChuan']
# Random and patch-disjoint frames per scene (WHU-Hi with all blocks comes from Phase 8)
RAND = {'PaviaU': runs, 'KSC': runs, 'WHU_Hi_HanChuan': p8s('whu_random')}
DISJ = {'PaviaU': dj, 'KSC': dj, 'WHU_Hi_HanChuan': p8s('whu_disjoint')}
# Indian Pines protocols: random 5%, random with the standard split's class counts, standard fixed split
# (region-separated, but every test patch overlaps a training patch at 15x15).
sc = pd.read_csv(os.path.join(FINAL, 'phase7_scarcity_runs.csv'))       # only the 3 smallest classes changed
IP_PROTOCOLS = [('Random 5\\%', runs), ('Random, smallest 15', sc[sc.study == 'smallfix']),
                ('Random, 50/15', cnt), ('Standard split', std)]


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


MARGIN = 1.0   # equivalence margin (AA points) for the TOST equivalence test


def t_ci(d, level=0.95):
    # Student-t interval of the mean paired difference.
    h = t_dist.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() - h, d.mean() + h


assert np.allclose(t_ci(np.array([1.0, 2.0, 3.0, 4.0])), (2.5 - 3.182446 * 0.645497, 2.5 + 3.182446 * 0.645497))


def equivalent(d):
    # Paired TOST at alpha = 0.05: equivalent within +-MARGIN if the 90% t interval lies inside (-MARGIN, MARGIN).
    lo, hi = t_ci(d, 0.90)
    return -MARGIN < lo and hi < MARGIN


all_p = []   # raw Wilcoxon p-values of every reported comparison, for a global Benjamini-Hochberg check


def paired(df, ds, col):
    # SNSM (gamma+beta) minus each baseline on shared seeds: mean, 95% t interval, wins, n, Holm-adjusted Wilcoxon p
    # (Holm family = the seven comparisons of one scene and protocol), raw p, equivalence within +-MARGIN.
    ref = vals(df, ds, 'snsm', col)
    diffs = [(ref - vals(df, ds, m, col)).dropna() for m in BASES]
    assert all(len(d) == len(ref) for d in diffs), f'{ds}: a block is missing seeds that SNSM has'
    raw = np.array([wilcoxon(d).pvalue for d in diffs])
    p = holm(raw)
    if col == 'AA':
        all_p.extend(raw)
    return {m: (d.mean(), t_ci(d.to_numpy()), int((d > 0).sum()), len(d), pv, pr, equivalent(d.to_numpy()))
            for m, d, pv, pr in zip(BASES, diffs, p, raw)}


def bh_count(p, q=0.05):
    # Number of discoveries under Benjamini-Hochberg at false discovery rate q.
    p = np.sort(np.asarray(p))
    ok = np.nonzero(p <= q * np.arange(1, len(p) + 1) / len(p))[0]
    return 0 if len(ok) == 0 else int(ok[-1]) + 1


assert bh_count([0.001, 0.02, 0.04, 0.9]) == 2 and bh_count([0.001, 0.02, 0.03, 0.9]) == 3 and bh_count([0.5, 0.6]) == 0


# Table: random 5% splits, OA / AA on the four core scenes (best mean per column in bold).
# The paper shows the eight backbone variants; the supplement adds FC (gamma+beta), HybridSN and SSFTT.
def main_table(models, name):
    lines = [r'\begin{tabular}{l' + 'cc' * len(CORE) + '}', r'\toprule',
             'Method & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{SHORT[d]}}}' for d in CORE) + r' \\',
             ' & ' + ' & '.join(['OA & AA'] * len(CORE)) + r' \\', r'\midrule']
    best = {(d, c): max(vals(runs, d, m, c).mean() for m in models if len(vals(runs, d, m, c))) for d in CORE for c in ['OA', 'AA']}
    for m in models:
        if m == 'fc_gamma_beta':
            lines.append(r'\midrule')
        cells = [fmt(v, best[(d, c)]) if len(v := vals(runs, d, m, c)) else '--' for d in CORE for c in ['OA', 'AA']]
        lines.append(LABELS[m] + ' & ' + ' & '.join(cells) + r' \\')
    write(name, lines + [r'\bottomrule', r'\end{tabular}'])


main_table(BLOCKS, 'main.tex')
main_table(list(LABELS), 'supp_main.tex')

def block_table(cols, header, name):
    # cols: list of (dataframe, scene, metric); bold = best block mean per column; last row = range of block means
    top = [max(vals(df, d, m, c).mean() for m in BLOCKS) for df, d, c in cols]
    lines = [r'\begin{tabular}{l' + 'c' * len(cols) + '}', r'\toprule'] + header + [r'\midrule']
    for m in BLOCKS:
        lines.append(LABELS[m] + ' & ' + ' & '.join(fmt(vals(df, d, m, c), t) for (df, d, c), t in zip(cols, top)) + r' \\')
    spread = [np.ptp([vals(df, d, m, c).mean() for m in BLOCKS]) for df, d, c in cols]
    lines += [r'\midrule', 'Range of block means & ' + ' & '.join(f'{s:.2f}' for s in spread) + r' \\',
              r'\bottomrule', r'\end{tabular}']
    write(name, lines)
    return spread


# Table: Indian Pines, AA under three protocols
ip_spread = block_table([(df, 'IndianPines', 'AA') for _, df in IP_PROTOCOLS],
                        ['Block & ' + ' & '.join(n for n, _ in IP_PROTOCOLS) + r' \\'], 'protocol_ip.tex')
# Table: PU and KSC, random vs patch-disjoint
dj_cols = [(df, d, c) for d in DJ_SETS for df, c in [(RAND[d], 'AA'), (DISJ[d], 'OA'), (DISJ[d], 'AA')]]
dj_spread = block_table(dj_cols, ['Block & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{SHORT[d]}}}' for d in DJ_SETS) + r' \\',
                                  ' & ' + ' & '.join(['AA, random & OA, disjoint & AA, disjoint'] * len(DJ_SETS)) + r' \\'],
                        'protocol_dj.tex')

# Phase 7: change only the three smallest classes of a random 5% split (AA table goes to the supplement)
SC_COLS = [('IP, 5\\%', runs, 'IndianPines'), ('IP, smallest 15', sc[sc.study == 'smallfix'], 'IndianPines'),
           ('PU, 5\\%', runs, 'PaviaU'), ('PU, smallest 1/1/2', sc[sc.study == 'scarce'], 'PaviaU'),
           ('KSC, 5\\%', runs, 'KSC'), ('KSC, smallest 1/1/2', sc[sc.study == 'scarce'], 'KSC')]
sc_spread = block_table([(df, d, 'AA') for _, df, d in SC_COLS],
                        ['Block & ' + ' & '.join(n for n, _, _ in SC_COLS) + r' \\'], 'supp_scarcity.tex')

# Phase 8 AA per block (supplement)
P8_COLS = [('IP, deeper', 'deep_ip_random', 'IndianPines'), ('KSC disj., deeper', 'deep_ksc_disjoint', 'KSC'),
           ('KSC disj., train PCA', 'trainpca_ksc_disjoint', 'KSC'), ('PU, hardest', 'hard_pu', 'PaviaU'),
           ('KSC, hardest', 'hard_ksc', 'KSC'), ('IP, lr 3e-4', 'lr3e-4_ip_random', 'IndianPines'),
           ('IP, lr 3e-3', 'lr3e-3_ip_random', 'IndianPines')]
block_table([(p8s(st), d, 'AA') for _, st, d in P8_COLS],
            ['Block & ' + ' & '.join(n for n, _, _ in P8_COLS) + r' \\'], 'supp_robust.tex')

# Paired SNSM (gamma+beta) - block differences: one row per setting, one column per block.
# * = Holm-corrected Wilcoxon p < 0.05 within the row; dagger = equivalent within +-MARGIN (paired TOST).
RULE = (None, None, None)   # a horizontal rule between groups of settings
SETTINGS = [('IP, random 5\\%', runs, 'IndianPines'), ('IP, smallest 15', sc[sc.study == 'smallfix'], 'IndianPines'),
            ('IP, random 50/15', cnt, 'IndianPines'), ('IP, standard', std, 'IndianPines'), RULE,
            ('PU, smallest 1/1/2', sc[sc.study == 'scarce'], 'PaviaU'),
            ('KSC, smallest 1/1/2', sc[sc.study == 'scarce'], 'KSC'),
            ('PU, hardest 1/1/2', p8s('hard_pu'), 'PaviaU'), ('KSC, hardest 1/1/2', p8s('hard_ksc'), 'KSC'), RULE,
            ('WHU, random 5\\%', p8s('whu_random'), 'WHU_Hi_HanChuan'),
            ('PU, patch-disjoint', dj, 'PaviaU'), ('KSC, patch-disjoint', dj, 'KSC'),
            ('WHU, patch-disjoint', p8s('whu_disjoint'), 'WHU_Hi_HanChuan'), RULE,
            ('IP, random 5\\%, deeper host', p8s('deep_ip_random'), 'IndianPines'),
            ('KSC, patch-disjoint, deeper host', p8s('deep_ksc_disjoint'), 'KSC'),
            ('KSC, patch-disjoint, train-only PCA', p8s('trainpca_ksc_disjoint'), 'KSC'),
            ('IP, random 5\\%, lr $3\\times10^{-4}$', p8s('lr3e-4_ip_random'), 'IndianPines'),
            ('IP, random 5\\%, lr $3\\times10^{-3}$', p8s('lr3e-3_ip_random'), 'IndianPines')]
SHORT_BLOCK = {'plain': 'Plain', 'se_gamma': 'SE', 'eca_gamma': 'ECA', 'cbam': 'CBAM', 'gct': 'GCT', 'srm': 'SRM',
               'snsm_gamma': r'SNSM ($\gamma$)'}
res = {name: paired(df, d, 'AA') for name, df, d in SETTINGS if name}
res_oa = {name: paired(df, d, 'OA') for name, df, d in SETTINGS if name}


def mark(v):
    return ('$^{*}$' if v[4] < 0.05 else '') + ('$^{\\dagger}$' if v[6] else '')


def paired_table(r, cell, name):
    lines = [r'\begin{tabular}{l' + 'c' * len(BASES) + 'c}', r'\toprule',
             'Setting & ' + ' & '.join(SHORT_BLOCK[m] for m in BASES) + r' & Seeds \\', r'\midrule']
    for s, _, _ in SETTINGS:
        if s is None:
            lines.append(r'\midrule')
            continue
        lines.append(s + ' & ' + ' & '.join(cell(r[s][m]) for m in BASES) + f' & {r[s][BASES[0]][3]}' + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write(name, lines)


paired_table(res, lambda v: f'{v[0]:+.2f}{mark(v)}', 'paired.tex')
paired_table(res, lambda v: f'{v[0]:+.2f} [{v[1][0]:+.1f}, {v[1][1]:+.1f}]', 'supp_paired_ci.tex')
paired_table(res_oa, lambda v: f'{v[0]:+.2f}{mark(v)}', 'supp_paired_oa.tex')
n_sig = {s: sum(v[4] < 0.05 for v in res[s].values()) for s in res}
n_eq = {s: sum(v[6] for v in res[s].values()) for s in res}
ci_width = {s: np.median([v[1][1] - v[1][0] for v in res[s].values()]) for s in res}

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
         r'Method & Params (M) & MMACs & Latency (ms) & Train IP (s) & Train PU (s) \\', r'\midrule']
for m in LABELS:
    e = eff[(eff.model == m) & (eff.dataset == 'IndianPines')].iloc[0]
    t = {d: runs[(runs.dataset == d) & (runs.model == m)].train_time_s.mean() for d in ['IndianPines', 'PaviaU']}
    tpu = '--' if np.isnan(t['PaviaU']) else f"{t['PaviaU']:.1f}"
    lines.append(f"{LABELS[m]} & {e.params / 1e6:.3f} & {e.macs_convlinear / 1e6:.1f} & "
                 f"{e.latency_ms:.2f} & {t['IndianPines']:.1f} & {tpu} \\\\")
lines += [r'\bottomrule', r'\end{tabular}']
write('supp_efficiency.tex', lines)

# Table: datasets. Sensor, band count and ground sample distance are the published scene properties;
# size, classes and labelled pixels are read from the ground truth.
SCENES = [('IndianPines', 'Indian Pines (IP)', 'AVIRIS', 200, '20'),
          ('PaviaU', 'Pavia University (PU)', 'ROSIS', 103, '1.3'),
          ('KSC', 'Kennedy Space Center (KSC)', 'AVIRIS', 176, '18'),
          ('Botswana', 'Botswana (BW)', 'Hyperion', 145, '30'),
          ('WHU_Hi_HanChuan', 'WHU-Hi HanChuan (WHU)', 'Headwall Nano-Hyperspec', 274, '0.109')]
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
w = p8s('whu_random').iloc[0]
macros += [mac('ntrainWHU', thousands(w.n_train)), mac('ntestWHU', thousands(w.n_test))]
s = std[(std.dataset == 'IndianPines') & (std.model == 'snsm')].iloc[0]
macros += [mac('ntrainIPstd', thousands(s.n_train)), mac('ntestIPstd', thousands(s.n_test))]
s = cnt[(cnt.dataset == 'IndianPines') & (cnt.model == 'snsm')].iloc[0]
macros += [mac('ntrainIPcnt', thousands(s.n_train)), mac('ntestIPcnt', thousands(s.n_test))]
for d in ['PaviaU', 'KSC']:
    t = dj[(dj.dataset == d) & (dj.model == 'snsm')]
    macros.append(mac(f'djtest{SHORT[d]}', f'{thousands(t.n_test.min())}--{thousands(t.n_test.max())}'))
    macros.append(mac(f'djclasses{SHORT[d]}', int(t.n_test_classes.min())))
for i, d in enumerate(DJ_SETS):   # mean AA drop from random to patch-disjoint, averaged over blocks
    drop = np.mean([vals(RAND[d], d, m, 'AA').mean() - vals(DISJ[d], d, m, 'AA').mean() for m in BLOCKS])
    macros.append(mac(f'drop{SHORT[d]}', f'{drop:.1f}'))
    macros.append(mac(f'spreadDJ{SHORT[d]}', f'{dj_spread[3 * i + 2]:.2f}'))
for (name, _), key in zip(IP_PROTOCOLS, ['Rand', 'Small', 'Cnt', 'Std']):
    macros.append(mac(f'spreadIP{key}', f'{ip_spread[[n for n, _ in IP_PROTOCOLS].index(name)]:.2f}'))
for name, key in [('IP, random 5\\%', 'IPrand'), ('IP, random 50/15', 'IPcnt'), ('IP, standard', 'IPstd'),
                  ('PU, patch-disjoint', 'PUdj'), ('KSC, patch-disjoint', 'KSCdj'), ('IP, smallest 15', 'IPsmall'),
                  ('PU, smallest 1/1/2', 'PUscarce'), ('KSC, smallest 1/1/2', 'KSCscarce')]:
    macros.append(mac(f'nsig{key}', n_sig[name]))
    macros.append(mac(f'ciw{key}', f'{ci_width[name]:.1f}'))
for (name, _, _), key, sp in zip(SC_COLS, ['IPfive', 'IPsmall', 'PUfive', 'PUscarce', 'KSCfive', 'KSCscarce'], sc_spread):
    macros.append(mac(f'spreadSC{key}', f'{sp:.2f}'))
# Numbers and claims written in the paper's text. Numbers become macros; qualitative claims (which blocks
# are significant, lowest, highest) are asserted, so a data change breaks this script instead of the text.
def sig(r):
    return {m for m, v in r.items() if v[4] < 0.05}


def rng(a, b, d=2):
    return f'{a:.{d}f}--{b:.{d}f}'


r = res['IP, random 5\\%']
assert sig(r) == {'plain', 'se_gamma', 'eca_gamma', 'cbam', 'gct'}, sig(r)
g = sorted(r[m][0] for m in sig(r))
macros += [mac('ipRandGain', rng(g[0], g[-1])), mac('ipRandGainShort', rng(g[0], g[-1], 1)),
           mac('ipRandGainInt', rng(g[0], g[-1], 0))]
other = max(np.ptp([vals(runs, d, m, 'AA').mean() for m in BLOCKS]) for d in ['PaviaU', 'KSC', 'Botswana'])
macros.append(mac('randOtherSpread', f'{other:.2f}'))
P1 = ['plain', 'se_gamma', 'eca_gamma', 'cbam', 'fc_gamma_beta', 'snsm_gamma', 'snsm']   # variants also run before the KSC fix
k = [vals(runs, 'KSC', m, 'AA').mean() for m in P1]
assert min(k) > 99                         # "raises seven convolutional variants ... to over 99% AA"
macros.append(mac('kscFixedRange', rng(min(k), max(k), 1)))

ipr = runs[(runs.dataset == 'IndianPines') & runs.seed.isin(range(42, 57))]
pc = {m: np.nanmean(np.array([[np.nan if a is None else a for a in json.loads(p)]
                              for p in ipr[ipr.model == m].per_class], dtype=float), axis=0) for m in BLOCKS}
DISP, REST = ['srm', 'snsm_gamma', 'snsm'], ['plain', 'se_gamma', 'eca_gamma', 'cbam', 'gct']
for cls, key in [(6, 'Gpm'), (8, 'Oats')]:   # Grass-pasture-mowed, Oats: one training pixel each
    macros += [mac(f'pc{key}Disp', rng(min(pc[m][cls] for m in DISP), max(pc[m][cls] for m in DISP), 1)),
               mac(f'pc{key}Rest', rng(min(pc[m][cls] for m in REST), max(pc[m][cls] for m in REST), 1))]
macros.append(mac('pcAlfalfaSpread', f'{np.ptp([pc[m][0] for m in BLOCKS]):.1f}'))
rest13 = [np.mean(np.delete(pc[m], [0, 6, 8])) for m in BLOCKS]
assert np.ptp(rest13) < 1, rest13          # "differs by less than one point"

ci_max = max(max(abs(v[1][0]), abs(v[1][1])) for v in res['IP, random 50/15'].values())
assert sig(res['IP, random 50/15']) == set()
macros.append(mac('cntCImax', f'{np.ceil(ci_max * 10) / 10:.1f}'))
assert sig(res['IP, standard']) == {'eca_gamma'}
macros.append(mac('stdEcaDiff', f"{res['IP, standard']['eca_gamma'][0]:.2f}"))

r = res['IP, smallest 15']
assert sig(r) == set(BASES)
g = sorted(v[0] for v in r.values())
macros += [mac('smallGain', rng(g[0], g[-1])), mac('spreadSCIPsmallShort', f'{sc_spread[1]:.1f}')]
assert {m for m, v in r.items() if v[0] >= 0.5} == {'cbam'}   # "exceeds 0.5 only against CBAM"
for d, key in [('PaviaU', 'PU'), ('KSC', 'KSC')]:
    s = sc[(sc.study == 'scarce') & (sc.dataset == d)]
    drop = np.mean([vals(runs, d, m, 'AA').mean() - s[s.model == m].AA.mean() for m in BLOCKS])
    macros.append(mac(f'dropScarce{key}', f'{drop:.0f}'))
means = sc[(sc.study == 'scarce') & (sc.dataset == 'PaviaU')].groupby('model').AA.mean()
assert means.idxmin() == 'snsm' and sig(res['PU, smallest 1/1/2']) == {'eca_gamma', 'snsm_gamma'}
means = sc[(sc.study == 'scarce') & (sc.dataset == 'KSC')].groupby('model').AA.mean()
assert means.idxmin() == 'srm' and sig(res['KSC, smallest 1/1/2']) == set()

drops = []
for d in DJ_SETS:
    oa = [vals(DISJ[d], d, m, 'OA').mean() for m in BLOCKS]
    macros.append(mac(f'djOA{SHORT[d]}', rng(min(oa), max(oa), 0)))
    drops.append(np.mean([vals(RAND[d], d, m, 'AA').mean() - vals(DISJ[d], d, m, 'AA').mean() for m in BLOCKS]))
assert min(vals(RAND[d], d, m, 'AA').mean() - vals(DISJ[d], d, m, 'AA').mean()
           for d in DJ_SETS for m in BLOCKS) > 30   # "every block lost more than 30 AA points"
macros.append(mac('dropRange', rng(min(drops), max(drops), 0)))
sd = [vals(DISJ[d], d, m, 'AA').std(ddof=1) for d in DJ_SETS for m in BLOCKS]
macros.append(mac('djStdRange', rng(min(sd), max(sd), 1)))
assert sig(res['PU, patch-disjoint']) == set() and sig(res['KSC, patch-disjoint']) == {'plain', 'gct', 'srm'}
ksc = dj[dj.dataset == 'KSC'].groupby('model').AA.mean()
assert ksc.idxmax() == 'se_gamma' and ksc.idxmin() == 'srm'
assert 'srm' not in sig(res['IP, random 5\\%'])                         # "SRM, tied best on random 5% IP"
assert all(3 <= ci_width[c] <= 4.5 for c in ['PU, patch-disjoint', 'KSC, patch-disjoint'])   # "about two points"

a = vals(abl, 'IndianPines', 'snsm', 'AA')
macros.append(mac('ablMeanDrop', f"{(a - vals(abl, 'IndianPines', 'snsm_mean', 'AA')).mean():.2f}"))
assert ((vals(abl, 'IndianPines', 'snsm_noinit', 'AA') - a) < 0).all()   # "lower on all five seeds"

e = eff[eff.dataset == 'IndianPines'].set_index('model')
extra = (e.params - e.params['plain']).drop('plain')
blk = extra[[m for m in BLOCKS if m != 'plain']]
assert blk.idxmin() == 'eca_gamma' and blk.idxmax() == 'se_gamma'
macros += [mac('paramsEca', int(blk.min())), mac('paramsSe', thousands(blk.max())), mac('paramsSnsm', int(extra['snsm'])),
           mac('latencyRatio', f"{e.latency_ms['snsm'] / e.latency_ms['plain']:.1f}"),
           mac('trainRatio', f"{ipr[ipr.model == 'snsm'].train_time_s.mean() / ipr[ipr.model == 'plain'].train_time_s.mean():.1f}")]

# Phase 8 claims: WHU-Hi, deeper host, learning rate, train-only PCA, hardest-class scarcity
assert n_eq['WHU, random 5\\%'] == len(BASES)                      # "all equivalent on WHU-Hi random"
whu = p8s('whu_disjoint').groupby('model').AA.mean()
assert whu.idxmin() == 'snsm' and sig(res['WHU, patch-disjoint']) == set()   # "SNSM lowest, nothing significant"
for s in ['PU, hardest 1/1/2', 'KSC, hardest 1/1/2', 'IP, random 5\\%, deeper host', 'KSC, patch-disjoint, deeper host']:
    assert sig(res[s]) == set(), s                                   # "no significant difference"
hard = max(abs(v[0]) for s in ['PU, hardest 1/1/2', 'KSC, hardest 1/1/2'] for v in res[s].values())
deep = p8s('deep_ip_random').groupby('model').AA.mean()
assert deep.idxmax() != 'snsm'                                       # "SNSM is no longer the best block"
assert sig(res['KSC, patch-disjoint, train-only PCA']) == {'cbam'}
tp = p8s('trainpca_ksc_disjoint').groupby('model').AA.mean()
sp = dj[dj.dataset == 'KSC'].groupby('model').AA.mean()[BLOCKS]
lo, hi = (sorted(v[0] for v in res[f'IP, random 5\\%, lr $3\\times10^{{-{e}}}$'].values()) for e in (4, 3))
assert min(lo) > 5 and max(hi) < 2.5                                 # gap grows at lr 3e-4, shrinks at 3e-3
macros += [mac('hardMax', f'{hard:.1f}'), mac('deepIPspread', f'{np.ptp(deep):.2f}'),
           mac('trainPcaRange', rng(tp.min(), tp.max(), 1)), mac('scenePcaRange', rng(sp.min(), sp.max(), 1)),
           mac('lrLowGain', rng(lo[0], lo[-1], 1)), mac('lrHighMax', f'{hi[-1]:.1f}'),
           mac('spreadWHUrand', f'{dj_spread[6]:.2f}'),
           mac('whuDisjAA', rng(whu.min(), whu.max(), 1))]

# Equivalence (TOST, +-MARGIN) and global false-discovery-rate claims
for s in ['IP, random 50/15', 'IP, standard', 'IP, smallest 15']:
    assert n_eq[s] == len(BASES), (s, n_eq[s])            # "all seven comparisons equivalent within +-1"
for s in ['PU, patch-disjoint', 'KSC, patch-disjoint', 'WHU, patch-disjoint']:
    assert n_eq[s] == 0, (s, n_eq[s])                      # "equivalence cannot be shown on patch-disjoint splits"
assert n_eq['IP, random 5\\%'] == 0
seed_sd = np.median([vals(runs, 'IndianPines', m, 'AA').std(ddof=1) for m in BLOCKS])
assert MARGIN < seed_sd / 2, seed_sd                       # "less than half the seed-to-seed standard deviation"
macros += [mac('margin', f'{MARGIN:.0f}'), mac('seedSdIP', f'{seed_sd:.1f}'),
           mac('bhSig', bh_count(all_p)), mac('bhTotal', len(all_p)),
           mac('holmSig', sum(n_sig.values())), mac('nSettings', len(res))]
for s, key in [('PU, smallest 1/1/2', 'PUscarce'), ('KSC, smallest 1/1/2', 'KSCscarce')]:
    macros.append(mac(f'neq{key}', n_eq[s]))

ip_oa = [vals(runs, 'IndianPines', m, 'OA').mean() for m in BLOCKS]
macros.append(mac('ipRandOAspread', f'{np.ptp(ip_oa):.2f}'))
macros.append(mac('nseedsIP', vals(runs, 'IndianPines', 'snsm', 'AA').size))
write('numbers.tex', macros)

print('Wrote', sorted(os.listdir(OUT)))
