"""Writes the LaTeX tables and inline numbers of the RSL paper straight from experiments/results/*.csv.

Every number in the paper's tables comes from here; nothing is typed by hand.
Usage: python experiments/make_paper_tables.py
"""
import json
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
cnt = pd.read_csv(os.path.join(FINAL, 'phase6_counts_runs.csv'))    # random IP split, standard class counts
DJ_SETS = ['PaviaU', 'KSC']
# Indian Pines protocols: random 5%, random with the standard split's class counts, standard fixed split
# (region-separated, but every test patch overlaps a training patch at 15x15).
IP_PROTOCOLS = [('Random 5\\%', runs), ('Random, 50/15 per class', cnt), ('Standard split', std)]


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


def boot_ci(d, n=10000):
    # 95% percentile bootstrap interval of the mean paired difference (fixed generator, reproducible).
    m = np.random.RandomState(0).choice(d, size=(n, len(d)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


_lo, _hi = boot_ci(np.array([1.0, 2.0, 3.0, 4.0]))
assert _lo <= 2.5 <= _hi and 1.0 <= _lo and _hi <= 4.0


def paired(df, ds, col):
    # SNSM (gamma+beta) minus each baseline on shared seeds: mean, 95% CI, wins, n, Holm-adjusted Wilcoxon p
    # (Holm family = the seven comparisons of one scene and protocol).
    ref = vals(df, ds, 'snsm', col)
    diffs = [(ref - vals(df, ds, m, col)).dropna() for m in BASES]
    assert all(len(d) == len(ref) for d in diffs), f'{ds}: a block is missing seeds that SNSM has'
    p = holm(np.array([wilcoxon(d).pvalue for d in diffs]))
    return {m: (d.mean(), boot_ci(d.to_numpy()), int((d > 0).sum()), len(d), pv) for m, d, pv in zip(BASES, diffs, p)}


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
dj_cols = [(df, d, c) for d in DJ_SETS for df, c in [(runs, 'AA'), (dj, 'OA'), (dj, 'AA')]]
dj_spread = block_table(dj_cols, ['Block & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{SHORT[d]}}}' for d in DJ_SETS) + r' \\',
                                  ' & ' + ' & '.join(['AA, random & OA, disjoint & AA, disjoint'] * len(DJ_SETS)) + r' \\'],
                        'protocol_dj.tex')

# Table: paired SNSM (gamma+beta) - block AA differences with 95% bootstrap CI; * = Holm-adjusted p < 0.05
P_COLS = [('IP random 5\\%', runs, 'IndianPines'), ('IP random 50/15', cnt, 'IndianPines'),
          ('IP standard', std, 'IndianPines'), ('PU disjoint', dj, 'PaviaU'), ('KSC disjoint', dj, 'KSC')]
res = {name: paired(df, d, 'AA') for name, df, d in P_COLS}
lines = [r'\begin{tabular}{l' + 'c' * len(P_COLS) + '}', r'\toprule',
         r'SNSM ($\gamma{+}\beta$) $-$ & ' + ' & '.join(name for name, _, _ in P_COLS) + r' \\', r'\midrule']
for m in BASES:
    cells = []
    for name, _, _ in P_COLS:
        mu, (lo, hi), w, n, pv = res[name][m]
        cells.append(f'{mu:+.2f}{"$^{*}$" if pv < 0.05 else ""} [{lo:+.1f}, {hi:+.1f}]')
    lines.append(LABELS[m] + ' & ' + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
write('paired.tex', lines)
n_sig = {name: sum(r[4] < 0.05 for r in res[name].values()) for name in res}
ci_width = {name: np.median([r[1][1] - r[1][0] for r in res[name].values()]) for name in res}

# Phase 7: change only the three smallest classes of a random 5% split
sc = pd.read_csv(os.path.join(FINAL, 'phase7_scarcity_runs.csv'))
SC_COLS = [('IP, 5\\%', runs, 'IndianPines'), ('IP, smallest 15', sc[sc.study == 'smallfix'], 'IndianPines'),
           ('PU, 5\\%', runs, 'PaviaU'), ('PU, smallest 1/1/2', sc[sc.study == 'scarce'], 'PaviaU'),
           ('KSC, 5\\%', runs, 'KSC'), ('KSC, smallest 1/1/2', sc[sc.study == 'scarce'], 'KSC')]
sc_spread = block_table([(df, d, 'AA') for _, df, d in SC_COLS],
                        ['Block & ' + ' & '.join(n for n, _, _ in SC_COLS) + r' \\'], 'scarcity.tex')
sres = {name: paired(df, d, 'AA') for name, df, d in SC_COLS[1::2]}
lines = [r'\begin{tabular}{l' + 'c' * len(sres) + '}', r'\toprule',
         r'SNSM ($\gamma{+}\beta$) $-$ & ' + ' & '.join(sres) + r' \\', r'\midrule']
for m in BASES:
    cells = []
    for name in sres:
        mu, (lo, hi), w, n, pv = sres[name][m]
        cells.append(f'{mu:+.2f}{"$^{*}$" if pv < 0.05 else ""} [{lo:+.1f}, {hi:+.1f}]')
    lines.append(LABELS[m] + ' & ' + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
write('paired_scarcity.tex', lines)

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
s = cnt[(cnt.dataset == 'IndianPines') & (cnt.model == 'snsm')].iloc[0]
macros += [mac('ntrainIPcnt', thousands(s.n_train)), mac('ntestIPcnt', thousands(s.n_test))]
for d in ['PaviaU', 'KSC']:
    t = dj[(dj.dataset == d) & (dj.model == 'snsm')]
    macros.append(mac(f'djtest{SHORT[d]}', f'{thousands(t.n_test.min())}--{thousands(t.n_test.max())}'))
    macros.append(mac(f'djclasses{SHORT[d]}', int(t.n_test_classes.min())))
for i, d in enumerate(DJ_SETS):   # mean AA drop from random to patch-disjoint, averaged over blocks
    drop = np.mean([vals(runs, d, m, 'AA').mean() - vals(dj, d, m, 'AA').mean() for m in BLOCKS])
    macros.append(mac(f'drop{SHORT[d]}', f'{drop:.1f}'))
    macros.append(mac(f'spreadDJ{SHORT[d]}', f'{dj_spread[3 * i + 2]:.2f}'))
for (name, _), key in zip(IP_PROTOCOLS, ['Rand', 'Cnt', 'Std']):
    macros.append(mac(f'spreadIP{key}', f'{ip_spread[[n for n, _ in IP_PROTOCOLS].index(name)]:.2f}'))
for name, key in zip(res, ['IPrand', 'IPcnt', 'IPstd', 'PUdj', 'KSCdj']):
    macros.append(mac(f'nsig{key}', n_sig[name]))
    macros.append(mac(f'ciw{key}', f'{ci_width[name]:.1f}'))
for (name, _, _), key, sp in zip(SC_COLS, ['IPfive', 'IPsmall', 'PUfive', 'PUscarce', 'KSCfive', 'KSCscarce'], sc_spread):
    macros.append(mac(f'spreadSC{key}', f'{sp:.2f}'))
for name, key in zip(sres, ['IPsmall', 'PUscarce', 'KSCscarce']):
    macros.append(mac(f'nsig{key}', sum(r[4] < 0.05 for r in sres[name].values())))
# Numbers and claims written in the paper's text. Numbers become macros; qualitative claims (which blocks
# are significant, lowest, highest) are asserted, so a data change breaks this script instead of the text.
def sig(r):
    return {m for m, v in r.items() if v[4] < 0.05}


def rng(a, b, d=2):
    return f'{a:.{d}f}--{b:.{d}f}'


r = res['IP random 5\\%']
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

ci_max = max(max(abs(v[1][0]), abs(v[1][1])) for v in res['IP random 50/15'].values())
assert sig(res['IP random 50/15']) == set()
macros.append(mac('cntCImax', f'{np.ceil(ci_max * 10) / 10:.1f}'))
assert sig(res['IP standard']) == {'eca_gamma'}
macros.append(mac('stdEcaDiff', f"{res['IP standard']['eca_gamma'][0]:.2f}"))

r = sres['IP, smallest 15']
assert sig(r) == set(BASES)
g = sorted(v[0] for v in r.values())
macros += [mac('smallGain', rng(g[0], g[-1])), mac('spreadSCIPsmallShort', f'{sc_spread[1]:.1f}')]
assert {m for m, v in r.items() if v[0] >= 0.5} == {'cbam'}   # "exceeds 0.5 only against CBAM"
for d, key in [('PaviaU', 'PU'), ('KSC', 'KSC')]:
    s = sc[(sc.study == 'scarce') & (sc.dataset == d)]
    drop = np.mean([vals(runs, d, m, 'AA').mean() - s[s.model == m].AA.mean() for m in BLOCKS])
    macros.append(mac(f'dropScarce{key}', f'{drop:.0f}'))
means = sc[(sc.study == 'scarce') & (sc.dataset == 'PaviaU')].groupby('model').AA.mean()
assert means.idxmin() == 'snsm' and sig(sres['PU, smallest 1/1/2']) == {'eca_gamma', 'snsm_gamma'}
means = sc[(sc.study == 'scarce') & (sc.dataset == 'KSC')].groupby('model').AA.mean()
assert means.idxmin() == 'srm' and sig(sres['KSC, smallest 1/1/2']) == set()

drops = []
for d in DJ_SETS:
    oa = [vals(dj, d, m, 'OA').mean() for m in BLOCKS]
    macros.append(mac(f'djOA{SHORT[d]}', rng(min(oa), max(oa), 0)))
    drops.append(np.mean([vals(runs, d, m, 'AA').mean() - vals(dj, d, m, 'AA').mean() for m in BLOCKS]))
assert min(drops) > 30                    # "every block lost more than 30 AA points"
macros.append(mac('dropRange', rng(min(drops), max(drops), 0)))
sd = [vals(dj, d, m, 'AA').std(ddof=1) for d in DJ_SETS for m in BLOCKS]
macros.append(mac('djStdRange', rng(min(sd), max(sd), 1)))
assert sig(res['PU disjoint']) == set() and sig(res['KSC disjoint']) == {'plain', 'gct', 'srm'}
ksc = dj[dj.dataset == 'KSC'].groupby('model').AA.mean()
assert ksc.idxmax() == 'se_gamma' and ksc.idxmin() == 'srm'
assert 'srm' not in sig(res['IP random 5\\%'])                         # "SRM, tied best on random 5% IP"
assert all(3 <= ci_width[c] <= 4.5 for c in ['PU disjoint', 'KSC disjoint'])   # "about two points"

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

ip_oa = [vals(runs, 'IndianPines', m, 'OA').mean() for m in BLOCKS]
macros.append(mac('ipRandOAspread', f'{np.ptp(ip_oa):.2f}'))
macros.append(mac('nseedsIP', vals(runs, 'IndianPines', 'snsm', 'AA').size))
write('numbers.tex', macros)

print('Wrote', sorted(os.listdir(OUT)))
