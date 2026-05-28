# =============================================================================
# Cell 16: Statistical Significance Testing (Reviewer Comment #9)
# =============================================================================
# Adds: 95% confidence intervals for ALL 5 datasets
# Uses EXISTING 3-seed results — no retraining needed
#
# Run AFTER cell_8 (main execution) has completed.

import json
import numpy as np
from scipy import stats
import os

RESULTS_DIR = '/kaggle/working/results/'

def load_seed_metrics(dataset_name):
    """Load per-seed metrics for a dataset."""
    metrics = []
    for seed in [42, 100, 2024]:
        path = os.path.join(RESULTS_DIR, f'{dataset_name}_seed{seed}_metrics.json')
        if os.path.exists(path):
            with open(path) as f:
                metrics.append(json.load(f))
    return metrics

def compute_confidence_interval(values, confidence=0.95):
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if n < 2:
        return mean, std, (mean, mean)
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
    ci = (mean - t_crit * se, mean + t_crit * se)
    return mean, std, ci

# ── Run on ALL 5 datasets ──
datasets = ['IndianPines', 'PaviaU', 'Botswana', 'KSC', 'WHU_Hi']
all_stats = {}

print("=" * 80)
print("  STATISTICAL SIGNIFICANCE ANALYSIS — ALL DATASETS")
print("  3 seeds, 95% confidence intervals")
print("=" * 80)

for ds in datasets:
    metrics = load_seed_metrics(ds)
    if len(metrics) < 2:
        print(f"\n  ⚠ {ds}: insufficient seeds ({len(metrics)}), skipping")
        continue
    
    print(f"\n{'─'*60}")
    print(f"  {ds} ({len(metrics)} seeds)")
    print(f"{'─'*60}")
    
    ds_stats = {}
    for metric_name in ['OA', 'AA', 'Kappa', 'Precision', 'Recall', 'F1']:
        values = [m[metric_name] for m in metrics]
        mean, std, ci = compute_confidence_interval(values)
        ds_stats[metric_name] = {
            'values': [float(v) for v in values],
            'mean': round(float(mean), 4),
            'std': round(float(std), 4),
            'ci_lower': round(float(ci[0]), 4),
            'ci_upper': round(float(ci[1]), 4)
        }
        
        if metric_name == 'Kappa':
            print(f"  {metric_name:12s}: {mean:.4f} ± {std:.4f}  "
                  f"[95% CI: {ci[0]:.4f} — {ci[1]:.4f}]")
        else:
            print(f"  {metric_name:12s}: {mean:.2f}% ± {std:.2f}%  "
                  f"[95% CI: {ci[0]:.2f}% — {ci[1]:.2f}%]")
    
    all_stats[ds] = ds_stats

# Save
save_path = os.path.join(RESULTS_DIR, 'statistical_analysis.json')
with open(save_path, 'w') as f:
    json.dump(all_stats, f, indent=2)
print(f"\n✅ Saved: {save_path}")
print("   Cite as: 'mean ± std [95% CI]' in the paper")
