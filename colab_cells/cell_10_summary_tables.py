# =============================================================================
# Cell 10: Cross-Dataset Summary Table & Final Paper-Ready Outputs
# =============================================================================
# Addresses Reviewer Issues:
#   7  - Multi-dataset evaluation summary
#   11 - Efficiency comparison table
#   19 - Benchmark table consistency (automated formatting)
#   Minor - Formatted LaTeX-ready tables

import json
import os
import numpy as np
import pandas as pd

def generate_summary_tables():
    """
    Reads all saved results and generates formatted summary tables.
    Run this AFTER cell 8 and cell 9 have completed.
    """
    results_path = '/kaggle/working/results/all_results.json'
    
    if not os.path.exists(results_path):
        print("ERROR: Run cell 8 first to generate results.")
        return
    
    with open(results_path, 'r') as f:
        all_results = json.load(f)
    
    # ── 1. Cross-Dataset Performance Table ──
    print("\n" + "="*70)
    print("  TABLE 1: Cross-Dataset Performance Comparison (Mean ± Std)")
    print("="*70)
    
    rows = []
    for ds, data in all_results.items():
        metrics_list = data['metrics']
        row = {'Dataset': ds}
        for metric in ['OA', 'AA', 'Precision', 'Recall', 'F1']:
            vals = [m[metric] for m in metrics_list]
            row[metric] = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"
        kappa_vals = [m['Kappa'] for m in metrics_list]
        row['Kappa'] = f"{np.mean(kappa_vals):.4f} ± {np.std(kappa_vals):.4f}"
        rows.append(row)
    
    df_perf = pd.DataFrame(rows)
    print(df_perf.to_string(index=False))
    df_perf.to_csv('/kaggle/working/results/cross_dataset_performance.csv', index=False)
    
    # ── 2. Computational Efficiency Table ──
    print("\n" + "="*70)
    print("  TABLE 2: Computational Efficiency")
    print("="*70)
    
    eff_rows = []
    for ds, data in all_results.items():
        eff = data.get('efficiency', {})
        eff_rows.append({
            'Dataset': ds,
            'Params (M)': f"{eff.get('total_params', 0)/1e6:.2f}",
            'FLOPs (M)': f"{eff.get('flops', 0)/1e6:.2f}" if eff.get('flops') else 'N/A',
            'Inference (ms)': f"{eff.get('inference_latency_ms', 0):.2f}",
            'GPU Mem (MB)': f"{eff.get('gpu_memory_mb', 0):.1f}",
            'Train Time (s)': f"{data.get('avg_train_time_s', 0):.1f}"
        })
    
    df_eff = pd.DataFrame(eff_rows)
    print(df_eff.to_string(index=False))
    df_eff.to_csv('/kaggle/working/results/computational_efficiency.csv', index=False)
    
    # ── 3. LaTeX-ready table ──
    print("\n" + "="*70)
    print("  LaTeX Table (copy-paste into your paper)")
    print("="*70)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Cross-dataset classification performance of the proposed HIA-PAL-SMCNN.}")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("Dataset & OA (\\%) & AA (\\%) & $\\kappa$ & Precision (\\%) & Recall (\\%) & F1 (\\%) \\\\")
    print("\\hline")
    for _, row in df_perf.iterrows():
        print(f"{row['Dataset']} & {row['OA']} & {row['AA']} & {row['Kappa']} & {row['Precision']} & {row['Recall']} & {row['F1']} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # ── 4. List all generated files ──
    print("\n" + "="*70)
    print("  ALL GENERATED FILES")
    print("="*70)
    
    for folder in ['/kaggle/working/paper_graphs/', '/kaggle/working/results/', '/kaggle/working/saved_models/']:
        if os.path.exists(folder):
            for f in sorted(os.listdir(folder)):
                size = os.path.getsize(os.path.join(folder, f))
                print(f"  {folder}{f}  ({size/1024:.1f} KB)")
    
    print("\nDone! Download the entire /kaggle/working/ folder for your paper.")


# ── RUN ──
generate_summary_tables()
