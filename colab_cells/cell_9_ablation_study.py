# =============================================================================
# Cell 9: Ablation Study
# =============================================================================
# Addresses Reviewer Issue 8 (No Ablation Study Provided).
# Tests 5 configurations to isolate the contribution of each component:
#   1. Baseline CNN (no preprocessing, no modulation, no SFWOA)
#   2. + AFDKF denoising only
#   3. + AFDKF + PLCWT features
#   4. + AFDKF + PLCWT + Self-Modulation (SMCNN)
#   5. Full Proposed (SMCNN + SFWOA)
#
# Also tests: SFWOA vs plain AdamW (Reviewer Issue 10)

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import gc

def run_ablation_study(dataset_name='IndianPines'):
    """
    Runs 5 ablation configurations and generates a grouped bar chart.
    """
    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY — {dataset_name}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = 11
    pca_components = 30
    epochs = 30
    seed = 42
    
    X, y = load_dataset(dataset_name)
    target_names = DATASET_INFO[dataset_name]['target_names']
    num_classes = len(target_names)
    
    # Precompute all preprocessing variants
    X_pca, _ = apply_pca(X, num_components=pca_components)
    X_denoised = apply_kalman_denoising(X_pca)
    X_wavelet = apply_polar_wavelet_transform(X_denoised)
    
    configs = [
        {
            'name': 'Baseline CNN',
            'data': X_pca,
            'use_modulation': False,
            'use_sfwoa': False,
        },
        {
            'name': '+ AFDKF',
            'data': X_denoised,
            'use_modulation': False,
            'use_sfwoa': False,
        },
        {
            'name': '+ AFDKF + PLCWT',
            'data': X_wavelet,
            'use_modulation': False,
            'use_sfwoa': False,
        },
        {
            'name': '+ Self-Modulation',
            'data': X_wavelet,
            'use_modulation': True,
            'use_sfwoa': False,
        },
        {
            'name': 'Full Proposed\n(+ SFWOA)',
            'data': X_wavelet,
            'use_modulation': True,
            'use_sfwoa': True,
        },
    ]
    
    results = []
    
    for i, cfg in enumerate(configs):
        print(f"\n  [{i+1}/{len(configs)}] {cfg['name'].replace(chr(10), ' ')}")
        
        X_train, X_test, y_train, y_test = create_disjoint_patches(
            cfg['data'], y, window_size=window_size, train_ratio=0.15, seed=seed)
        
        num_bands = X_train.shape[3]
        model = SMCNN(num_classes=num_classes, num_bands=num_bands,
                      window_size=window_size, use_modulation=cfg['use_modulation'])
        
        _, X_test_t, y_test_t, history, val_acc, elapsed = train_model(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=64, seed=seed,
            use_sfwoa=cfg['use_sfwoa'],
            tag=f"ablation_{i}", dataset_name=dataset_name)
        
        metrics, _, _, _ = full_evaluation(model, X_test_t, y_test_t,
                                           target_names, dataset_name, tag=f"ablation_{i}")
        
        results.append({
            'config': cfg['name'].replace('\n', ' '),
            'OA': metrics['OA'],
            'AA': metrics['AA'],
            'Kappa': metrics['Kappa'],
            'F1': metrics['F1'],
            'time_s': elapsed
        })
        
        # Cleanup GPU
        del model, X_train, X_test, X_test_t, y_test_t
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print results table
    print(f"\n  {'Config':<30s} {'OA':>8s} {'AA':>8s} {'κ':>8s} {'F1':>8s} {'Time':>8s}")
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['config']:<30s} {r['OA']:>7.2f}% {r['AA']:>7.2f}% {r['Kappa']:>7.4f} {r['F1']:>7.2f}% {r['time_s']:>6.1f}s")
    
    # Save ablation results
    with open(os.path.join(RESULTS_DIR, f'ablation_{dataset_name}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # ── Plot Ablation Bar Chart ──
    config_names = [r['config'] for r in results]
    oa_vals      = [r['OA'] for r in results]
    
    colors = ['#bdc3c7', '#85c1e9', '#5dade2', '#2e86c1', '#1a5276']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(config_names)), oa_vals, color=colors, 
                  edgecolor='white', linewidth=1.5, width=0.6)
    
    # Value labels on bars
    for bar, val in zip(bars, oa_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, fontsize=10)
    ax.set_ylabel('Overall Accuracy (%)', fontsize=13)
    ax.set_title(f'Ablation Study — {dataset_name}', fontsize=16, fontweight='bold')
    ax.set_ylim(max(0, min(oa_vals) - 10), 100)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'ablation_{dataset_name}.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")
    
    # ── Plot Ablation: All Metrics Grouped ──
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(config_names))
    width = 0.2
    
    ax.bar(x - width*1.5, [r['OA'] for r in results], width, label='OA', color='#2e86c1')
    ax.bar(x - width*0.5, [r['AA'] for r in results], width, label='AA', color='#27ae60')
    ax.bar(x + width*0.5, [r['F1'] for r in results], width, label='F1', color='#e74c3c')
    ax.bar(x + width*1.5, [r['Kappa']*100 for r in results], width, label='κ×100', color='#f39c12')
    
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=9)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(f'Ablation Study (All Metrics) — {dataset_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'ablation_all_metrics_{dataset_name}.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")


# ── RUN ABLATION ──
run_ablation_study('IndianPines')
