# =============================================================================
# Cell 17: Extended Noise Robustness — ALL 5 DATASETS (Reviewer Comment #7)
# =============================================================================
# Tests: Gaussian, stripe, impulse, band dropout, calibration drift
# Loads saved model checkpoints for each dataset.
#
# Run AFTER cell_8 has completed (needs saved checkpoints + results).

import torch
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json
import os
import gc

GRAPH_DIR = '/kaggle/working/paper_graphs/'
RESULTS_DIR = '/kaggle/working/results/'
MAX_NOISE_SAMPLES = 10000  # Subsample large test sets to avoid OOM

# ── Noise Functions ──

def add_stripe_noise(X, intensity=0.1):
    X_noisy = X.copy()
    B, H, W, C = X_noisy.shape
    n_stripes = max(1, int(W * intensity))
    for _ in range(n_stripes):
        col = np.random.randint(0, W)
        X_noisy[:, :, col, :] += np.random.normal(0, 0.3)
    return X_noisy

def add_impulse_noise(X, ratio=0.05):
    X_noisy = X.copy()
    mask = np.random.random(X_noisy.shape)
    X_noisy[mask < ratio / 2] = 0.0
    X_noisy[mask > 1 - ratio / 2] = 1.0
    return X_noisy

def add_band_dropout(X, drop_ratio=0.1):
    X_noisy = X.copy()
    n_bands = X_noisy.shape[3]
    n_drop = max(1, int(n_bands * drop_ratio))
    drop_bands = np.random.choice(n_bands, n_drop, replace=False)
    X_noisy[:, :, :, drop_bands] = 0.0
    return X_noisy

def add_spectral_shift(X, shift_magnitude=0.05):
    X_noisy = X.copy()
    n_bands = X_noisy.shape[3]
    shifts = np.random.normal(0, shift_magnitude, n_bands).astype(np.float32)
    X_noisy += shifts[np.newaxis, np.newaxis, np.newaxis, :]
    return X_noisy

def eval_noisy(model, X_np, y_np, noise_fn, params, device):
    results = []
    for name, val in params:
        X_n = noise_fn(X_np, val)
        preds = []
        for i in range(0, len(X_n), 256):
            batch = torch.tensor(X_n[i:i+256], dtype=torch.float32, device=device)
            with torch.no_grad():
                _, p = model(batch).max(1)
            preds.append(p.cpu().numpy())
            del batch
        del X_n  # Free noise copy immediately
        acc = accuracy_score(y_np, np.concatenate(preds)) * 100
        results.append((name, acc))
        print(f"      {name}: {acc:.2f}%")
    return results

# ── Main runner — all datasets ──

DATASETS_TO_TEST = ['IndianPines', 'PaviaU', 'Botswana', 'KSC', 'WHU_Hi']
WINDOW_SIZE = 11
PCA_COMPONENTS = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for dataset_name in DATASETS_TO_TEST:
    print(f"\n{'='*60}")
    print(f"  NOISE ROBUSTNESS — {dataset_name}")
    print(f"{'='*60}")
    
    # Check for checkpoint (cell 8 saves here)
    ckpt_path = f'/kaggle/working/saved_models/best_model_{dataset_name}.pth'
    ckpt_path_alt = f'/kaggle/working/saved_models/{dataset_name}_proposed_seed42/best_model.pth'
    
    if os.path.exists(ckpt_path):
        load_path = ckpt_path
        load_type = 'state_dict'
    elif os.path.exists(ckpt_path_alt):
        load_path = ckpt_path_alt
        load_type = 'checkpoint'
    else:
        print(f"  ⚠ No checkpoint found, skipping")
        continue
    
    # Reload and preprocess data (SAME pipeline as cell 8)
    try:
        X_raw, y_raw = load_dataset(dataset_name)
    except Exception as e:
        print(f"  ⚠ Dataset error: {e}, skipping")
        continue
    
    X_pca, _ = apply_pca(X_raw, num_components=PCA_COMPONENTS)
    X_denoised = apply_kalman_denoising(X_pca)
    X_features = apply_polar_wavelet_transform(X_denoised)
    
    target_names = DATASET_INFO[dataset_name]['target_names']
    n_classes = len(target_names)
    n_bands = X_features.shape[2]
    
    _, X_test, _, y_test = create_disjoint_patches(
        X_features, y_raw, window_size=WINDOW_SIZE, train_ratio=0.15, seed=42)
    
    # Free raw data before GPU ops
    del X_raw, X_pca, X_denoised, X_features
    gc.collect()
    
    # Subsample if too large (WHU_Hi has 218K patches → OOM)
    if len(X_test) > MAX_NOISE_SAMPLES:
        print(f"  Subsampling: {len(X_test)} → {MAX_NOISE_SAMPLES} patches (OOM prevention)")
        np.random.seed(42)
        idx = np.random.choice(len(X_test), MAX_NOISE_SAMPLES, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]
    
    # Rebuild model and load weights
    model = SMCNN(num_classes=n_classes, num_bands=n_bands,
                  window_size=WINDOW_SIZE, use_modulation=True).to(device)
    
    if load_type == 'state_dict':
        model.load_state_dict(torch.load(load_path, weights_only=False, map_location=device))
    else:
        ckpt = torch.load(load_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    X_np = X_test
    y_np = y_test
    
    # Clean baseline
    clean_preds = []
    for i in range(0, len(X_np), 256):
        batch = torch.tensor(X_np[i:i+256], dtype=torch.float32, device=device)
        with torch.no_grad():
            _, p = model(batch).max(1)
        clean_preds.append(p.cpu().numpy())
    clean_acc = accuracy_score(y_np, np.concatenate(clean_preds)) * 100
    print(f"  Clean OA: {clean_acc:.2f}%")
    
    # Run all noise types
    print(f"\n  1. Gaussian:")
    gauss = eval_noisy(model, X_np, y_np,
        lambda x, s: x + np.random.normal(0, s, x.shape).astype(np.float32),
        [('σ=0.01', 0.01), ('σ=0.05', 0.05), ('σ=0.10', 0.10), ('σ=0.15', 0.15), ('σ=0.20', 0.20)], device)
    
    print(f"  2. Stripe:")
    stripe = eval_noisy(model, X_np, y_np, add_stripe_noise,
        [('5%', 0.05), ('10%', 0.10), ('15%', 0.15), ('20%', 0.20), ('30%', 0.30)], device)
    
    print(f"  3. Impulse:")
    impulse = eval_noisy(model, X_np, y_np, add_impulse_noise,
        [('1%', 0.01), ('3%', 0.03), ('5%', 0.05), ('10%', 0.10), ('15%', 0.15)], device)
    
    print(f"  4. Band Dropout:")
    bdrop = eval_noisy(model, X_np, y_np, add_band_dropout,
        [('5%', 0.05), ('10%', 0.10), ('15%', 0.15), ('20%', 0.20), ('30%', 0.30)], device)
    
    print(f"  5. Calibration Drift:")
    drift = eval_noisy(model, X_np, y_np, add_spectral_shift,
        [('0.02', 0.02), ('0.05', 0.05), ('0.08', 0.08), ('0.10', 0.10), ('0.15', 0.15)], device)
    
    # ── Plot ──
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    noise_data = [
        ('Gaussian', gauss, 'tab:blue'), ('Stripe', stripe, 'tab:orange'),
        ('Impulse', impulse, 'tab:green'), ('Band Drop', bdrop, 'tab:red'),
        ('Cal. Drift', drift, 'tab:purple')
    ]
    for ax, (name, res, color) in zip(axes, noise_data):
        labels = [r[0] for r in res]
        accs = [r[1] for r in res]
        ax.bar(range(len(labels)), accs, color=color, alpha=0.8)
        ax.axhline(y=clean_acc, color='black', linestyle='--', alpha=0.5, label=f'Clean ({clean_acc:.1f}%)')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_ylabel('OA (%)' if ax == axes[0] else '')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(max(0, min(accs) - 15), 105)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Extended Noise Robustness — {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'extended_noise_{dataset_name}')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save JSON
    save_data = {'dataset': dataset_name, 'clean': clean_acc,
                 'gaussian': gauss, 'stripe': stripe, 'impulse': impulse,
                 'band_dropout': bdrop, 'calibration_drift': drift}
    with open(os.path.join(RESULTS_DIR, f'extended_noise_{dataset_name}.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"  ✅ Saved: {path}.png")
    
    del model, X_test, y_test, X_np, y_np
    torch.cuda.empty_cache(); gc.collect()

print("\n✅ Extended noise robustness complete — all datasets!")
