# =============================================================================
# Cell 18: Deep Spectral Interpretability — ALL 5 DATASETS (Comments #6, #8)
# =============================================================================
# Per-class γ/β heatmaps, attention entropy, Fisher criterion,
# inter-class distance, representation compactness.
# Loads saved checkpoints for each dataset.

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import json
import os
import gc

GRAPH_DIR = '/kaggle/working/paper_graphs/'
RESULTS_DIR = '/kaggle/working/results/'
MAX_INTERP_SAMPLES = 15000  # Subsample large test sets to avoid OOM


def extract_modulation_params(model, X_test_t, y_test_t):
    """Extract per-sample γ and β from SSMRB block1."""
    model.eval()
    gamma_pre, beta_pre = [], []
    
    def hook_gamma(m, i, o): gamma_pre.append(o.detach().cpu())
    def hook_beta(m, i, o): beta_pre.append(o.detach().cpu())
    
    h1 = model.block1.fc_gamma.register_forward_hook(hook_gamma)
    h2 = model.block1.fc_beta.register_forward_hook(hook_beta)
    
    with torch.no_grad():
        for i in range(0, len(X_test_t), 128):
            _ = model(X_test_t[i:i+128])
    
    h1.remove(); h2.remove()
    gammas = torch.sigmoid(torch.cat(gamma_pre, dim=0)).numpy()
    betas = torch.cat(beta_pre, dim=0).numpy()
    labels = y_test_t.cpu().numpy()
    return gammas, betas, labels


def run_interpretability(model, X_test_t, y_test_t, target_names, dataset_name, device):
    """Full interpretability suite for one dataset."""
    n_classes = len(target_names)
    
    # ── 1. Extract γ/β ──
    print(f"  Extracting modulation parameters...")
    gammas, betas, labels = extract_modulation_params(model, X_test_t, y_test_t)
    n_channels = gammas.shape[1]
    
    # Per-class means
    gamma_pc = np.zeros((n_classes, n_channels))
    beta_pc = np.zeros((n_classes, n_channels))
    gamma_std_pc = np.zeros((n_classes, n_channels))
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            gamma_pc[c] = gammas[mask].mean(axis=0)
            beta_pc[c] = betas[mask].mean(axis=0)
            gamma_std_pc[c] = gammas[mask].std(axis=0)
    
    # ── 2. Plot γ/β heatmaps ──
    fig, axes = plt.subplots(1, 3, figsize=(24, max(6, n_classes * 0.5)))
    
    im1 = axes[0].imshow(gamma_pc, aspect='auto', cmap='RdYlBu_r', vmin=0.3, vmax=0.7)
    axes[0].set_title('Mean γ (Scale)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Channel'); axes[0].set_ylabel('Class')
    axes[0].set_yticks(range(n_classes)); axes[0].set_yticklabels(target_names, fontsize=7)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    im2 = axes[1].imshow(beta_pc, aspect='auto', cmap='coolwarm')
    axes[1].set_title('Mean β (Shift)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Channel')
    axes[1].set_yticks(range(n_classes)); axes[1].set_yticklabels(target_names, fontsize=7)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    im3 = axes[2].imshow(gamma_std_pc, aspect='auto', cmap='Oranges')
    axes[2].set_title('γ Std Dev (input-dependent)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Channel')
    axes[2].set_yticks(range(n_classes)); axes[2].set_yticklabels(target_names, fontsize=7)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    plt.suptitle(f'SSMRB Modulation — {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'modulation_{dataset_name}')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Modulation heatmaps: {path}.png")
    
    # ── 3. Attention entropy ──
    entropies = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            mg = gammas[mask].mean(axis=0)
            gn = mg / (mg.sum() + 1e-10)
            entropies.append(float(-np.sum(gn * np.log(gn + 1e-10))))
        else:
            entropies.append(0.0)
    
    plt.figure(figsize=(12, 5))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    plt.bar(range(n_classes), entropies, color=colors, edgecolor='gray', alpha=0.9)
    plt.axhline(y=np.mean(entropies), color='red', linestyle='--', alpha=0.7,
                label=f'Mean = {np.mean(entropies):.2f}')
    plt.xlabel('Class'); plt.ylabel('H(γ)')
    plt.title(f'Attention Entropy — {dataset_name}', fontsize=13, fontweight='bold')
    plt.xticks(range(n_classes), target_names, rotation=45, ha='right', fontsize=7)
    plt.legend(); plt.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'attention_entropy_{dataset_name}')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Attention entropy: {path}.png")
    
    # ── 4. Representation quality (Fisher + compactness) ──
    print(f"  Computing representation quality...")
    all_features = []
    for i in range(0, len(X_test_t), 256):
        with torch.no_grad():
            feats = model.extract_features(X_test_t[i:i+256])
        all_features.append(feats.cpu().numpy())
    features = np.concatenate(all_features)
    
    centroids = np.zeros((n_classes, features.shape[1]))
    compactness = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = features[mask].mean(axis=0)
            compactness.append(float(np.mean(np.sqrt(np.sum((features[mask] - centroids[c])**2, axis=1)))))
        else:
            compactness.append(0.0)
    
    global_mean = features.mean(axis=0)
    inter_var = sum(((labels == c).sum() * np.sum((centroids[c] - global_mean)**2))
                    for c in range(n_classes) if (labels == c).sum() > 0)
    intra_var = sum(np.sum((features[labels == c] - centroids[c])**2)
                    for c in range(n_classes) if (labels == c).sum() > 0)
    fisher = float(inter_var / (intra_var + 1e-10))
    
    inter_dist = pairwise_distances(centroids, metric='euclidean')
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    im = axes[0].imshow(inter_dist, cmap='viridis')
    axes[0].set_title('Inter-Class Distance', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(n_classes)); axes[0].set_yticks(range(n_classes))
    axes[0].set_xticklabels(target_names, rotation=45, ha='right', fontsize=6)
    axes[0].set_yticklabels(target_names, fontsize=6)
    plt.colorbar(im, ax=axes[0], shrink=0.8)
    
    axes[1].bar(range(n_classes), compactness,
                color=plt.cm.tab20(np.linspace(0, 1, n_classes)), edgecolor='gray')
    axes[1].set_title(f'Intra-Class Compactness\nFisher = {fisher:.2f}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Class'); axes[1].set_ylabel('Mean Dist to Centroid')
    axes[1].set_xticks(range(n_classes))
    axes[1].set_xticklabels(target_names, rotation=45, ha='right', fontsize=6)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Representation Quality — {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'representation_{dataset_name}')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Representation quality: {path}.png")
    
    # Save metrics
    quality = {
        'dataset': dataset_name,
        'fisher_criterion': fisher,
        'mean_compactness': float(np.mean(compactness)),
        'per_class_compactness': compactness,
        'min_inter_class_dist': float(inter_dist[inter_dist > 0].min()),
        'max_inter_class_dist': float(inter_dist.max()),
        'mean_entropy': float(np.mean(entropies)),
        'per_class_entropy': entropies
    }
    with open(os.path.join(RESULTS_DIR, f'interpretability_{dataset_name}.json'), 'w') as f:
        json.dump(quality, f, indent=2)
    
    return quality


# ── Run on ALL 5 datasets ──
DATASETS_TO_ANALYZE = ['IndianPines', 'PaviaU', 'Botswana', 'KSC', 'WHU_Hi']
WINDOW_SIZE = 11
PCA_COMPONENTS = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_quality = {}

for dataset_name in DATASETS_TO_ANALYZE:
    print(f"\n{'='*60}")
    print(f"  INTERPRETABILITY — {dataset_name}")
    print(f"{'='*60}")
    
    # Check for checkpoint
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
    
    # Reload and preprocess (SAME as cell 8)
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
    if len(X_test) > MAX_INTERP_SAMPLES:
        print(f"  Subsampling: {len(X_test)} → {MAX_INTERP_SAMPLES} patches (OOM prevention)")
        np.random.seed(42)
        idx = np.random.choice(len(X_test), MAX_INTERP_SAMPLES, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]
    
    # Rebuild model
    model = SMCNN(num_classes=n_classes, num_bands=n_bands,
                  window_size=WINDOW_SIZE, use_modulation=True).to(device)
    
    if load_type == 'state_dict':
        model.load_state_dict(torch.load(load_path, weights_only=False, map_location=device))
    else:
        ckpt = torch.load(load_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)
    
    quality = run_interpretability(model, X_test_t, y_test_t, target_names, dataset_name, device)
    all_quality[dataset_name] = quality
    
    del model, X_test_t, y_test_t, X_test, y_test
    torch.cuda.empty_cache(); gc.collect()

# Summary
print(f"\n{'='*60}")
print("  INTERPRETABILITY SUMMARY")
print(f"{'='*60}")
for ds, q in all_quality.items():
    print(f"  {ds:15s}: Fisher={q['fisher_criterion']:.2f}, "
          f"Compact={q['mean_compactness']:.4f}, "
          f"Entropy={q['mean_entropy']:.2f}")

print("\n✅ Deep interpretability complete — all datasets!")
