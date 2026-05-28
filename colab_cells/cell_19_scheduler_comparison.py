# =============================================================================
# Cell 19: SFWOA Scheduler Comparison — Indian Pines + KSC (Comment #10)
# =============================================================================
# Compares 5 LR schedulers: Constant, StepDecay, CosineAnnealing, OneCycle, SFWOA
# Runs on Indian Pines (best case) + KSC (worst case) to show full picture.
#
# Each scheduler trains a FRESH model from scratch with seed=42.

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
import gc

RESULTS_DIR = '/kaggle/working/results/'
GRAPH_DIR = '/kaggle/working/paper_graphs/'
WINDOW_SIZE = 11
PCA_COMPONENTS = 30

def train_with_scheduler(n_classes, n_bands, X_train, y_train, X_test, y_test,
                         scheduler_name, epochs=100, batch_size=64, seed=42,
                         early_stop_patience=10):
    """Train a fresh SMCNN model with a specific LR scheduler."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SMCNN(num_classes=n_classes, num_bands=n_bands,
                  window_size=WINDOW_SIZE, use_modulation=True).to(device)
    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.long, device=device)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_te = torch.tensor(y_test, dtype=torch.long, device=device)
    
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    base_lr = 0.001
    
    if scheduler_name == 'SFWOA':
        sfwoa = SFWOA_HyperTuner(model, base_lr=base_lr)
        optimizer = sfwoa.get_optimizer()
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
        sfwoa = None
        if scheduler_name == 'CosineAnnealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        elif scheduler_name == 'OneCycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=base_lr * 10, total_steps=epochs * len(train_loader),
                pct_start=0.3, anneal_strategy='cos')
        elif scheduler_name == 'StepDecay':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        else:  # Constant
            scheduler = None
    
    history = {'lr': [], 'val_acc': [], 'train_acc': []}
    best_val_acc = 0.0
    prev_loss = float('inf')
    no_improve = 0
    
    print(f"    Training with {scheduler_name}...", end=' ')
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if scheduler_name == 'OneCycle' and scheduler:
                scheduler.step()
            r_loss += loss.item() * labels.size(0)
            _, p = out.max(1)
            total += labels.size(0)
            correct += (p == labels).sum().item()
        
        train_loss = r_loss / total
        train_acc = 100.0 * correct / total
        
        # Epoch-level scheduler step
        if scheduler_name not in ['OneCycle', 'SFWOA', 'Constant'] and scheduler:
            scheduler.step()
        if sfwoa:
            _, _ = sfwoa.step(train_loss, prev_loss, epoch, epochs)
        prev_loss = train_loss
        
        # Validate
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for i in range(0, len(X_te), 256):
                batch_x = X_te[i:i+256]
                batch_y = y_te[i:i+256]
                out = model(batch_x)
                _, p = out.max(1)
                vt += batch_y.size(0)
                vc += (p == batch_y).sum().item()
        val_acc = 100.0 * vc / vt
        
        history['lr'].append(float(optimizer.param_groups[0]['lr']))
        history['val_acc'].append(float(val_acc))
        history['train_acc'].append(float(train_acc))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
        if early_stop_patience and no_improve >= early_stop_patience:
            break
    
    elapsed = time.time() - start
    print(f"Best={best_val_acc:.2f}%, {elapsed:.1f}s, {epoch+1} epochs")
    
    del model, X_tr, y_tr, X_te, y_te
    torch.cuda.empty_cache(); gc.collect()
    
    return best_val_acc, history, elapsed


# ── Run on Indian Pines (best) + KSC (worst) ──
SCHEDULER_DATASETS = ['IndianPines', 'KSC']
SCHEDULERS = ['Constant', 'StepDecay', 'CosineAnnealing', 'OneCycle', 'SFWOA']
SCHED_COLORS = {'Constant': 'gray', 'StepDecay': 'tab:blue', 'CosineAnnealing': 'tab:green',
                'OneCycle': 'tab:orange', 'SFWOA': 'tab:red'}

for dataset_name in SCHEDULER_DATASETS:
    print(f"\n{'='*60}")
    print(f"  SCHEDULER COMPARISON — {dataset_name}")
    print(f"{'='*60}")
    
    # Load and preprocess (SAME as cell 8)
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
    
    X_train, X_test, y_train, y_test = create_disjoint_patches(
        X_features, y_raw, window_size=WINDOW_SIZE, train_ratio=0.15, seed=42)
    
    results = {}
    for sched in SCHEDULERS:
        acc, hist, elapsed = train_with_scheduler(
            n_classes, n_bands, X_train, y_train, X_test, y_test,
            sched, epochs=100, seed=42)
        results[sched] = {'best_val_acc': round(acc, 2), 'time_s': round(elapsed, 2), 'history': hist}
    
    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for sched, data in results.items():
        ep = range(1, len(data['history']['val_acc']) + 1)
        axes[0].plot(ep, data['history']['val_acc'],
                     label=f"{sched} ({data['best_val_acc']:.1f}%)",
                     color=SCHED_COLORS[sched], linewidth=1.5)
        axes[1].plot(ep, data['history']['lr'],
                     label=sched, color=SCHED_COLORS[sched], linewidth=1.5)
    
    axes[0].set_title('Validation Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Val Acc (%)')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[1].set_title('Learning Rate', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('LR'); axes[1].set_yscale('log')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'LR Scheduler Comparison — {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'scheduler_comparison_{dataset_name}')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    print(f"\n  {'Scheduler':<18} {'Best OA':>10} {'Time':>10}")
    print(f"  {'─'*40}")
    for s in SCHEDULERS:
        d = results[s]
        tag = ' ← ours' if s == 'SFWOA' else ''
        print(f"  {s:<18} {d['best_val_acc']:>8.2f}% {d['time_s']:>8.1f}s{tag}")
    
    with open(os.path.join(RESULTS_DIR, f'scheduler_comparison_{dataset_name}.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  ✅ Saved: {path}.png")
    
    del X_train, X_test, y_train, y_test, X_raw, X_pca, X_denoised, X_features
    gc.collect()

print("\n✅ Scheduler comparison complete!")
