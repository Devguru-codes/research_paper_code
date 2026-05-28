# =============================================================================
# Cell 7: Evaluation, Statistics & Publication-Quality Graphs
# =============================================================================
# Addresses Reviewer Issues:
#   4  - Theoretical insight (t-SNE, spectral attention visualization)
#   6  - Statistical significance (mean ± std over seeds, confidence intervals)
#   8  - Ablation bar chart
#   14 - Publication-quality figures (300 DPI PDF)
#   17 - Discussion support (class confusion analysis, per-class accuracy)
#   18 - Error analysis (misclassification visualization, noise robustness)
#   19 - Benchmark table fixes (automated consistent formatting)
#   Minor: confusion matrices, per-class tables, confidence intervals

import torch
import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             classification_report, cohen_kappa_score,
                             precision_score, recall_score, f1_score)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json

GRAPH_DIR = '/kaggle/working/paper_graphs/'
RESULTS_DIR = '/kaggle/working/results/'

# ── 1. Full Statistical Evaluation ──────────────────────────────────────────

def full_evaluation(model, X_test_t, y_test_t, target_names, dataset_name, tag="proposed"):
    """
    Complete evaluation: OA, AA, Kappa, Precision, Recall, F1, per-class accuracy.
    Returns a dict with all metrics.
    """
    model.eval()
    device = X_test_t.device
    
    # Batch prediction to avoid OOM on large test sets
    all_preds = []
    batch_size = 256
    for i in range(0, len(X_test_t), batch_size):
        batch = X_test_t[i:i+batch_size]
        with torch.no_grad():
            outputs = model(batch)
            _, preds = outputs.max(1)
        all_preds.append(preds.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = y_test_t.cpu().numpy()
    
    oa        = accuracy_score(y_true, y_pred) * 100
    kappa     = cohen_kappa_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    recall    = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1        = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = []
    for i in range(len(target_names)):
        if cm[i].sum() > 0:
            per_class_acc.append(cm[i, i] / cm[i].sum() * 100)
        else:
            per_class_acc.append(0.0)
    aa = np.mean(per_class_acc)
    
    metrics = {
        'OA': oa, 'AA': aa, 'Kappa': kappa,
        'Precision': precision, 'Recall': recall, 'F1': f1,
        'per_class_acc': per_class_acc,
        'y_true': y_true.tolist(), 'y_pred': y_pred.tolist()
    }
    
    # Save metrics JSON
    save_path = os.path.join(RESULTS_DIR, f'{dataset_name}_{tag}_metrics.json')
    metrics_save = {k: v for k, v in metrics.items() if k not in ['y_true', 'y_pred']}
    with open(save_path, 'w') as f:
        json.dump(metrics_save, f, indent=2)
    
    return metrics, y_true, y_pred, cm


def print_multi_seed_stats(all_metrics, target_names, dataset_name):
    """
    Prints Mean ± Std across seeds for all metrics.
    Addresses Reviewer Issue 6 (report multiple runs, provide std).
    """
    print(f"\n{'='*60}")
    print(f"  Statistical Analysis — {dataset_name} ({len(all_metrics)} runs)")
    print(f"{'='*60}")
    
    for metric in ['OA', 'AA', 'Kappa', 'Precision', 'Recall', 'F1']:
        vals = [m[metric] for m in all_metrics]
        if metric == 'Kappa':
            print(f"  {metric:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        else:
            print(f"  {metric:12s}: {np.mean(vals):.2f}% ± {np.std(vals):.2f}%")
    
    # Per-class accuracy table
    print(f"\n  Per-Class Accuracy (Best Run):")
    best_idx = np.argmax([m['OA'] for m in all_metrics])
    pca = all_metrics[best_idx]['per_class_acc']
    df = pd.DataFrame({'Class': target_names, 'Accuracy (%)': [f"{a:.2f}" for a in pca]})
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv(os.path.join(RESULTS_DIR, f'{dataset_name}_per_class_accuracy.csv'), index=False)
    
    return best_idx

# ── 2. Confusion Matrix ─────────────────────────────────────────────────────

def plot_confusion_matrix(cm, target_names, dataset_name, tag=""):
    plt.figure(figsize=(max(10, len(target_names)*0.8), max(8, len(target_names)*0.7)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix — {dataset_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'confusion_matrix_{dataset_name}{tag}.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")

# ── 3. t-SNE Feature Embedding ──────────────────────────────────────────────

def plot_tsne(model, X_test_t, y_test_t, target_names, dataset_name, tag=""):
    print("  Generating t-SNE plot...")
    model.eval()
    device = X_test_t.device
    
    # Extract features in batches
    all_features = []
    batch_size = 256
    for i in range(0, len(X_test_t), batch_size):
        batch = X_test_t[i:i+batch_size]
        with torch.no_grad():
            feats = model.extract_features(batch)
        all_features.append(feats.cpu().numpy())
    features = np.concatenate(all_features)
    y_true = y_test_t.cpu().numpy()
    
    # Subsample if too many points (t-SNE is O(n²))
    max_points = 5000
    if len(features) > max_points:
        idx = np.random.choice(len(features), max_points, replace=False)
        features = features[idx]
        y_true = y_true[idx]
    
    tsne = TSNE(n_components=2, random_state=42, init='pca',
                learning_rate='auto', perplexity=min(30, len(features)-1))
    embedded = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1],
                          c=y_true, cmap='tab20', alpha=0.7, s=12, edgecolors='none')
    handles, _ = scatter.legend_elements()
    plt.legend(handles, target_names[:len(handles)],
               bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.title(f't-SNE Feature Embeddings — {dataset_name}', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'tsne_{dataset_name}{tag}.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")

# ── 4. Learning Curves ──────────────────────────────────────────────────────

def plot_learning_curves(history, dataset_name, tag=""):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=1.5, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=1.5, label='Val Loss')
    axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', linewidth=1.5, label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', linewidth=1.5, label='Val Acc')
    axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    # Learning Rate (SFWOA convergence analysis - Reviewer Issue 4)
    if 'lr' in history and history['lr']:
        axes[2].plot(epochs, history['lr'], 'g-', linewidth=1.5)
        axes[2].set_title('SFWOA Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{dataset_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'learning_curves_{dataset_name}{tag}.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")

# ── 5. Robustness to Noise (Reviewer Issue 18) ──────────────────────────────

def plot_noise_robustness(model, X_test, y_test, target_names, dataset_name):
    """Tests model accuracy under different levels of Gaussian noise."""
    print("  Running noise robustness analysis...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    accuracies = []
    
    X_test_np = X_test if isinstance(X_test, np.ndarray) else X_test.cpu().numpy()
    y_test_np = y_test if isinstance(y_test, np.ndarray) else y_test.cpu().numpy()
    
    for sigma in noise_levels:
        all_preds = []
        batch_size = 256
        for i in range(0, len(X_test_np), batch_size):
            batch_np = X_test_np[i:i+batch_size]
            if sigma > 0:
                batch_np = batch_np + np.random.normal(0, sigma, batch_np.shape).astype(np.float32)
            batch_t = torch.tensor(batch_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = model(batch_t)
                _, preds = out.max(1)
            all_preds.append(preds.cpu().numpy())
            del batch_t
        
        y_pred = np.concatenate(all_preds)
        acc = accuracy_score(y_test_np, y_pred) * 100
        accuracies.append(acc)
        print(f"    σ={sigma:.2f}: OA={acc:.2f}%")
    
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.fill_between(noise_levels, accuracies, alpha=0.15, color='blue')
    plt.title(f'Robustness to Gaussian Noise — {dataset_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Noise Standard Deviation (σ)', fontsize=12)
    plt.ylabel('Overall Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'noise_robustness_{dataset_name}.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")

# ── 6. Spectral Attention Visualization (Reviewer Issue 4) ──────────────────

def plot_spectral_attention(model, X_test_t, dataset_name):
    """
    Visualizes the gamma (scale) values from the SSMRB to show
    which spectral-channel features are being emphasized.
    """
    if not hasattr(model, 'block1') or not hasattr(model.block1, 'fc_gamma'):
        print("  Skipping attention viz (model has no modulation).")
        return
    
    print("  Generating spectral attention heatmap...")
    model.eval()
    
    # Hook to capture gamma values
    gamma_values = []
    def hook_fn(module, input, output):
        gamma_values.append(output.detach().cpu())
    
    hook = model.block1.fc_gamma.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        sample = X_test_t[:min(100, len(X_test_t))]
        _ = model(sample)
    
    hook.remove()
    
    if gamma_values:
        gammas = torch.sigmoid(torch.cat(gamma_values, dim=0)).numpy()  # (N, 64)
        mean_gamma = gammas.mean(axis=0)  # (64,)
        
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(mean_gamma)), mean_gamma, color='steelblue', alpha=0.8)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Neutral (γ=0.5)')
        plt.title(f'SSMRB Spectral Attention (γ values) — {dataset_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Channel Index', fontsize=12)
        plt.ylabel('Mean γ (Scale Factor)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(GRAPH_DIR, f'spectral_attention_{dataset_name}.pdf')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"  Saved: {path}")

print("All evaluation and visualization functions ready.")
