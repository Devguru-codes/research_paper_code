# =============================================================================
# Cell 13: Baseline Model Comparisons (Reviewer Issue 5)
# =============================================================================
# The reviewer demands comparison against modern baselines.
# We implement 3 standard baselines and run them on the SAME splits:
#   1. 2D-CNN (standard convolutions, no modulation)
#   2. 3D-CNN (uses 3D convolutions on the spectral-spatial cube)
#   3. HybridSN-style (3D conv → 2D conv → FC)
# All trained with the same hyperparameters for fair comparison.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import gc

# ── Baseline 1: Standard 2D-CNN ─────────────────────────────────────────────

class Baseline_2DCNN(nn.Module):
    """Standard 2D-CNN without any modulation or attention."""
    def __init__(self, num_classes, num_bands, window_size):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_bands, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))
    
    def extract_features(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.pool(x).flatten(1)


# ── Baseline 2: 3D-CNN ──────────────────────────────────────────────────────

class Baseline_3DCNN(nn.Module):
    """3D-CNN that processes the spectral-spatial cube jointly."""
    def __init__(self, num_classes, num_bands, window_size):
        super().__init__()
        # Input: (B, 1, Bands, H, W)
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(8), nn.ReLU())
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(16), nn.ReLU())
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU())
        
        # After 3D convs: (B, 32, Bands, H, W) → flatten spectral dim
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # x: (B, H, W, Bands) → (B, 1, Bands, H, W)
        x = x.permute(0, 3, 1, 2).unsqueeze(1)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))
    
    def extract_features(self, x):
        x = x.permute(0, 3, 1, 2).unsqueeze(1)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return self.pool(x).flatten(1)


# ── Baseline 3: HybridSN-style ──────────────────────────────────────────────

class Baseline_HybridSN(nn.Module):
    """HybridSN-inspired: 3D conv → reshape → 2D conv → FC."""
    def __init__(self, num_classes, num_bands, window_size):
        super().__init__()
        # 3D convolution block
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(8), nn.ReLU())
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(16), nn.ReLU())
        
        # 2D convolution block (after collapsing spectral into channels)
        self.conv2d = nn.Sequential(
            nn.Conv2d(16 * num_bands, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU())
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(64, num_classes)
        self.num_bands = num_bands
    
    def forward(self, x):
        b = x.size(0)
        # (B, H, W, Bands) → (B, 1, Bands, H, W)
        x = x.permute(0, 3, 1, 2).unsqueeze(1)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        # (B, 16, Bands, H, W) → (B, 16*Bands, H, W)
        x = x.reshape(b, -1, x.size(3), x.size(4))
        x = self.conv2d(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))
    
    def extract_features(self, x):
        b = x.size(0)
        x = x.permute(0, 3, 1, 2).unsqueeze(1)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = x.reshape(b, -1, x.size(3), x.size(4))
        x = self.conv2d(x)
        return self.pool(x).flatten(1)


# ── Run Baseline Comparison ─────────────────────────────────────────────────

def run_baseline_comparison(dataset_name='IndianPines'):
    """Trains all baselines and the proposed model on the same split."""
    print(f"\n{'='*60}")
    print(f"  BASELINE COMPARISON — {dataset_name}")
    print(f"{'='*60}")
    
    window_size = 11
    seed = 42
    
    X, y = load_dataset(dataset_name)
    X_pca, _ = apply_pca(X, num_components=30)
    X_denoised = apply_kalman_denoising(X_pca)
    X_features = apply_polar_wavelet_transform(X_denoised)
    
    target_names = DATASET_INFO[dataset_name]['target_names']
    num_classes = len(target_names)
    num_bands = X_features.shape[2]
    
    X_train, X_test, y_train, y_test = create_disjoint_patches(
        X_features, y, window_size=window_size, train_ratio=0.15, seed=seed)
    
    models = {
        '2D-CNN':     Baseline_2DCNN(num_classes, num_bands, window_size),
        '3D-CNN':     Baseline_3DCNN(num_classes, num_bands, window_size),
        'HybridSN':   Baseline_HybridSN(num_classes, num_bands, window_size),
        'SMCNN (Ours)': SMCNN(num_classes, num_bands, window_size, use_modulation=True),
    }
    
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for name, model in models.items():
        print(f"\n  Training: {name}...")
        
        _, X_test_t, y_test_t, history, val_acc, elapsed = train_model(
            model, X_train, y_train, X_test, y_test,
            epochs=100, seed=seed, use_sfwoa=(name == 'SMCNN (Ours)'),
            tag=f"baseline_{name.replace(' ', '_')}", dataset_name=dataset_name)
        
        metrics, _, _, _ = full_evaluation(model, X_test_t, y_test_t,
                                           target_names, dataset_name,
                                           tag=f"baseline_{name.replace(' ', '_')}")
        
        # Get params/FLOPs
        dummy = torch.randn(1, window_size, window_size, num_bands, device=device)
        try:
            flops, params = thop_profile(model.to(device), inputs=(dummy,), verbose=False)
        except:
            flops, params = 0, sum(p.numel() for p in model.parameters())
        
        results.append({
            'Method': name,
            'OA': metrics['OA'],
            'AA': metrics['AA'],
            'Kappa': metrics['Kappa'],
            'Params (M)': params / 1e6,
            'FLOPs (M)': flops / 1e6,
            'Time (s)': elapsed
        })
        
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print comparison table
    print(f"\n  {'Method':<20s} {'OA':>8s} {'AA':>8s} {'κ':>8s} {'Params':>10s} {'FLOPs':>10s}")
    print(f"  {'-'*65}")
    for r in results:
        print(f"  {r['Method']:<20s} {r['OA']:>7.2f}% {r['AA']:>7.2f}% {r['Kappa']:>7.4f} {r['Params (M)']:>8.3f}M {r['FLOPs (M)']:>8.2f}M")
    
    # Save results
    with open(os.path.join(RESULTS_DIR, f'baseline_comparison_{dataset_name}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = [r['Method'] for r in results]
    oas = [r['OA'] for r in results]
    colors = ['#bdc3c7', '#85c1e9', '#5dade2', '#1a5276']
    
    bars = ax.bar(methods, oas, color=colors, width=0.6, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, oas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Overall Accuracy (%)', fontsize=13)
    ax.set_title(f'Baseline Comparison — {dataset_name}', fontsize=15, fontweight='bold')
    ax.set_ylim(min(oas) - 5, 100)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'baseline_comparison_{dataset_name}.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")

# ── RUN ──
run_baseline_comparison('IndianPines')
