# =============================================================================
# Cell 11: Full-Scene Classification Maps
# =============================================================================
# MUST-HAVE for any HSI paper. Generates the colorful pixel-wise classification
# map for the entire scene, plus ground truth side-by-side.
# Addresses Reviewer Issues 14 (figures), 18 (qualitative analysis)

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os

def generate_classification_map(model, X_full, y_full, dataset_name, window_size=11, pca_obj=None):
    """
    Generates a full-scene classification map by predicting every labeled pixel.
    
    Args:
        model: Trained SMCNN model
        X_full: Full preprocessed image cube (H, W, Bands) — after PCA+AFDKF+PLCWT
        y_full: Ground truth label map (H, W)
        dataset_name: For saving
        window_size: Patch size used during training
    """
    print(f"  Generating classification map for {dataset_name}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    margin = (window_size - 1) // 2
    h, w = y_full.shape
    
    # Pad the image
    padded_X = np.zeros((h + 2*margin, w + 2*margin, X_full.shape[2]), dtype=np.float32)
    padded_X[margin:margin+h, margin:margin+w, :] = X_full
    
    # Predict every labeled pixel
    pred_map = np.zeros((h, w), dtype=np.int32)
    labeled_pixels = np.argwhere(y_full > 0)
    
    # Batch prediction for speed
    batch_size = 512
    for start in range(0, len(labeled_pixels), batch_size):
        end = min(start + batch_size, len(labeled_pixels))
        batch_coords = labeled_pixels[start:end]
        
        patches = []
        for r, c in batch_coords:
            patch = padded_X[r:r+window_size, c:c+window_size, :]
            patches.append(patch)
        
        patches_t = torch.tensor(np.array(patches), dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs = model(patches_t)
            _, preds = outputs.max(1)
        
        for i, (r, c) in enumerate(batch_coords):
            pred_map[r, c] = preds[i].item() + 1  # 1-indexed like ground truth
    
    # Get target names and create color map
    target_names = DATASET_INFO[dataset_name]['target_names']
    num_classes = len(target_names)
    
    # Use a high-quality colormap
    base_colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_classes)))
    colors = ['black'] + [base_colors[i] for i in range(num_classes)]  # black for background
    cmap = ListedColormap(colors[:num_classes + 1])
    
    # Plot: Ground Truth | Predicted | Overlay
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Ground Truth
    axes[0].imshow(y_full, cmap=cmap, vmin=0, vmax=num_classes)
    axes[0].set_title(f'Ground Truth — {dataset_name}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Predicted
    axes[1].imshow(pred_map, cmap=cmap, vmin=0, vmax=num_classes)
    axes[1].set_title(f'Predicted (SMCNN) — {dataset_name}', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Error map (misclassifications in red)
    error_map = np.zeros((h, w, 3), dtype=np.float32)
    labeled_mask = y_full > 0
    correct_mask = (pred_map == y_full) & labeled_mask
    wrong_mask = (pred_map != y_full) & labeled_mask
    error_map[correct_mask] = [0.2, 0.8, 0.2]   # Green = correct
    error_map[wrong_mask] = [1.0, 0.0, 0.0]      # Red = misclassified
    
    axes[2].imshow(error_map)
    axes[2].set_title(f'Error Map (Red=Wrong) — {dataset_name}', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Legend
    patches_legend = [mpatches.Patch(color=colors[i+1], label=target_names[i]) 
                      for i in range(num_classes)]
    fig.legend(handles=patches_legend, loc='lower center', ncol=min(8, num_classes),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'classification_map_{dataset_name}.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")
    
    # Calculate and print error statistics
    total_labeled = labeled_mask.sum()
    total_correct = correct_mask.sum()
    total_wrong = wrong_mask.sum()
    print(f"  Map stats: {total_correct}/{total_labeled} correct ({100*total_correct/total_labeled:.2f}%), {total_wrong} errors")
    
    return pred_map


def run_classification_maps():
    """Run classification maps for all available datasets."""
    datasets_to_map = ['IndianPines', 'PaviaU', 'Salinas', 'Botswana', 'KSC', 'Houston2013', 'WHU_Hi']
    window_size = 11
    pca_components = 30
    
    for ds_name in datasets_to_map:
        print(f"\n  === Classification Map: {ds_name} ===")
        try:
            X, y = load_dataset(ds_name)
        except Exception as e:
            print(f"  SKIPPED: {e}")
            continue
        
        X_pca, _ = apply_pca(X, num_components=pca_components)
        X_denoised = apply_kalman_denoising(X_pca)
        X_features = apply_polar_wavelet_transform(X_denoised)
        
        # Load best saved model
        model_path = f'/kaggle/working/saved_models/best_model_{ds_name}.pth'
        if not os.path.exists(model_path):
            print(f"  SKIPPED: No saved model for {ds_name}")
            continue
        
        target_names = DATASET_INFO[ds_name]['target_names']
        num_classes = len(target_names)
        num_bands = X_features.shape[2]
        
        model = SMCNN(num_classes=num_classes, num_bands=num_bands,
                      window_size=window_size, use_modulation=True)
        state_dict = torch.load(model_path, weights_only=False)
        # Remove thop-injected keys (total_ops, total_params)
        state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
        model.load_state_dict(state_dict, strict=False)
        model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        generate_classification_map(model, X_features, y, ds_name, window_size)
        
        del model, X, X_pca, X_denoised, X_features
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ── RUN ──
run_classification_maps()
