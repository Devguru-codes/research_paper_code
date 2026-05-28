# =============================================================================
# Cell 14: Grad-CAM Spatial Attention Visualization
# =============================================================================
# Shows WHERE the model focuses for each class prediction.
# Addresses Reviewer Issues 4 (theoretical insight), 18 (qualitative analysis)
#
# Grad-CAM computes gradient-weighted class activation maps by:
#   1. Forward pass to get prediction
#   2. Backward pass to get gradients of target class w.r.t. last conv layer
#   3. Global average pool the gradients → channel weights
#   4. Weighted sum of feature maps → heatmap

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

class GradCAM:
    """Gradient-weighted Class Activation Mapping for SMCNN."""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        # Hook into the last block (block2)
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.model.block2.register_forward_hook(forward_hook)
        self.model.block2.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generates a Grad-CAM heatmap for a single input patch.
        
        Args:
            input_tensor: (1, H, W, Bands)
            target_class: int or None (auto-selects predicted class)
        Returns:
            heatmap: (H, W) numpy array, normalized to [0, 1]
        """
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Global average pool gradients → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only positive contributions
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, target_class


def plot_gradcam_samples(model, X_test_t, y_test_t, target_names, dataset_name, num_samples=8):
    """
    Generates Grad-CAM visualizations for random test samples.
    Shows: Input RGB composite | Grad-CAM heatmap | Overlay
    """
    print(f"  Generating Grad-CAM visualizations for {dataset_name}...")
    
    device = X_test_t.device
    gradcam = GradCAM(model)
    
    # Select random samples from different classes
    unique_classes = torch.unique(y_test_t).cpu().numpy()
    selected_indices = []
    for cls in unique_classes[:num_samples]:
        cls_indices = (y_test_t == cls).nonzero(as_tuple=True)[0]
        if len(cls_indices) > 0:
            idx = cls_indices[np.random.randint(len(cls_indices))].item()
            selected_indices.append(idx)
    
    num_show = min(len(selected_indices), num_samples)
    fig, axes = plt.subplots(num_show, 3, figsize=(12, 3 * num_show))
    if num_show == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(selected_indices[:num_show]):
        sample = X_test_t[idx:idx+1]  # (1, H, W, Bands)
        true_label = y_test_t[idx].item()
        
        # Generate Grad-CAM
        cam, pred_class = gradcam.generate(sample.clone())
        
        # Create RGB composite from first 3 PCA bands
        patch_np = sample.squeeze().cpu().numpy()  # (H, W, Bands)
        rgb = patch_np[:, :, :3]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        
        # Resize CAM to patch size
        from scipy.ndimage import zoom as scipy_zoom
        if cam.shape[0] != patch_np.shape[0]:
            cam = scipy_zoom(cam, (patch_np.shape[0]/cam.shape[0], 
                                    patch_np.shape[1]/cam.shape[1]), order=1)
        
        # Plot
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f'True: {target_names[true_label]}', fontsize=9)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(cam, cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Grad-CAM (Pred: {target_names[pred_class]})', fontsize=9)
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(rgb)
        axes[i, 2].imshow(cam, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        correct = "✓" if pred_class == true_label else "✗"
        axes[i, 2].set_title(f'Overlay {correct}', fontsize=9)
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Grad-CAM Attention Maps — {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f'gradcam_{dataset_name}.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {path}")


def run_gradcam():
    """Load best model and generate Grad-CAM for each dataset."""
    datasets = ['IndianPines', 'PaviaU', 'Salinas']
    window_size = 11
    
    for ds_name in datasets:
        model_path = f'/kaggle/working/saved_models/best_model_{ds_name}.pth'
        if not os.path.exists(model_path):
            print(f"  SKIPPED {ds_name}: no saved model")
            continue
        
        try:
            X, y = load_dataset(ds_name)
        except:
            continue
        
        X_pca, _ = apply_pca(X, num_components=30)
        X_denoised = apply_kalman_denoising(X_pca)
        X_features = apply_polar_wavelet_transform(X_denoised)
        
        target_names = DATASET_INFO[ds_name]['target_names']
        num_classes = len(target_names)
        num_bands = X_features.shape[2]
        
        # Get a small test set for visualization
        _, X_test, _, y_test = create_disjoint_patches(
            X_features, y, window_size=window_size, train_ratio=0.15, seed=42)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SMCNN(num_classes=num_classes, num_bands=num_bands,
                      window_size=window_size, use_modulation=True)
        sd = torch.load(model_path, weights_only=False)
        sd = {k: v for k, v in sd.items() if 'total_ops' not in k and 'total_params' not in k}
        model.load_state_dict(sd, strict=False)
        model = model.to(device)
        
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)
        
        plot_gradcam_samples(model, X_test_t, y_test_t, target_names, ds_name)
        
        del model, X, X_pca, X_denoised, X_features, X_test_t
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ── RUN ──
run_gradcam()
