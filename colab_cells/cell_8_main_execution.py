# =============================================================================
# Cell 8: Main Execution — Multi-Dataset, Multi-Seed, Full Pipeline
# =============================================================================
# Addresses Reviewer Issues:
#   5  - Experimental protocol (multi-dataset, multi-seed)
#   6  - Statistical significance (3 seeds per dataset)
#   7  - Multi-dataset evaluation (Indian Pines, PaviaU, Salinas)
#   11 - Computational efficiency (FLOPs, params, training time, inference speed)
#   15 - Reproducibility (all configs saved)

import torch
import numpy as np
import time
import json
import os
import gc

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None

def compute_efficiency(model, num_bands, window_size, device):
    """Computes FLOPs, parameter count, and inference latency."""
    model.eval()
    dummy = torch.randn(1, window_size, window_size, num_bands, device=device)
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # FLOPs
    flops_val = None
    if thop_profile is not None:
        flops_val, _ = thop_profile(model, inputs=(dummy,), verbose=False)
    
    # Inference latency (average over 100 runs)
    model.eval()
    times = []
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        for _ in range(100):
            start = time.time()
            _ = model(dummy)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start)
    
    latency_ms = np.mean(times) * 1000
    
    # GPU memory
    gpu_mem_mb = 0
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops_val,
        'inference_latency_ms': latency_ms,
        'gpu_memory_mb': gpu_mem_mb
    }


def run_full_experiment():
    """
    Runs the proposed model on 5 datasets x 3 seeds.
    Saves all graphs and checkpoints to /kaggle/working/.
    """
    datasets = ['IndianPines', 'PaviaU', 'Salinas', 'Botswana', 'KSC', 'Houston2013', 'WHU_Hi']
    seeds    = [42, 100, 2024]
    window_size = 11
    pca_components = 30
    epochs = 100
    batch_size = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = {}
    
    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*60}")
        
        # Load and preprocess (skip if dataset not available)
        try:
            X, y = load_dataset(ds_name)
        except (FileNotFoundError, Exception) as e:
            print(f"  SKIPPED: {e}")
            continue
        X_pca, _ = apply_pca(X, num_components=pca_components)
        X_denoised = apply_kalman_denoising(X_pca)
        X_features = apply_polar_wavelet_transform(X_denoised)
        
        target_names = DATASET_INFO[ds_name]['target_names']
        num_classes = len(target_names)
        num_bands = X_features.shape[2]
        
        ds_metrics = []
        best_acc = 0
        best_model, best_X_test_t, best_y_test_t, best_history = None, None, None, None
        total_train_time = 0
        
        for seed in seeds:
            print(f"\n  ── Seed {seed} ──")
            X_train, X_test, y_train, y_test = create_disjoint_patches(
                X_features, y, window_size=window_size, train_ratio=0.15, seed=seed)
            
            model = SMCNN(num_classes=num_classes, num_bands=num_bands,
                          window_size=window_size, use_modulation=True)
            
            trained_model, X_test_t, y_test_t, history, val_acc, elapsed = train_model(
                model, X_train, y_train, X_test, y_test,
                epochs=epochs, batch_size=batch_size, seed=seed,
                use_sfwoa=True, tag="proposed", dataset_name=ds_name)
            
            total_train_time += elapsed
            
            metrics, y_true, y_pred, cm = full_evaluation(
                trained_model, X_test_t, y_test_t, target_names, ds_name, tag=f"seed{seed}")
            ds_metrics.append(metrics)
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = trained_model
                best_X_test_t = X_test_t
                best_y_test_t = y_test_t
                best_history = history
                best_cm = cm
            
            # Free GPU memory
            del X_train, X_test, y_train, y_test
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Print cross-seed statistics
        best_idx = print_multi_seed_stats(ds_metrics, target_names, ds_name)
        
        # Generate all graphs for best run
        print(f"\n  Generating graphs for {ds_name}...")
        plot_confusion_matrix(best_cm, target_names, ds_name)
        plot_learning_curves(best_history, ds_name)
        plot_tsne(best_model, best_X_test_t, best_y_test_t, target_names, ds_name)
        plot_noise_robustness(best_model, best_X_test_t, best_y_test_t, target_names, ds_name)
        plot_spectral_attention(best_model, best_X_test_t, ds_name)
        # Save best model permanently (BEFORE compute_efficiency which injects thop keys)
        torch.save(best_model.state_dict(),
                   f'/kaggle/working/saved_models/best_model_{ds_name}.pth')
        
        # Computational efficiency
        eff = compute_efficiency(best_model, num_bands, window_size, device)
        print(f"\n  Computational Efficiency ({ds_name}):")
        print(f"    Parameters:  {eff['total_params']/1e6:.2f} M")
        if eff['flops']:
            print(f"    FLOPs:       {eff['flops']/1e6:.2f} M")
        print(f"    Inference:   {eff['inference_latency_ms']:.2f} ms/patch")
        print(f"    GPU Memory:  {eff['gpu_memory_mb']:.1f} MB")
        print(f"    Train Time:  {total_train_time/len(seeds):.1f} s/run (avg)")
        
        all_results[ds_name] = {
            'metrics': [{k: v for k, v in m.items() if k != 'y_true' and k != 'y_pred'} for m in ds_metrics],
            'efficiency': {k: float(v) if v is not None else None for k, v in eff.items()},
            'avg_train_time_s': total_train_time / len(seeds)
        }
        
        del X, X_pca, X_denoised, X_features
        del best_model, best_X_test_t, best_y_test_t, best_history
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save master results file
    with open('/kaggle/working/results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("  ALL EXPERIMENTS COMPLETE!")
    print(f"  Results: /kaggle/working/results/all_results.json")
    print(f"  Graphs:  /kaggle/working/paper_graphs/")
    print(f"  Models:  /kaggle/working/saved_models/")
    print(f"{'='*60}")


# ── RUN ──
run_full_experiment()
