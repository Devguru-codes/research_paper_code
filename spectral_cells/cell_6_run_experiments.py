# =============================================================================
# Cell 6: RUN ALL EXPERIMENTS
# =============================================================================
# Runs SpectralFormer + SSFTT on all 5 datasets × 3 seeds = 30 runs
# Estimated time: ~30-60 minutes on T4
#
# IMPORTANT: If Kaggle disconnects, just re-run this cell.
# Already-completed results are saved to JSON and will be skipped.

SEEDS = [42, 100, 2024]
WINDOW_SIZE = 15
TRAIN_RATIO = 0.15
EPOCHS = 100
EARLY_STOP_PATIENCE = 10
BATCH_SIZE = 64
LR = 0.001

# ── Which datasets to run ──
# Comment out any you want to skip to save time
DATASETS_TO_RUN = [
    'IndianPines',   # ~200 bands, 16 classes
    'PaviaU',        # ~103 bands, 9 classes
    'Botswana',      # ~145 bands, 14 classes
    'KSC',           # ~176 bands, 13 classes
    # 'WHU_Hi',      # ~274 bands, 16 classes — UNCOMMENT if you uploaded WHU dataset
]

MODELS_TO_RUN = ['SpectralFormer', 'SSFTT']

# ── Storage for all results ──
all_results = {}

for dataset_name in DATASETS_TO_RUN:
    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*70}")
    
    # Load and normalize
    X_raw, y_raw = load_dataset(dataset_name)
    X_norm = normalize(X_raw.copy())
    num_bands = X_norm.shape[2]
    num_classes = DATASET_INFO[dataset_name]['num_classes']
    
    for model_name in MODELS_TO_RUN:
        print(f"\n  ── Model: {model_name} ──")
        
        result_file = os.path.join(RESULTS_DIR, f'{model_name}_{dataset_name}_results.json')
        
        # Skip if already completed
        if os.path.exists(result_file):
            print(f"  ⏭ Already completed — loading from {result_file}")
            with open(result_file) as f:
                all_results[f'{model_name}_{dataset_name}'] = json.load(f)
            continue
        
        seed_metrics = []
        seed_times = []
        
        for seed in SEEDS:
            # Create patches (IDENTICAL split to main paper)
            X_train, X_test, y_train, y_test = create_disjoint_patches(
                X_norm, y_raw, window_size=WINDOW_SIZE,
                train_ratio=TRAIN_RATIO, seed=seed
            )
            
            # Create model
            if model_name == 'SpectralFormer':
                model = SpectralFormer(
                    num_classes=num_classes,
                    num_bands=num_bands,
                    patch_size=WINDOW_SIZE,
                    embed_dim=64,
                    depth=4,
                    num_heads=4,
                    dropout=0.1
                )
            elif model_name == 'SSFTT':
                model = SSFTT(
                    num_classes=num_classes,
                    num_bands=num_bands,
                    patch_size=WINDOW_SIZE,
                    embed_dim=64,
                    depth=2,
                    num_heads=4,
                    dropout=0.1
                )
            
            print(f"    Params: {count_params(model)/1e6:.3f}M")
            
            # Train
            model, X_test_t, y_test_t, best_acc, elapsed = train_transformer(
                model, X_train, y_train, X_test, y_test,
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                seed=seed, model_name=model_name, dataset_name=dataset_name,
                early_stop_patience=EARLY_STOP_PATIENCE
            )
            
            # Evaluate
            metrics = evaluate_model(model, X_test_t, y_test_t, num_classes, dataset_name, model_name)
            metrics['seed'] = seed
            metrics['train_time'] = round(elapsed, 2)
            seed_metrics.append(metrics)
            seed_times.append(elapsed)
            
            print(f"    Seed {seed}: OA={metrics['OA']:.2f}%, AA={metrics['AA']:.2f}%, "
                  f"Kappa={metrics['Kappa']:.4f}, F1={metrics['F1']:.2f}%")
            
            # Clean up GPU memory
            del model, X_test_t, y_test_t, X_train, X_test
            torch.cuda.empty_cache()
            gc.collect()
        
        # Compute mean ± std
        summary = {
            'model': model_name,
            'dataset': dataset_name,
            'params_M': round(count_params(
                SpectralFormer(num_classes, num_bands, WINDOW_SIZE) if model_name == 'SpectralFormer'
                else SSFTT(num_classes, num_bands, WINDOW_SIZE)
            ) / 1e6, 3),
            'seeds': SEEDS,
            'per_seed': seed_metrics,
            'mean_OA':   round(np.mean([m['OA'] for m in seed_metrics]), 2),
            'std_OA':    round(np.std([m['OA'] for m in seed_metrics]), 2),
            'mean_AA':   round(np.mean([m['AA'] for m in seed_metrics]), 2),
            'std_AA':    round(np.std([m['AA'] for m in seed_metrics]), 2),
            'mean_Kappa': round(np.mean([m['Kappa'] for m in seed_metrics]), 4),
            'std_Kappa':  round(np.std([m['Kappa'] for m in seed_metrics]), 4),
            'mean_Precision': round(np.mean([m['Precision'] for m in seed_metrics]), 2),
            'mean_Recall':    round(np.mean([m['Recall'] for m in seed_metrics]), 2),
            'mean_F1':        round(np.mean([m['F1'] for m in seed_metrics]), 2),
            'mean_time':      round(np.mean(seed_times), 2),
        }
        
        # Save
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        all_results[f'{model_name}_{dataset_name}'] = summary
        
        print(f"\n  ✅ {model_name} on {dataset_name}:")
        print(f"     OA: {summary['mean_OA']:.2f} ± {summary['std_OA']:.2f}%")
        print(f"     AA: {summary['mean_AA']:.2f} ± {summary['std_AA']:.2f}%")
        print(f"     Kappa: {summary['mean_Kappa']:.4f}")
        print(f"     Saved: {result_file}")
        
        torch.cuda.empty_cache()
        gc.collect()

print(f"\n{'='*70}")
print("  ALL EXPERIMENTS COMPLETE")
print(f"{'='*70}")
