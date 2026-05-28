# =============================================================================
# Cell 7: Results Summary & Comparison Table
# =============================================================================
# Generates the final comparison table matching the paper's format
# Also measures FLOPs and inference latency

import pandas as pd

print("=" * 80)
print("  TRANSFORMER BASELINE RESULTS SUMMARY")
print("=" * 80)

# ── Load all results ──
result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('_results.json')]
all_loaded = {}
for rf in sorted(result_files):
    with open(os.path.join(RESULTS_DIR, rf)) as f:
        data = json.load(f)
        key = f"{data['model']}_{data['dataset']}"
        all_loaded[key] = data

# ── Print per-dataset comparison ──
for dataset_name in DATASETS_TO_RUN:
    print(f"\n{'─'*60}")
    print(f"  {dataset_name}")
    print(f"{'─'*60}")
    
    rows = []
    for model_name in MODELS_TO_RUN:
        key = f"{model_name}_{dataset_name}"
        if key in all_loaded:
            d = all_loaded[key]
            rows.append({
                'Model': model_name,
                'OA (%)': f"{d['mean_OA']:.2f} ± {d['std_OA']:.2f}",
                'AA (%)': f"{d['mean_AA']:.2f} ± {d['std_AA']:.2f}",
                'Kappa': f"{d['mean_Kappa']:.4f}",
                'F1 (%)': f"{d['mean_F1']:.2f}",
                'Params (M)': f"{d['params_M']:.3f}",
                'Time (s)': f"{d['mean_time']:.1f}"
            })
    
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

# ── Measure inference latency ──
print(f"\n{'─'*60}")
print("  INFERENCE LATENCY (ms per patch)")
print(f"{'─'*60}")

for dataset_name in DATASETS_TO_RUN:
    X_raw, y_raw = load_dataset(dataset_name)
    num_bands = X_raw.shape[2]
    num_classes = DATASET_INFO[dataset_name]['num_classes']
    
    for model_name in MODELS_TO_RUN:
        if model_name == 'SpectralFormer':
            model = SpectralFormer(num_classes, num_bands, WINDOW_SIZE).to(DEVICE)
        else:
            model = SSFTT(num_classes, num_bands, WINDOW_SIZE).to(DEVICE)
        
        model.eval()
        dummy = torch.randn(1, WINDOW_SIZE, WINDOW_SIZE, num_bands).to(DEVICE)
        
        # Warmup
        with torch.no_grad():
            for _ in range(20):
                _ = model(dummy)
        torch.cuda.synchronize()
        
        # Measure
        start = time.time()
        with torch.no_grad():
            for _ in range(200):
                _ = model(dummy)
        torch.cuda.synchronize()
        latency_ms = (time.time() - start) / 200 * 1000
        
        print(f"  {model_name:20s} on {dataset_name:15s}: {latency_ms:.2f} ms/patch, "
              f"Params: {count_params(model)/1e6:.3f}M")
        
        del model, dummy
        torch.cuda.empty_cache()

# ── Save combined results for paper update ──
combined = {
    'experiment_config': {
        'seeds': SEEDS,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'window_size': WINDOW_SIZE,
        'train_ratio': TRAIN_RATIO,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'note': 'Identical protocol to main paper (no SFWOA for transformers, uses cosine schedule)'
    },
    'results': all_loaded
}

combined_path = os.path.join(RESULTS_DIR, 'transformer_baselines_combined.json')
with open(combined_path, 'w') as f:
    json.dump(combined, f, indent=2)

print(f"\n✅ Combined results saved: {combined_path}")
print("\n📋 COPY THESE NUMBERS INTO create_paper.js Table 11!")
print("   Then regenerate the docx with: node create_paper.js")
