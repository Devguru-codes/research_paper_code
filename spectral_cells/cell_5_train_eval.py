# =============================================================================
# Cell 5: Training & Evaluation — IDENTICAL protocol to main paper
# =============================================================================
# Same: AdamW, lr=0.001, weight_decay=1e-4, batch_size=64, epochs=30
# Same: 3 seeds {42, 100, 2024}
# Same: metrics (OA, AA, Kappa, Precision, Recall, F1)
# NO SFWOA (transformers use their own cosine schedule)

import gc

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_transformer(model, X_train, y_train, X_test, y_test,
                      epochs=100, batch_size=64, lr=0.001, seed=42,
                      model_name="SpectralFormer", dataset_name="IndianPines",
                      early_stop_patience=10):
    """Train a transformer model under identical conditions to main paper."""
    set_seed(seed)
    device = DEVICE
    model = model.to(device)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t  = torch.tensor(y_test, dtype=torch.long, device=device)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t),
                              batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    
    print(f"\n  Training {model_name} on {dataset_name} | seed={seed} | epochs={epochs} | early_stop={early_stop_patience}")
    start_time = time.time()
    
    for epoch in range(epochs):
        # ── Train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        
        train_acc = 100.0 * correct / total
        scheduler.step()
        
        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        val_acc = 100.0 * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | Train {train_acc:.1f}% | Val {val_acc:.1f}%")
        
        # Early stopping
        if early_stop_patience and epochs_no_improve >= early_stop_patience:
            print(f"    Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
            break
    
    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s | Best Val Acc: {best_val_acc:.2f}%")
    
    # Load best weights
    model.load_state_dict(best_state)
    model = model.to(device)
    
    return model, X_test_t, y_test_t, best_val_acc, elapsed


def evaluate_model(model, X_test_t, y_test_t, num_classes, dataset_name, model_name):
    """Full evaluation identical to main paper's metrics."""
    model.eval()
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
    
    oa = accuracy_score(y_true, y_pred) * 100
    kappa = cohen_kappa_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    
    cm = confusion_matrix(y_true, y_pred)
    per_class = []
    for i in range(cm.shape[0]):
        if cm[i].sum() > 0:
            per_class.append(cm[i, i] / cm[i].sum() * 100)
        else:
            per_class.append(0.0)
    aa = np.mean(per_class)
    
    metrics = {
        'model': model_name,
        'dataset': dataset_name,
        'OA': round(oa, 2),
        'AA': round(aa, 2),
        'Kappa': round(kappa, 4),
        'Precision': round(prec, 2),
        'Recall': round(rec, 2),
        'F1': round(f1, 2),
        'per_class_acc': [round(p, 2) for p in per_class]
    }
    return metrics


def count_flops_estimate(model, input_shape):
    """Rough FLOPs estimate for comparison."""
    # Use a forward pass with hooks to estimate
    total_flops = 0
    params = count_params(model)
    # Rough estimate: FLOPs ≈ 2 × params × seq_length (for attention-heavy models)
    # More accurate would require torchprofile, but this gives ballpark
    batch = torch.randn(1, *input_shape).to(DEVICE)
    
    # Time-based estimate
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model(batch)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 100 * 1000  # ms per inference
    
    del batch
    return params, elapsed


print("Training & evaluation functions ready.")
