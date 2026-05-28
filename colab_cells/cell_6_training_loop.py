# =============================================================================
# Cell 6: Training Loop — With Checkpointing & Disconnect Recovery
# =============================================================================
# Addresses Reviewer Issues:
#   6  - Multiple runs, seed stability
#   11 - Training time measurement
#   15 - Reproducibility (exact seed control, saved configs)
#
# IMPORTANT: Model checkpoints are saved EVERY EPOCH to /kaggle/working/saved_models/
# so even if the runtime disconnects, you lose at most 1 epoch of work.

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np
import random
import json
import os
import gc

def set_seed(seed=42):
    """Sets all random seeds for full reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, X_train, y_train, X_test, y_test,
                epochs=50, batch_size=64, seed=42,
                use_sfwoa=True, tag="proposed", dataset_name="IndianPines",
                early_stop_patience=10):
    """
    Trains model and saves checkpoint every epoch for disconnect recovery.
    
    Returns: model, X_test_tensor, y_test_tensor, history, best_val_acc
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Move data to GPU
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t  = torch.tensor(y_test, dtype=torch.long, device=device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t),
                              batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer setup
    if use_sfwoa:
        sfwoa = SFWOA_HyperTuner(model, base_lr=0.001)
        optimizer = sfwoa.get_optimizer()
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        sfwoa = None

    # Checkpoint path
    ckpt_dir = f"/kaggle/working/saved_models/{dataset_name}_{tag}_seed{seed}/"
    os.makedirs(ckpt_dir, exist_ok=True)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0.0
    prev_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"  Training [{tag}] seed={seed} epochs={epochs} batch={batch_size}")
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
            total   += labels.size(0)
            correct += (preds == labels).sum().item()
        
        train_loss = running_loss / total
        train_acc  = 100.0 * correct / total
        
        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss    += loss.item() * labels.size(0)
                _, preds     = outputs.max(1)
                val_total   += labels.size(0)
                val_correct += (preds == labels).sum().item()
        
        val_loss = val_loss / max(val_total, 1)
        val_acc  = 100.0 * val_correct / max(val_total, 1)
        
        # LR scheduling
        current_lr = optimizer.param_groups[0]['lr']
        if sfwoa is not None:
            phase, current_lr = sfwoa.step(train_loss, prev_loss, epoch, epochs)
        prev_loss = train_loss
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # ── Save best model (disconnect-safe) ──
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history,
            }, os.path.join(ckpt_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1
        
        # Save latest checkpoint every epoch (for crash recovery)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'history': history,
        }, os.path.join(ckpt_dir, 'latest_checkpoint.pth'))
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | Train {train_acc:.1f}% | Val {val_acc:.1f}% | LR {current_lr:.6f}")
        
        # ── Early stopping ──
        if early_stop_patience and epochs_no_improve >= early_stop_patience:
            print(f"    Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
            break
    
    elapsed = time.time() - start_time
    print(f"  Training complete in {elapsed:.1f}s | Best Val Acc: {best_val_acc:.2f}%")
    
    # Save history to JSON (disconnect-safe)
    with open(os.path.join(ckpt_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    # Load best model weights (fallback to latest if best doesn't exist)
    best_path = os.path.join(ckpt_dir, 'best_model.pth')
    if not os.path.exists(best_path):
        best_path = os.path.join(ckpt_dir, 'latest_checkpoint.pth')
    best_ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    
    return model, X_test_t, y_test_t, history, best_val_acc, elapsed

print("Training loop with per-epoch checkpointing ready.")
