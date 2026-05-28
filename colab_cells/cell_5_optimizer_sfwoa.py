# =============================================================================
# Cell 5: Optimizer — SFWOA (Superb Fairy-Wren Optimization Algorithm)
# =============================================================================
# Addresses Reviewer Issue 10:
#   "The authors must prove why SFWOA is fundamentally required,
#    why SGD/Adam are insufficient."
#
# Design rationale: We do NOT use SFWOA to optimize millions of CNN weights.
# Instead, SFWOA dynamically tunes the learning rate schedule by detecting
# loss plateaus (local minima) and applying Lévy-flight perturbations to
# escape them. This is analogous to cyclical learning rates but biologically
# motivated and adaptive.

import torch.optim as optim
import numpy as np

class SFWOA_HyperTuner:
    """
    Superb Fairy-Wren Optimization Algorithm — Learning Rate Scheduler.
    
    Three phases per the original SFOA paper:
      1. Juvenile Growth (early epochs): High LR for exploration
      2. Breeding/Feeding (mid epochs): Gradual LR decay for exploitation 
      3. Predator Evasion (plateau detected): Lévy flight LR perturbation
    
    Mathematical formulation for evasion phase:
        lr_{t+1} = lr_base × α ⊕ Lévy(β)
        where α is a step size and Lévy(β) provides heavy-tailed random jumps.
    """
    def __init__(self, model, base_lr=0.001, weight_decay=1e-4, patience=3):
        self.model = model
        self.base_lr = base_lr
        self.patience = patience
        self.plateau_count = 0
        self.optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        self.lr_history = []

    def levy_flight(self, beta=1.5):
        """Generates a Lévy flight step (Mantegna's algorithm)."""
        from math import gamma as math_gamma
        sigma = (math_gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math_gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = u / (abs(v) ** (1 / beta))
        return step

    def step(self, current_loss, prev_loss, epoch, max_epochs):
        """Called after each epoch to adjust learning rate."""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Detect plateau
        improvement = prev_loss - current_loss
        if improvement < 1e-4 and improvement >= 0:
            self.plateau_count += 1
        else:
            self.plateau_count = 0
        
        if self.plateau_count >= self.patience:
            # Phase 3: Predator Evasion — Lévy flight perturbation
            levy_step = abs(self.levy_flight())
            new_lr = self.base_lr * max(0.1, min(levy_step, 3.0))
            self.plateau_count = 0
            phase = "EVASION (Lévy)"
        else:
            # Phase 2: Breeding — cosine-like decay
            progress = epoch / max_epochs
            new_lr = self.base_lr * (0.5 * (1 + np.cos(np.pi * progress)))
            new_lr = max(new_lr, 1e-6)
            phase = "BREEDING (decay)"

        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr
        
        self.lr_history.append(new_lr)
        return phase, new_lr

    def get_optimizer(self):
        return self.optimizer

print("SFWOA optimizer with proper Lévy flight implementation ready.")
