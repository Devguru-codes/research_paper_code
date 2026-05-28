# =============================================================================
# Cell 4: Model Architecture — Self-Modulating CNN (SMCNN)
# =============================================================================
# Addresses Reviewer Issues:
#   1  - Core novelty (SSMRB is a data-dependent spectral attention mechanism)
#   2  - Architecture poorly defined (every layer, kernel, dimension documented)
#   3  - Math formulation (self-modulation equation: y = γ·Norm(x) + β)
#   9  - Handcrafted features (SSMRB learns its own spectral attention)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralSelfModulatingResidualBlock(nn.Module):
    """
    Spectral Self-Modulating Residual Block (SSMRB).
    
    Core Novelty: Unlike standard CNNs with fixed convolutional weights,
    SSMRB dynamically generates channel-wise scale (γ) and shift (β) 
    parameters conditioned on the input's global spectral context.
    
    Mathematical formulation:
        context = GAP(Conv(x))                    ∈ R^C
        γ = σ(W₂ · ReLU(W₁ · context))           ∈ R^C  (scale)
        β = W₄ · ReLU(W₃ · context)              ∈ R^C  (shift)
        y_c = γ_c · BN(Conv(x))_c + β_c          (per-channel modulation)
        output = ReLU(BN(Conv(y)) + x)            (residual connection)
    
    Architecture Details:
        Conv1: 3×3, same padding, no bias  → BN → ReLU → Modulation
        Conv2: 3×3, same padding, no bias  → BN → Residual Add → ReLU
        GAP:   Global Average Pooling (H×W → 1×1)
        FC1:   C → C//4 (reduction) → ReLU
        FC_γ:  C//4 → C → Sigmoid  (scale ∈ [0,1])
        FC_β:  C//4 → C            (shift ∈ R)
    """
    def __init__(self, channels):
        super().__init__()
        # Main path
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        
        # Self-modulation gating network
        reduction = max(channels // 4, 8)  # Prevent reduction to 0 for small channel counts
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.fc1      = nn.Linear(channels, reduction)
        self.fc_gamma = nn.Linear(reduction, channels)
        self.fc_beta  = nn.Linear(reduction, channels)

    def forward(self, x):
        residual = x
        
        # Standard convolution + normalization
        out = F.relu(self.bn1(self.conv1(x)))
        
        # ── Spectral Self-Modulation ──
        ctx = self.gap(out).flatten(1)             # (B, C)
        ctx = F.relu(self.fc1(ctx))                # (B, C//4)
        gamma = torch.sigmoid(self.fc_gamma(ctx))  # (B, C) ∈ [0, 1]
        beta  = self.fc_beta(ctx)                  # (B, C) ∈ R
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        out   = out * gamma + beta                 # Channel-wise affine modulation
        # ──────────────────────────────
        
        out = self.bn2(self.conv2(out))
        out = out + residual  # Skip connection
        return F.relu(out)


class StandardResidualBlock(nn.Module):
    """Standard residual block WITHOUT self-modulation (for ablation studies)."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class SMCNN(nn.Module):
    """
    Self-Modulating Convolutional Neural Network.
    
    Architecture Summary:
        Input:  (B, H, W, Bands)  — spatial-spectral patch
        Conv2d: Bands → 64, 3×3, pad=1, BN, ReLU
        SSMRB1: 64 → 64 (self-modulating residual block)
        SSMRB2: 64 → 64 (self-modulating residual block)
        GAP:    64 × H × W → 64
        Dropout: p=0.4
        FC:     64 → num_classes
    """
    def __init__(self, num_classes, num_bands, window_size, use_modulation=True):
        super().__init__()
        self.use_modulation = use_modulation
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        if use_modulation:
            self.block1 = SpectralSelfModulatingResidualBlock(64)
            self.block2 = SpectralSelfModulatingResidualBlock(64)
        else:
            self.block1 = StandardResidualBlock(64)
            self.block2 = StandardResidualBlock(64)
        
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc      = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, H, W, Bands) → (B, Bands, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.initial_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        features = self.pool(x).flatten(1)  # (B, 64)
        out = self.dropout(features)
        logits = self.fc(out)
        return logits

    def extract_features(self, x):
        """Returns 64-dim feature vector for t-SNE / analysis."""
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.initial_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.pool(x).flatten(1)

print("SMCNN architecture with SSMRB + StandardResBlock (ablation) defined.")
