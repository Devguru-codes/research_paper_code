# =============================================================================
# Cell 3: SpectralFormer Model
# =============================================================================
# Reference: Hong et al., "SpectralFormer: Rethinking Hyperspectral Image
#            Classification with Transformers," IEEE TGRS, 2022.
#
# Adapted for FAIR comparison: same patch size (15×15), same input format

import math
from einops import rearrange

class SpectralEmbedding(nn.Module):
    """Embeds spectral bands as tokens with positional encoding."""
    def __init__(self, num_bands, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Each band's spatial patch → flattened → projected to embed_dim
        self.proj = nn.Linear(patch_size * patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_bands + 1, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (B, H, W, Bands) → (B, Bands, H, W)
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        # Each band becomes a token: (B, C, H*W) → (B, C, embed_dim)
        x = x.reshape(B, C, H * W)
        x = self.proj(x)  # (B, C, embed_dim)
        
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, C+1, embed_dim)
        
        # Add positional encoding (truncate or pad if needed)
        if x.size(1) <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :x.size(1), :]
        else:
            # Interpolate positional embeddings for larger band counts
            pos = F.interpolate(
                self.pos_embed.permute(0, 2, 1), size=x.size(1), mode='linear'
            ).permute(0, 2, 1)
            x = x + pos
        
        return self.norm(x)


class CrossLayerAdaptiveFusion(nn.Module):
    """Cross-layer adaptive fusion (CAF) from SpectralFormer."""
    def __init__(self, embed_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x_current, x_previous):
        combined = torch.cat([x_current, x_previous], dim=-1)
        gate = self.gate(combined)
        return gate * x_current + (1 - gate) * x_previous


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SpectralFormer(nn.Module):
    """
    SpectralFormer with Cross-Layer Adaptive Fusion.
    
    Architecture:
        Spectral Embedding → [TransformerBlock + CAF] × depth → CLS → FC
    """
    def __init__(self, num_classes, num_bands, patch_size=15,
                 embed_dim=64, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embed = SpectralEmbedding(num_bands, patch_size, embed_dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.cafs = nn.ModuleList([
            CrossLayerAdaptiveFusion(embed_dim)
            for _ in range(depth - 1)  # CAF between consecutive layers
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (B, H, W, Bands)
        x = self.embed(x)  # (B, bands+1, embed_dim)
        
        prev = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i > 0:  # Apply CAF from second layer onwards
                x = self.cafs[i - 1](x, prev)
            prev = x
        
        x = self.norm(x)
        cls_token = x[:, 0]  # CLS token
        return self.head(self.dropout(cls_token))
    
    def extract_features(self, x):
        x = self.embed(x)
        prev = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i > 0:
                x = self.cafs[i - 1](x, prev)
            prev = x
        x = self.norm(x)
        return x[:, 0]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick test
_test = SpectralFormer(num_classes=16, num_bands=200, patch_size=15)
print(f"SpectralFormer ready. Test params: {count_params(_test)/1e6:.3f}M")
del _test
