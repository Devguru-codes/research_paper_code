# =============================================================================
# Cell 4: SSFTT Model
# =============================================================================
# Reference: Sun et al., "Spectral-Spatial Feature Tokenization Transformer
#            for HSI Classification," IEEE TGRS, 2022.
#
# SSFTT = Spectral-Spatial Feature Tokenization + Gaussian Weighted Transformer
# Hybrid: 3D-Conv tokenizer → Transformer encoder → classifier

class SpectralSpatialTokenizer(nn.Module):
    """
    3D Conv tokenizer: extracts spectral-spatial tokens from HSI patches.
    Conv3D reduces spectral dim while preserving spatial structure,
    then tokens are created by flattening the spatial grid.
    """
    def __init__(self, num_bands, embed_dim=64):
        super().__init__()
        # Spectral reduction via 3D convolution
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=(2, 1, 1), padding=(3, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # After 3 stride-2 reductions: spectral_dim ≈ num_bands / 8
        # We'll flatten spectral × channels and project to embed_dim
        self._probe_dim = None  # Will be computed in forward
        self.proj = None  # Lazy init
        self.embed_dim = embed_dim
    
    def _init_proj(self, feat_dim):
        self.proj = nn.Linear(feat_dim, self.embed_dim).to(
            next(self.conv3d_1.parameters()).device
        )
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        # x: (B, H, W, Bands) → (B, 1, Bands, H, W)
        x = x.permute(0, 3, 1, 2).unsqueeze(1)
        
        x = self.conv3d_1(x)   # (B, 8, Bands/2, H, W)
        x = self.conv3d_2(x)   # (B, 16, Bands/4, H, W)
        x = self.conv3d_3(x)   # (B, 32, Bands/8, H, W)
        
        B, C, D, H, W = x.shape
        # Reshape: spatial tokens = H*W, each token = C*D features
        x = x.permute(0, 3, 4, 1, 2)  # (B, H, W, C, D)
        x = x.reshape(B, H * W, C * D)  # (B, H*W, C*D)
        
        # Lazy projection init
        if self.proj is None:
            self._init_proj(C * D)
        
        x = self.proj(x)  # (B, H*W, embed_dim)
        return x, H, W


class GaussianWeightedSelfAttention(nn.Module):
    """
    Self-attention with Gaussian position bias (SSFTT's key contribution).
    Spatial tokens that are physically closer get higher attention.
    """
    def __init__(self, embed_dim, num_heads, spatial_size=15):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Gaussian positional bias
        self.sigma = nn.Parameter(torch.tensor(2.0))
        coords = torch.stack(torch.meshgrid(
            torch.arange(spatial_size), torch.arange(spatial_size), indexing='ij'
        ), dim=-1).float().reshape(-1, 2)
        dist = torch.cdist(coords, coords, p=2)
        self.register_buffer('dist_matrix', dist)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add Gaussian spatial bias
        if N <= self.dist_matrix.size(0):
            gauss_bias = torch.exp(-self.dist_matrix[:N, :N] ** 2 / (2 * self.sigma ** 2 + 1e-6))
            attn = attn + gauss_bias.unsqueeze(0).unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SSFTTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, spatial_size=15):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = GaussianWeightedSelfAttention(embed_dim, num_heads, spatial_size)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SSFTT(nn.Module):
    """
    Spectral-Spatial Feature Tokenization Transformer.
    
    Architecture:
        3D Conv Tokenizer → [Gaussian-Weighted Transformer] × depth → GAP → FC
    """
    def __init__(self, num_classes, num_bands, patch_size=15,
                 embed_dim=64, depth=2, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.tokenizer = SpectralSpatialTokenizer(num_bands, embed_dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            SSFTTBlock(embed_dim, num_heads, mlp_ratio, dropout, patch_size)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, H, W, Bands)
        tokens, H, W = self.tokenizer(x)  # (B, H*W, embed_dim)
        
        # Prepend CLS token
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        
        for block in self.blocks:
            tokens = block(tokens)
        
        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]  # CLS token output
        return self.head(self.dropout_layer(cls_out))
    
    def extract_features(self, x):
        tokens, H, W = self.tokenizer(x)
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0]


# Quick test
_test = SSFTT(num_classes=16, num_bands=200, patch_size=15)
_dummy = torch.randn(2, 15, 15, 200)
_out = _test(_dummy)
print(f"SSFTT ready. Test params: {count_params(_test)/1e6:.3f}M, output: {_out.shape}")
del _test, _dummy, _out
