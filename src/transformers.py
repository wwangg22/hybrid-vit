"""
Different transformer architectures that mix Performer and regular attention.

TEXT TRANSFORMERS:
1. AlternatingTransformer: Switches between Performer and regular attention
2. PerformerFirstTransformer: Performer layers first, then regular
3. RegularFirstTransformer: Regular layers first, then Performer
4. CustomPatternTransformer: Custom pattern of layer types

VISION TRANSFORMERS (ViT):
1. VanillaViT: Standard ViT with all regular attention (baseline)
2. AlternatingViT: Switches between Performer and regular attention
3. PerformerFirstViT: Performer layers first, then regular
4. RegularFirstViT: Regular layers first, then Performer
5. CustomPatternViT: Custom pattern of layer types

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from performer_pytorch import SelfAttention as PerformerSelfAttention
from einops import rearrange
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    """Basic transformer feedforward network."""
    
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class PatchEmbedding(nn.Module):
    """Turns an image into patch embeddings for ViT."""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Use conv to extract patches
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, num_patches, embed_dim)
        return x


class RegularAttentionLayer(nn.Module):
    """Standard multi-head attention."""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        
        # Make Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calculate attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class PerformerAttentionLayer(nn.Module):
    """Performer (linear attention) wrapper."""
    
    def __init__(self, dim, num_heads=8, dropout=0.1, nb_features=None):
        super().__init__()
        self.attention = PerformerSelfAttention(
            dim=dim,
            heads=num_heads,
            dropout=dropout,
            nb_features=nb_features  # Random features for approximation
        )
    
    def forward(self, x, mask=None):
        return self.attention(x, mask=mask)


class TransformerBlock(nn.Module):
    """Single transformer block with attention + feedforward."""
    
    def __init__(self, dim, num_heads, ff_hidden_dim, attention_type, 
                 dropout=0.1, nb_features=None):
        super().__init__()
        self.attention_type = attention_type
        
        # Pick attention type
        if attention_type == 'performer':
            self.attn = PerformerAttentionLayer(dim, num_heads, dropout, nb_features)
        else:
            self.attn = RegularAttentionLayer(dim, num_heads, dropout)
        
        # Feedforward
        self.ff = FeedForward(dim, ff_hidden_dim, dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x, mask=None):
        # Attention with residual
        x = x + self.attn(self.norm1(x), mask)
        
        # Feedforward with residual
        x = x + self.ff(self.norm2(x))
        
        return x


class AlternatingTransformer(nn.Module):
    """
    Transformer that switches between Performer and regular attention.
    Pattern: Performer, Regular, Performer, Regular, ...
    """
    
    def __init__(self, vocab_size, max_seq_len, dim=512, num_layers=6, num_heads=8, 
                 ff_hidden_dim=2048, dropout=0.1, nb_features=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks - alternating pattern
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                attention_type='performer' if i % 2 == 0 else 'regular',
                dropout=dropout,
                nb_features=nb_features
            )
            for i in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
    
    def forward(self, x, mask=None):
        B, N = x.shape
        
        # Embeddings
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output
        x = self.norm(x)
        logits = self.head(x)
        
        return logits


class PerformerFirstTransformer(nn.Module):
    """
    Transformer with Performer layers first, then regular attention.
    Good for catching long-range stuff efficiently first.
    """
    
    def __init__(self, vocab_size, max_seq_len, dim=512, num_layers=6, 
                 num_performer_layers=3, num_heads=8, ff_hidden_dim=2048, 
                 dropout=0.1, nb_features=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_performer_layers = num_performer_layers
        
        assert num_performer_layers <= num_layers, "num_performer_layers must be <= num_layers"
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks - Performer first, then regular
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                attention_type='performer' if i < num_performer_layers else 'regular',
                dropout=dropout,
                nb_features=nb_features
            )
            for i in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
    
    def forward(self, x, mask=None):
        B, N = x.shape
        
        # Embeddings
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output
        x = self.norm(x)
        logits = self.head(x)
        
        return logits


class RegularFirstTransformer(nn.Module):
    """
    Transformer with regular attention first, then Performer.
    Good for precise local attention first, then efficient global context.
    """
    
    def __init__(self, vocab_size, max_seq_len, dim=512, num_layers=6, 
                 num_regular_layers=3, num_heads=8, ff_hidden_dim=2048, 
                 dropout=0.1, nb_features=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_regular_layers = num_regular_layers
        
        assert num_regular_layers <= num_layers, "num_regular_layers must be <= num_layers"
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks - Regular first, then Performer
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                attention_type='regular' if i < num_regular_layers else 'performer',
                dropout=dropout,
                nb_features=nb_features
            )
            for i in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
    
    def forward(self, x, mask=None):
        B, N = x.shape
        
        # Embeddings
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output
        x = self.norm(x)
        logits = self.head(x)
        
        return logits


class CustomPatternTransformer(nn.Module):
    """
    Transformer with whatever pattern you want.
    Specify pattern as a list like ['performer', 'regular', 'performer', ...].
    """
    
    def __init__(self, vocab_size, max_seq_len, dim=512, layer_pattern=None, 
                 num_heads=8, ff_hidden_dim=2048, dropout=0.1, nb_features=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        if layer_pattern is None:
            layer_pattern = ['performer', 'regular'] * 3  # Default: 6 layers alternating
        
        self.layer_pattern = layer_pattern
        num_layers = len(layer_pattern)
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks - custom pattern
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                attention_type=layer_type,
                dropout=dropout,
                nb_features=nb_features
            )
            for layer_type in layer_pattern
        ])
        
        # Output
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
    
    def forward(self, x, mask=None):
        B, N = x.shape
        
        # Embeddings
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output
        x = self.norm(x)
        logits = self.head(x)
        
        return logits


# VISION TRANSFORMERS


class VanillaViT(nn.Module):
    """
    Standard Vision Transformer with all regular attention.
    This is the baseline for comparison.
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 dim=768, num_layers=12, num_heads=12, ff_hidden_dim=3072, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.num_classes = num_classes
        
        # patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        
        # transformer blocks - all regular attention
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                attention_type='regular',
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification (use CLS token)
        x = self.norm(x)
        cls_output = x[:, 0]  # Grab CLS token
        logits = self.head(cls_output)
        
        return logits


class AlternatingViT(nn.Module):
    """
    Vision Transformer that switches between Performer and regular attention.
    Pattern: Performer, Regular, Performer, Regular, ...
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 dim=768, num_layers=12, num_heads=12, ff_hidden_dim=3072, 
                 dropout=0.1, nb_features=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks - alternating pattern
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                attention_type='performer' if i % 2 == 0 else 'regular',
                dropout=dropout,
                nb_features=nb_features
            )
            for i in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification (use CLS token)
        x = self.norm(x)
        cls_output = x[:, 0]  # Grab CLS token
        logits = self.head(cls_output)
        
        return logits


class PerformerFirstViT(nn.Module):
    """
    Vision Transformer with Performer layers first, then regular attention.
    Good for catching long-range stuff efficiently first.
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 dim=768, num_layers=12, num_performer_layers=6, num_heads=12, 
                 ff_hidden_dim=3072, dropout=0.1, nb_features=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.num_classes = num_classes
        self.num_performer_layers = num_performer_layers
        
        assert num_performer_layers <= num_layers, "num_performer_layers must be <= num_layers"
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks - Performer first, then regular
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                attention_type='performer' if i < num_performer_layers else 'regular',
                dropout=dropout,
                nb_features=nb_features
            )
            for i in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification (use CLS token)
        x = self.norm(x)
        cls_output = x[:, 0]  # Grab CLS token
        logits = self.head(cls_output)
        
        return logits


class RegularFirstViT(nn.Module):
    """
    Vision Transformer with regular attention first, then Performer.
    Good for precise local attention first, then efficient global context.
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 dim=768, num_layers=12, num_regular_layers=6, num_heads=12, 
                 ff_hidden_dim=3072, dropout=0.1, nb_features=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.num_classes = num_classes
        self.num_regular_layers = num_regular_layers
        
        assert num_regular_layers <= num_layers, "num_regular_layers must be <= num_layers"
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks - Regular first, then Performer
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                attention_type='regular' if i < num_regular_layers else 'performer',
                dropout=dropout,
                nb_features=nb_features
            )
            for i in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification (use CLS token)
        x = self.norm(x)
        cls_output = x[:, 0]  # Grab CLS token
        logits = self.head(cls_output)
        
        return logits


class CustomPatternViT(nn.Module):
    """
    Vision Transformer with whatever pattern you want.
    Specify pattern as a list like ['performer', 'regular', 'performer', ...].
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 dim=768, layer_pattern=None, num_heads=12, ff_hidden_dim=3072, 
                 dropout=0.1, nb_features=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.num_classes = num_classes
        
        if layer_pattern is None:
            layer_pattern = ['performer', 'regular'] * 6  # Default: 12 layers alternating
        
        self.layer_pattern = layer_pattern
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks - custom pattern
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                attention_type=layer_type,
                dropout=dropout,
                nb_features=nb_features
            )
            for layer_type in layer_pattern
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification (use CLS token)
        x = self.norm(x)
        cls_output = x[:, 0]  # Grab CLS token
        logits = self.head(cls_output)
        
        return logits


# Helper functions to create models
def create_alternating_transformer(**kwargs):
    """Make an AlternatingTransformer."""
    return AlternatingTransformer(**kwargs)


def create_performer_first_transformer(**kwargs):
    """Make a PerformerFirstTransformer."""
    return PerformerFirstTransformer(**kwargs)


def create_regular_first_transformer(**kwargs):
    """Make a RegularFirstTransformer."""
    return RegularFirstTransformer(**kwargs)


def create_custom_pattern_transformer(**kwargs):
    """Make a CustomPatternTransformer."""
    return CustomPatternTransformer(**kwargs)


if __name__ == "__main__":
    # Test creating models
    print("="*80)
    print("CREATING TEXT TRANSFORMERS")
    print("="*80)
    
    # Config for text transformers
    text_config = {
        'vocab_size': 50000,
        'max_seq_len': 512,
        'dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'ff_hidden_dim': 2048,
        'dropout': 0.1,
    }
    
    # 1. Alternating transformer
    model1 = AlternatingTransformer(**text_config)
    print(f"AlternatingTransformer: {sum(p.numel() for p in model1.parameters()):,} parameters")
    
    # 2. Performer-first transformer
    model2 = PerformerFirstTransformer(**text_config, num_performer_layers=3)
    print(f"PerformerFirstTransformer: {sum(p.numel() for p in model2.parameters()):,} parameters")
    
    # 3. Regular-first transformer
    model3 = RegularFirstTransformer(**text_config, num_regular_layers=3)
    print(f"RegularFirstTransformer: {sum(p.numel() for p in model3.parameters()):,} parameters")
    
    # 4. Custom pattern transformer
    custom_pattern = ['performer', 'performer', 'regular', 'regular', 'performer', 'regular']
    text_config_custom = {k: v for k, v in text_config.items() if k != 'num_layers'}
    model4 = CustomPatternTransformer(**text_config_custom, layer_pattern=custom_pattern)
    print(f"CustomPatternTransformer: {sum(p.numel() for p in model4.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    x_text = torch.randint(0, text_config['vocab_size'], (batch_size, seq_len))
    
    print(f"\nTesting with input shape: {x_text.shape}")
    with torch.no_grad():
        out1 = model1(x_text)
        print(f"AlternatingTransformer output: {out1.shape}")
        
        out2 = model2(x_text)
        print(f"PerformerFirstTransformer output: {out2.shape}")
        
        out3 = model3(x_text)
        print(f"RegularFirstTransformer output: {out3.shape}")
        
        out4 = model4(x_text)
        print(f"CustomPatternTransformer output: {out4.shape}")
    
    print("\n" + "="*80)
    print("CREATING VISION TRANSFORMERS")
    print("="*80)
    
    # Config for ViT models
    vit_config = {
        'img_size': 224,
        'patch_size': 16,
        'in_channels': 3,
        'num_classes': 1000,
        'dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'ff_hidden_dim': 3072,
        'dropout': 0.1,
    }
    
    # 1. Vanilla ViT (baseline)
    vit1 = VanillaViT(**vit_config)
    print(f"VanillaViT: {sum(p.numel() for p in vit1.parameters()):,} parameters")
    
    # 2. Alternating ViT
    vit2 = AlternatingViT(**vit_config)
    print(f"AlternatingViT: {sum(p.numel() for p in vit2.parameters()):,} parameters")
    
    # 3. Performer-first ViT
    vit3 = PerformerFirstViT(**vit_config, num_performer_layers=6)
    print(f"PerformerFirstViT: {sum(p.numel() for p in vit3.parameters()):,} parameters")
    
    # 4. Regular-first ViT
    vit4 = RegularFirstViT(**vit_config, num_regular_layers=6)
    print(f"RegularFirstViT: {sum(p.numel() for p in vit4.parameters()):,} parameters")
    
    # 5. Custom pattern ViT
    vit_custom_pattern = ['performer', 'performer', 'regular', 'regular', 
                          'performer', 'performer', 'regular', 'regular',
                          'performer', 'performer', 'regular', 'regular']
    vit_config_custom = {k: v for k, v in vit_config.items() if k != 'num_layers'}
    vit5 = CustomPatternViT(**vit_config_custom, layer_pattern=vit_custom_pattern)
    print(f"CustomPatternViT: {sum(p.numel() for p in vit5.parameters()):,} parameters")
    
    # Test forward pass
    x_img = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nTesting with image input shape: {x_img.shape}")
    with torch.no_grad():
        vit_out1 = vit1(x_img)
        print(f"VanillaViT output: {vit_out1.shape}")
        
        vit_out2 = vit2(x_img)
        print(f"AlternatingViT output: {vit_out2.shape}")
        
        vit_out3 = vit3(x_img)
        print(f"PerformerFirstViT output: {vit_out3.shape}")
        
        vit_out4 = vit4(x_img)
        print(f"RegularFirstViT output: {vit_out4.shape}")
        
        vit_out5 = vit5(x_img)
        print(f"CustomPatternViT output: {vit_out5.shape}")
    
    print("\n" + "="*80)
    print("ALL MODELS CREATED AND TESTED!")
    print("="*80)