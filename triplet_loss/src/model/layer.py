# Standard library imports

# Third-party imports
import torch
import torch.nn as nn
from torch.nn import Linear

# Custom imports

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super(SelfAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.W_Q = Linear(hidden_dim, hidden_dim)
        self.W_K = Linear(hidden_dim, hidden_dim)
        self.W_V = Linear(hidden_dim, hidden_dim)

        self.scale = self.head_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)

        # Reshape and permute for multi-head attention
        Q = self.W_Q(x).view(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        K = self.W_K(x).view(N, self.num_heads, self.head_dim).permute(1, 0, 2)
        V = self.W_V(x).view(N, self.num_heads, self.head_dim).permute(1, 0, 2)

        # Compute attention scores and weights
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply attention weights to values and reshape
        out = torch.matmul(attn_weights, V).permute(1, 0, 2).contiguous().view(N, -1)

        return out
