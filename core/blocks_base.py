"""
File    : blocks_base.py
Purpose : Define basic blocks for PixArt without using xformers.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 3, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


#----------------------------------------------------------------------------
class MultilayerPerceptron(nn.Module):
    """
    Simple Multilayer Perceptron (MLP) module.

    Args:
        input_dim  (int): Number of input dimensions.
        hidden_dim (int): Number of dimensions in the hidden layer.
        output_dim (int): Number of output dimensions.
        gelu_approximation (str, optional): Approximation method for GELU activation.
            Options are "tanh" or "none". Defaults to "tanh".
    """

    def __init__(self,
                 input_dim         : int,
                 hidden_dim        : int,
                 output_dim        : int,
                 gelu_approximation: str = "tanh"
                 ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU(approximate=gelu_approximation)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#----------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module.

    Args:
        dim       (int): Number of input and output dimensions.
        num_heads (int): Number of attention heads.
        use_fp32 (Optional[bool]): Whether to use float32 precision for attention computation.
    """

    def __init__(self,
                 dim      : int,
                 num_heads: int,
                 use_fp32 : Optional[bool] = False
                 ):
        super().__init__()
        assert dim % num_heads == 0, "Self-Attention dim should be divisible by num_heads"
        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.use_fp32  = use_fp32
        self.qkv       = nn.Linear(dim, dim * 3)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, dim = x.shape
        assert dim == self.dim, f"Self-Attention input dimension ({dim}) must match layer dimension ({self.dim})"

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.use_fp32:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float()
                ).to(q.dtype)
        else:
            attn_output = F.scaled_dot_product_attention(q, k, v)

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, dim)
        return self.proj(attn_output)


#----------------------------------------------------------------------------
class MultiHeadCrossAttention(nn.Module):
    """
    Milti-Head Cross-Attention module.
    It allows one sequence (the query) to attend to another sequence (the key-values),
    facilitating information flow between different sources.

    Args:
        dim       (int): Number of input and output dimensions.
        num_heads (int): Number of attention heads.
    """

    def __init__(self,
                 dim      : int,
                 num_heads: int,
                 ):

        super().__init__()
        assert dim % num_heads == 0, "Cross-Attention dim should be divisible by num_heads"
        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.q_linear  = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, x, cond, cond_attn_mask=None):
        batch_size, seq_length, dim = x.shape
        assert dim == self.dim, f"Cross-Attention input dimension ({dim}) must match layer dimension ({self.dim})"

        # linear projections
        q  = self.q_linear(x)      # (batch_size,  seq_length, d_model)
        kv = self.kv_linear(cond)  # (batch_size, cond_length, d_model * 2)

        # reshape to (batch_size, num_heads, seq_length, head_dim)
        q    = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        kv   = kv.view(batch_size, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # k, v of shape (batch_size, num_heads, cond_length, head_dim)

        # generate mask
        if cond_attn_mask is not None:
            cond_attn_mask = cond_attn_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, cond_length)
            cond_attn_mask = cond_attn_mask.to(q.device).bool()

        # scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=cond_attn_mask)

        # reshape and concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, dim)

        # final linear projection
        return self.proj(attn_output)


