"""
  File    : blocks_pixart.py
  Brief   : bloques propios de la arquitectura PixArt
  Author  : Martin Rizzo | <martinrizzo@gmail.com>
  Date    : May 9, 2024
  Repo    : https://github.com/martin-rizzo/ComfyUI-PixArt
  License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      PixArt Node Collection for ComfyUI
             Nodes providing support for PixArt models in ComfyUI

     Copyright (c) 2024 Martin Rizzo

     Permission is hereby granted, free of charge, to any person obtaining
     a copy of this software and associated documentation files (the
     "Software"), to deal in the Software without restriction, including
     without limitation the rights to use, copy, modify, merge, publish,
     distribute, sublicense, and/or sell copies of the Software, and to
     permit persons to whom the Software is furnished to do so, subject to
     the following conditions:

     The above copyright notice and this permission notice shall be
     included in all copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
     TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import torch.nn as nn
import math

from .blocks_base import    \
    MultilayerPerceptron,   \
    MultiHeadSelfAttention, \
    MultiHeadCrossAttention

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


############################### PIXART BLOCKS ###############################

#----------------------------------------------------------------------------
class PatchEmbedder(nn.Module):
    # PixArtSigma.x_embedder
    # Embeds 2D image into vector representation

    def __init__(self, patch_size: int, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels  = input_dim,
                              out_channels = output_dim,
                              kernel_size  = patch_size,
                              stride       = patch_size
                              )

    def forward(self, x):
        # REF:
        #   - h = (latent_h/patch_size)
        #   - w = (latent_w/patch_size)
        #   - seq_length = h * w
        #
        # [batch_size, output_dim, h, w] <- [batch_size, input_dim, latent_h, latent_w]
        embedding = self.proj(x)
        # [batch_size, seq_length, output_dim] <- [batch_size, output_dim, h, w]
        embedding = embedding.flatten(2).transpose(1, 2)
        return embedding

#----------------------------------------------------------------------------
class CaptionEmbedder(nn.Module):
    # PixArtSigma.y_embedder
    # Embeds captions into vector representation

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.y_proj = MultilayerPerceptron(input_dim  = input_dim,
                                           hidden_dim = output_dim,
                                           output_dim = output_dim
                                           )

    def forward(self, caption):
        embedding = self.y_proj(caption)
        return embedding

#----------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    # PixArtSigma.t_embedder
    # Embeds scalar timesteps into vector representations.

    def __init__(self, hidden_dim, positional_dim=256, positional_dtype=torch.float32):
        super().__init__()
        assert (positional_dim % 2) == 0
        self.positional_dim   = positional_dim
        self.positional_dtype = positional_dtype
        self.mlp = nn.Sequential(
            nn.Linear(positional_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
            )

    def forward(self, timesteps):
        max_period = 10000
        pos_dtype  = self.positional_dtype
        mlp_dtype  = self.mlp[0].bias.dtype
        # generate positional encoding using sinusoidal functions
        half_dim    = self.positional_dim // 2
        frequencies = torch.exp(-math.log(max_period) * torch.arange(half_dim, dtype=pos_dtype, device=timesteps.device) / half_dim)
        angle_rates = timesteps[:, None].to(pos_dtype) * frequencies[None]
        positional_encoding = torch.cat([torch.cos(angle_rates), torch.sin(angle_rates)], dim=-1)
        # pass the positional encoding through the MLP to obtain the timestep embedding
        embedding = self.mlp( positional_encoding.to(mlp_dtype) )
        return embedding

#----------------------------------------------------------------------------
class PixArtMSBlock(nn.Module):
    # PixArtSigma.blocks[]

    def __init__(self,
                 inout_dim,
                 num_heads,
                 mlp_ratio  = 4.0,
                 input_size = None,
                 sampling   = None,
                 sr_ratio   = 1,
                 qk_norm    = False
                 ):
        super().__init__()
        assert input_size is None
        assert sampling   is None
        assert sr_ratio == 1
        assert qk_norm  == False

        self.scale_shift_table = nn.Parameter(torch.randn(6, inout_dim) / inout_dim ** 0.5)

        self.norm1 = nn.LayerNorm(inout_dim,
                                  elementwise_affine = False,
                                  eps                = 1e-6
                                  )
        self.attn = MultiHeadSelfAttention(inout_dim,
                                           num_heads = num_heads,
                                           )
        self.cross_attn = MultiHeadCrossAttention(inout_dim,
                                                  num_heads
                                                  )
        self.norm2 = nn.LayerNorm(inout_dim,
                                  elementwise_affine = False,
                                  eps                = 1e-6
                                  )
        self.mlp = MultilayerPerceptron(inout_dim,
                                        hidden_dim = int(inout_dim * mlp_ratio),
                                        output_dim = inout_dim
                                        )

    def forward(self, x, t6, y, y_attn_mask=None, **kwargs):
        # t6 = [batch_size, 6, inout_dim]
        shift_msa, scale_msa, gate_msa,  \
        shift_mlp, scale_mlp, gate_mlp = \
            (self.scale_shift_table.unsqueeze(0) + t6).chunk(6, dim=1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.cross_attn(x, y, y_attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

#----------------------------------------------------------------------------
class PixArtFinalLayer(nn.Module):
    # PixArtSigma.final_layer
    # The final layer with normalization, scaling, and linear transformation.

    def __init__(self, input_dim, patch_size, output_dim):
        super().__init__()
        self.norm_final        = nn.LayerNorm(input_dim, elementwise_affine=False, eps=1e-6)
        self.linear            = nn.Linear(input_dim, patch_size * patch_size * output_dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, input_dim) / input_dim ** 0.5)

    def forward(self, x, t1):
        # x = [batch_size, seq_length, input_dim]
        # t = [batch_size, input_dim]
        shift, scale = (self.scale_shift_table.unsqueeze(0) + t1).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
