"""
File    : pixart_model.py
Purpose : A basic implementation of the PixArt model using PyTorch.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 2, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import math
import torch
import torch.nn as nn
from .blocks_base import MultilayerPerceptron, MultiHeadSelfAttention, MultiHeadCrossAttention



def scale_and_shift(tensor, scale, shift):
    return tensor * (scale + 1) + shift

# def parametric_modulation(tensor, scale, shift):
#     return tensor * (scale + 1) + shift

#------------------------------ MODEL BLOCKS -------------------------------#

#----------------------------------------------------------------------------
class PatchEmbedder(nn.Module):
    """
    Projects a 2D latent image into a sequence of flattened image patches.
    Args:
        patch_size      (int): The size of each patch.
        input_channels  (int): Number of channels in the input latent image.
        output_channels (int): Number of channels in the projected patches.
    """
    def __init__(self,
                 patch_size     : int =    2,
                 input_channels : int =    4,
                 output_channels: int = 1152,
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels  = input_channels,
                              out_channels = output_channels,
                              kernel_size  = patch_size,
                              stride       = patch_size)

    def forward(self, x):
        assert x.shape[-1] % self.patch_size == 0 and x.shape[-2] % self.patch_size == 0, \
            f"Input latent image dimensions {tuple(x.shape[-2:])} must be divisible by patch_size ({self.patch_size})."
        # REF:
        #   - num_patches_v = lat_image_height/patch_size
        #   - num_patches_h = lat_image_width/patch_size
        #   - num_patches   = num_patches_v * num_patches_h
        # REF:
        #  x          -> [batch_size, input_channels , lat_image_height, lat_image_width]
        #  patches    -> [batch_size, output_channels, num_patches_v   , num_patches_h  ]
        #  embeddings -> [batch_size, num_patches, output_channels]
        patches    = self.proj(x)
        embeddings = patches.flatten(2).transpose(1, 2)
        return embeddings


#---------------------------------------------------------------------------#
class CaptionEmbedder(nn.Module):
    """
    Projects a T5-encoded caption into a lower-dimensional vector space.
    Args:
        input_channels  (int): Dimension of the input tensor, matching T5 encoder output.
        output_channels (int): Dimension of the output vector embeddings, controlling embedding complexity.
    """
    def __init__(self,
                 input_channels : int = 4096,
                 output_channels: int = 1152,
                 ):
        super().__init__()
        self.y_proj = MultilayerPerceptron(input_dim  = input_channels,
                                           hidden_dim = output_channels,
                                           output_dim = output_channels)

    def forward(self, caption):
        # REF:
        #  caption   -> [batch_size, 1, seq_length, input_channels ]
        #  embedding -> [batch_size, 1, seq_length, output_channels]
        embedding = self.y_proj(caption)
        return embedding


#---------------------------------------------------------------------------#
class TimestepEmbedder(nn.Module):
    def __init__(self,
                 hidden_channels    : int         = 1152,
                 positional_channels: int         = 256,
                 positional_dtype   : torch.dtype = torch.float32
                 ):
        super().__init__()
        assert (positional_channels % 2) == 0, "Positional channels must be even"
        self.positional_channels = positional_channels
        self.positional_dtype    = positional_dtype
        self.mlp = nn.Sequential(
            nn.Linear(positional_channels, hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels, bias=True))

    def forward(self, timesteps):
        # REF:
        #  timesteps            -> [batch_size]
        #  positional_encodings -> [batch_size, positional_channels]
        #  embedding            -> [batch_size,   hidden_channels  ]
        mlp_dtype            = self.mlp[0].bias.dtype
        positional_encodings = self.generate_positional_encodings(timesteps,
                                                                  self.positional_channels,
                                                                  self.positional_dtype)
        embedding = self.mlp( positional_encodings.to(mlp_dtype) )
        return embedding

    @staticmethod
    def generate_positional_encodings(timesteps: torch.Tensor,
                                      channels : int          = 256,
                                      dtype    : torch.dtype  = torch.float32
                                      ) -> torch.Tensor:
        """
        Returns a tensor containing the positional encodings for a sequence of timesteps.
        Args:
            timesteps (torch.Tensor): A tensor containing the timesteps.
            channels          (int) : The number of channels in the positional encoding.
            dtype      (torch.dtype): The data type of the positional encoding.
        """
        max_period = 10000
        half_channels = channels // 2
        frequencies   = torch.exp(-math.log(max_period) * torch.arange(half_channels, dtype=dtype, device=timesteps.device) / half_channels)
        angle_rates   = timesteps.unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat([torch.cos(angle_rates), torch.sin(angle_rates)], dim=-1)

#---------------------------------------------------------------------------#
class PixArtBlock(nn.Module):

    def __init__(self,
                 inout_dim: int   = 1152,
                 num_heads: int   =   16,
                 mlp_ratio: float =  4.0,
                 ):
        super().__init__()

        self.scale_shift_table = nn.Parameter(torch.randn(6, inout_dim) / inout_dim ** 0.5)

        self.norm1 = nn.LayerNorm(inout_dim,
                                  elementwise_affine = False,
                                  eps                = 1e-6)

        self.attn = MultiHeadSelfAttention(inout_dim,
                                           num_heads = num_heads)

        self.cross_attn = MultiHeadCrossAttention(inout_dim,
                                                  num_heads)

        self.norm2 = nn.LayerNorm(inout_dim,
                                  elementwise_affine = False,
                                  eps                = 1e-6)

        self.mlp = MultilayerPerceptron(inout_dim,
                                        hidden_dim = int(inout_dim * mlp_ratio),
                                        output_dim = inout_dim)

    def forward(self, x, time6, caption, caption_mask=None):
        # REF:
        #  t6 -> [batch_size, 6, inout_dim]
        shift_msa, scale_msa, gate_msa,  \
        shift_mlp, scale_mlp, gate_mlp = \
            (self.scale_shift_table.unsqueeze(0) + time6).chunk(6, dim=1)

        attn_input = scale_and_shift( self.norm1(x), scale_msa, shift_msa )
        x = x + gate_msa * self.attn( attn_input )
        x = x + self.cross_attn(x, caption, caption_mask)
        mlp_input = scale_and_shift( self.norm2(x), scale_mlp, shift_mlp )
        x = x + gate_mlp * self.mlp( mlp_input )
        return x


#---------------------------------------------------------------------------#
class PixArtFinalLayer(nn.Module):
    """
    The final layer of the PixArt model. This layer includes normalization,
    scaling, shifting, and a linear transformation to produce the final output.

    Args:
        input_channels (int): Number of input channels.
        patch_size     (int): Size of each patch in the input tensor,
                              the latent image is divided into patches of size
                              (patch_size x patch_size), normally 2x2.
        output_channels(int): Number of output channels.
    """
    def __init__(self,
                 input_channels : int = 1152,
                 patch_size     : int =    2,
                 output_channels: int =    8,
                 ):
        super().__init__()
        self.norm_final        = nn.LayerNorm(input_channels, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, input_channels) / input_channels ** 0.5)
        self.linear            = nn.Linear(input_channels, patch_size * patch_size * output_channels, bias=True)

    def forward(self, x, time):
        # REF:
        #  x                 -> [batch_size, seq_length, input_channels]
        #  time.unsqueeze(1) -> [batch_size,          1, input_channels]
        #  scale_params      -> [         1,          1, input_channels]
        #  shift_params      -> [         1,          1, input_channels]
        #  output            -> [batch_size, seq_length, patch_size * patch_size * output_channels]
        time                       = time.unsqueeze(1)
        shift_params, scale_params = self.scale_shift_table.unsqueeze(0).chunk(2, dim=1)
        x      = self.norm_final(x)
        x      = scale_and_shift(x, scale_params+time, shift_params+time)
        output = self.linear(x)
        return output

    # def forward_old(self, x, time):
    #     # REF:
    #     #  x                  -> [batch_size, seq_length, input_channels]
    #     #  time               -> [batch_size,          1, input_channels]
    #     #  shift_scale_params -> [         1,          2, input_channels]
    #     #  scale_plus_time    -> [batch_size,          1, input_channels]
    #     #  shift_plus_time    -> [batch_size,          1, input_channels]
    #     time                             = time.unsqueeze(1)
    #     shift_scale_params               = self.scale_shift_table.unsqueeze(0)
    #     shift_plus_time, scale_plus_time = (shift_scale_params + time).chunk(2, dim=1)
    #     x = self.norm_final(x)
    #     x = scale_and_shift(x, scale_plus_time, shift_plus_time)
    #     x = self.linear(x)
    #     return x



#============================== PIXART MODEL ===============================#
