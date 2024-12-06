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


#--------------------------------- HELPERS ---------------------------------#

# def _parametric_modulation(tensor, scale, shift):
def _scale_and_shift(tensor, scale, shift):
    """Scale and shift a tensor.
    """
    return tensor * (scale + 1) + shift


def _is_valid_mask(context_mask, context):
    """Checks if a given context mask is valid for a given context tensor.
    """
    batch_size = context.shape[0]
    seq_length = context.shape[1]
    return  isinstance(context_mask, torch.Tensor) and \
            context_mask.shape[0] == batch_size    and \
            context_mask.shape[1] == seq_length


def _generate_positional_encodings(timesteps: torch.Tensor,
                                   channels : int          = 256,
                                   dtype    : torch.dtype  = torch.float32
                                   ) -> torch.Tensor:
    """
    Returns a tensor containing the positional encodings for a sequence of timesteps.
    Args:
        timesteps (torch.Tensor): A tensor containing the timesteps.
        channels          (int) : The number of channels in the positional encoding.
        dtype      (torch.dtype): The data type of the positional encoding.
    Example:
        >>> timesteps = torch.tensor([799, 699, 599, 499, 399, 299, 199, 99])
        >>> encodings = _generate_positional_encodings(timesteps)
    """
    max_period = 10000
    half_channels = channels // 2
    frequencies   = torch.exp(-math.log(max_period) * torch.arange(half_channels, dtype=dtype, device=timesteps.device) / half_channels)
    angle_rates   = timesteps.unsqueeze(-1) * frequencies.unsqueeze(0)
    return torch.cat([torch.cos(angle_rates), torch.sin(angle_rates)], dim=-1)


def _compute_1d_sinusoidal_pos_embed(embed_dim, positions, dtype=torch.float64):
    """Compute the 1D sinusoidal positional embedding.
    """
    assert embed_dim % 2 == 0, "Embedding dimensionality must be even."
    half_embed_dim = embed_dim // 2

    frequencies = torch.arange(half_embed_dim, dtype=dtype) / half_embed_dim
    frequencies = 1.0 / (10000 ** frequencies)    # [half_embed_dim]
    positions   = positions.reshape(-1).to(dtype) # [num_positions]

    codes = torch.einsum("p,f->pf", positions, frequencies)        # [num_positions, half_embed_dim]
    embed = torch.cat([torch.sin(codes), torch.cos(codes)], dim=1) # [num_positions, embed_dim]
    return embed


#------------------------------ MODEL BLOCKS -------------------------------#
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
        output_channels (int): Dimension of the output vector embeddings.
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
    """
    Maps scalar timesteps to a high-dimensional embedding vector using an MLP and sinusoidal positional encodings.
    Args:
        output_channels          (int) : Dimension of the output vector embeddings.
        positional_channels      (int) : Dimension of the positional encoding.
        positional_dtype  (torch.dtype): Data type for positional encoding, default is float32.
    """
    def __init__(self,
                 output_channels    : int         = 1152,
                 positional_channels: int         = 256,
                 positional_dtype   : torch.dtype = torch.float32
                 ):
        super().__init__()
        assert (positional_channels % 2) == 0, "Positional channels must be even"
        self.positional_channels = positional_channels
        self.positional_dtype    = positional_dtype
        self.mlp = nn.Sequential(
            nn.Linear(positional_channels, output_channels, bias=True),
            nn.SiLU(),
            nn.Linear(output_channels, output_channels, bias=True))

    def forward(self, timesteps):
        # REF:
        #  timesteps            -> [batch_size]
        #  positional_encodings -> [batch_size, positional_channels]
        #  embedding            -> [batch_size,   output_channels  ]
        mlp_dtype            = self.mlp[0].bias.dtype
        positional_encodings = _generate_positional_encodings(timesteps, self.positional_channels, self.positional_dtype)
        embedding = self.mlp( positional_encodings.to(mlp_dtype) )
        return embedding


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

        attn_input = _scale_and_shift( self.norm1(x), scale_msa, shift_msa )
        x = x + gate_msa * self.attn( attn_input )
        x = x + self.cross_attn(x, caption, caption_mask)
        mlp_input = _scale_and_shift( self.norm2(x), scale_mlp, shift_mlp )
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
        x      = _scale_and_shift(x, scale_params+time, shift_params+time)
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

class PixArtModel(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(self,
                 latent_img_size     : int =  128, # 64 = 512x512px || 128 = 1024x1024px || 256 = 2048x2048px
                 latent_img_channels : int =    4, # number of channels in the latent image
                 internal_dim        : int = 1152, # internal dimensionality used
                 caption_dim         : int = 4096, # dimensionality of the caption input (T5 encoded prompt)
                 patch_size          : int =    2, # size of each patch (in latent blocks)
                 num_heads           : int =   16, # number of attention heads in the transformer
                 depth               : int =   28, # number of layers in the transformer
                 mlp_ratio           : int =  4.0, # ratio of the hidden dimension to the mlp dimension
                 device              : str | torch.device = "cpu",
                 **kwargs,
                 ):
        super().__init__()
        assert latent_img_size in [64, 128, 256], "only support 512px, 1024px, and 2048px models"
        assert internal_dim % num_heads == 0    , "internal dimension must be divisible by the number of heads"

        self.base_size        = latent_img_size // patch_size
        self.pe_interpolation = latent_img_size / 64.          # Positional Encoding interpolation
        self.in_channels      = latent_img_channels
        self.output_dim       = latent_img_channels * 2
        self.internal_dim     = internal_dim
        self.patch_size       = patch_size
        self.num_heads        = num_heads
        self.depth            = depth
        self.pos_embeddings_cache_key = None
        self.pos_embeddings           = None

        if isinstance(device, str):
            device = torch.device(device)

        with torch.device(device):

            #-- embedder blocks --#

            self.x_embedder = PatchEmbedder(patch_size,
                                            latent_img_channels,
                                            internal_dim)

            self.y_embedder = CaptionEmbedder(caption_dim,
                                              internal_dim)

            self.t_embedder = TimestepEmbedder(internal_dim,
                                               positional_channels = 256,
                                               positional_dtype    = torch.float32)

            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(internal_dim, 6 * internal_dim))

            #-- transformer blocks --#

            self.blocks = nn.ModuleList([
                PixArtBlock(
                    internal_dim,
                    num_heads,
                    mlp_ratio = mlp_ratio
                ) for _ in range(depth)
            ])

            self.final_layer = PixArtFinalLayer(internal_dim,
                                                patch_size,
                                                self.output_dim
                                                )


    def forward(self,
                x           : torch.Tensor,         # input images (in latent space).
                timestep    : torch.Tensor,         # time steps in the diffusion process.
                caption     : torch.Tensor = None,  # conditional info (from prompt).
                caption_mask: torch.Tensor = None,  # optional attention mask applied to conditional.
                return_eps_only: bool      = False, # if True then return only the noise prediction (eps), otherwise return the full output
                **kwargs):
        # REF:
        #  x            -> [batch_size, latent_channels, H, W]
        #  timestep     -> [batch_size]
        #  caption      -> [batch_size, caption_len, caption_dim]
        #  caption_mask -> [batch_size, caption_len]

        # `caption`` puede ser suministrado tambien como parametro `context` (como sucede en ComfyUI)
        # y puede tener cualquiera de estos dos shapes:
        #    [batch_size, caption_len, caption_dim]
        #    [batch_size, 1, caption_len, caption_dim]
        if caption is None:
            caption = kwargs["context"] if "context" in kwargs else torch.zeros(1, 0, 4096, device=self.device)
        if len(caption.shape) == 3:
            caption = caption.unsqueeze(1)  # [batch_size, 1, caption_len, caption_dim]

        dtype = self.dtype
        x            = x.to(dtype)
        timestep     = timestep.to(dtype)
        caption      = caption.to(dtype)
        caption_mask = caption_mask.to(dtype) if caption_mask is not None else None

        assert not self.training, \
            "This PixArtSigma class can only be used for inference, no ha sido probrada en training."
        assert caption_mask is None or _is_valid_mask(caption_mask, caption), \
            "la caption_mask suministrada en forward(..) tiene un formato inadecuado"

        batch_size, latent_height, latent_width = x.shape[0], x.shape[-2], x.shape[-1]
        height = latent_height // self.patch_size
        width  = latent_width  // self.patch_size


        tstep, tstep6 = self.get_cached_time_embeddings(timestep)
        pos           = self.cached_position_embeddings(height, width, x.device, x.dtype)
        x             = self.x_embedder(x) + pos           # [batch_size, patches_count, internal_dim]
        caption       = self.y_embedder(caption)           # [batch_size, 1, seq_length, internal_dim]


        for block in self.blocks:
            x = block(x, tstep6, caption, caption_mask)    # [batch_size, patches_count, internal_dim]
        x = self.final_layer(x, tstep)                     # [batch_size, patches_count, (patch_size^2)*(latent_channels*2)]

        # unpatchify
        # [batch_size, out_channels, lat_height, lat_width]
        #                <- [batch_size, patches, (patch_size^2)*output_dim]
        assert x.shape[1] == height * width
        x = x.view(batch_size, height, width, self.patch_size, self.patch_size, self.output_dim)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(batch_size, self.output_dim, latent_height, latent_width)

        if return_eps_only:
            x = x.chunk(2, dim=1)[0]

        return x

    def get_cached_time_embeddings(self, timesteps):
        #  .timesteps : [batch_size]
        #  .t         : [batch_size, internal_dim]
        #  .t6        : [batch_size, 6, internal_dim]
        batch_size = timesteps.shape[0]
        t  = self.t_embedder(timesteps)
        t6 = self.t_block(t).reshape(batch_size, 6, self.internal_dim)
        return t, t6

    def cached_position_embeddings(self,
                                   height: int,
                                   width : int,
                                   device: torch.device,
                                   dtype : torch.dtype
                                   ):
        """
        Computes and caches 2D sinusoidal position embeddings.

        This method maintains a small cache of position embeddings and calculates them
        only when the input dimensions (height, width), device, or data type change.

        Args:
            height (int)         : Height of the input feature map.
            width  (int)         : Width of the input feature map.
            device (torch.device): Device where the position embeddings will be stored.
            dtype  (torch.dtype) : Data type of the position embeddings.

        Returns:
            torch.Tensor: A tensor of size (1, internal_dim, height, width) containing the
                sinusoidal position embeddings.
        """
        cache_key = (height, width, str(device), str(dtype))

        if cache_key == self.pos_embeddings_cache_key:
            return self.pos_embeddings

        embedding_dim = self.internal_dim
        assert (embedding_dim % 4) == 0, "Embedding dimensionality must be multiplo de 4."

        grid_h = torch.arange(height, dtype=torch.float32) / (height / self.base_size) / self.pe_interpolation
        grid_w = torch.arange( width, dtype=torch.float32) / ( width / self.base_size) / self.pe_interpolation
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="xy")  # [height,width], [height,width]
        grid = torch.stack([grid_w, grid_h], dim=0)                     # [2, height, width]
        grid = grid.unsqueeze(1)                                        # [2, 1, height, width]

        emb_h = _compute_1d_sinusoidal_pos_embed(embedding_dim // 2, grid[0])  # [height*width, embedding_dim/2]
        emb_w = _compute_1d_sinusoidal_pos_embed(embedding_dim // 2, grid[1])  # [height*width, embedding_dim/2]

        pos_embeddings = torch.cat([emb_h, emb_w], dim=1) # [height*width, embedding_dim]
        pos_embeddings = pos_embeddings.unsqueeze(0).to(device, dtype=dtype)

        self.pos_embeddings_cache_key = cache_key
        self.pos_embeddings           = pos_embeddings
        return self.pos_embeddings


    @property
    def dtype(self):
        """Returns the data type of the model parameters."""
        return self.x_embedder.proj.weight.dtype


    @property
    def device(self):
        """Returns the device on which the model parameters are located."""
        return self.x_embedder.proj.weight.device


    def freeze(self) -> None:
        """Freeze all parameters of the model to prevent them from being updated during inference."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()


    def unfreeze(self) -> None:
        """Unfreeze all parameters of the model to allow them to be updated during training."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()
