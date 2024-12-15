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
import torch
import torch.nn as nn
import torch.nn.functional as F

#--------------------------------- HELPERS ---------------------------------#

def _scale_and_shift(tensor, scale, shift):
    """Scale and shift a tensor."""
    return tensor * (scale + 1) + shift


def _is_valid_mask(caption_mask, caption):
    """Checks if a given context mask is valid for a given caption tensor."""
    batch_size = caption.shape[0]
    seq_length = caption.shape[1]
    return  isinstance(caption_mask, torch.Tensor) and \
            caption_mask.shape[0] == batch_size    and \
            caption_mask.shape[1] == seq_length


def _generate_positional_encodings(positions: torch.Tensor,
                                   channels : int         = 576,
                                   sincos   : bool        = True,
                                   dtype    : torch.dtype = torch.float64
                                   ) -> torch.Tensor:
    """
    Generates positional encodings.

    This function creates sinusoidal positional embeddings based on the provided positions.
    The embeddings are created using sine and cosine functions, and they can be optionally
    returned in either a (sin, cos) or (cos, sin) order.

    Args:
        positions  (Tensor): A tensor containing the positions to be encoded.
                             It can be a 1D tensor of shape [num_positions], or a
                             higher-dimensional tensor, in which case it will be
                             flattened to [num_positions].
        channels      (int): The dimensionality of the positional encodings.
                             (must be an even number)
        sincos       (bool): If True, returns embeddings in (sin, cos) order;
                             if False, returns embeddings in (cos, sin) order.
        dtype (torch.dtype): The data type of the returned embeddings.

    Returns:
        A tensor containing the positional encodings.
        The shape of the tensor is [num_positions, channels].
    """
    assert channels % 2 == 0, "Embedding dimensionality must be even."
    MAX_PERIOD    = 10000
    half_channels = channels // 2
    frequencies   = torch.arange(half_channels, dtype=dtype, device=positions.device) / half_channels

    #frequencies = torch.exp(-math.log(MAX_PERIOD) * frequencies)                   # [half_channels]
    frequencies  = 1.0 / (MAX_PERIOD ** frequencies)                                # [half_channels]

    positions   = positions.reshape(-1).to(dtype)                                   # [num_positions]
    angle_rates = torch.einsum("p,f->pf", positions, frequencies)                   # [num_positions, half_channels]
    if sincos:
        embed = torch.cat([torch.sin(angle_rates), torch.cos(angle_rates)], dim=-1) # [num_positions, channels]
    else:
        embed = torch.cat([torch.cos(angle_rates), torch.sin(angle_rates)], dim=-1) # [num_positions, channels]
    return embed


class _MultilayerPerceptron(nn.Module):
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


class _MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module.
    Args:
        dim       (int): Number of input and output dimensions.
        num_heads (int): Number of attention heads.
    """
    def __init__(self,
                 dim      : int,
                 num_heads: int,
                 ):
        super().__init__()
        assert dim % num_heads == 0, "Self-Attention dim should be divisible by num_heads"
        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.qkv       = nn.Linear(dim, dim * 3)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        QKV_DIM = 3
        batch_size, x_length, dim = x.shape
        assert dim == self.dim, f"Self-Attention input dimension ({dim}) must match layer dimension ({self.dim})"

        # linear projection
        qkv = self.qkv(x)  # [batch_size, x_length, 3*dim]

        # reshape to [batch_size, num_heads, length, head_dim]
        qkv = qkv.reshape(batch_size, x_length, QKV_DIM, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(batch_size, x_length, dim)
        return self.proj(x)


class _MultiHeadCrossAttention(nn.Module):
    """
    Milti-Head Cross-Attention module.
    Args:
        dim       (int): Number of input and output dimensions.
        num_heads (int): Number of attention heads.
    """
    def __init__(self,
                 dim      : int = 1152,
                 num_heads: int = 16,
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
        # REF:
        #  x              -> [batch_size,    x_length, dim]
        #  cond           -> [batch_size, cond_length, dim]
        #  cond_attn_mask -> [batch_size, cond_length]
        KV_DIM = 2
        cond_length = cond.shape[1]
        batch_size, x_length, dim = x.shape
        assert dim == self.dim, f"Cross-Attention input dimension ({dim}) must match layer dimension ({self.dim})"

        # generate mask
        if cond_attn_mask is not None:
            cond_attn_mask = cond_attn_mask.unsqueeze(1)        # [batch_size, 1, cond_length]
            cond_attn_mask = cond_attn_mask.to(q.device).bool()

        # linear projection
        q  = self.q_linear(x)      # [batch_size,    x_length,   dim]
        kv = self.kv_linear(cond)  # [batch_size, cond_length, 2*dim]

        # reshape to [batch_size, num_heads, length, head_dim]
        q    =  q.view(batch_size,        x_length    , self.num_heads, self.head_dim).transpose(1, 2)
        kv   = kv.view(batch_size, cond_length, KV_DIM, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=cond_attn_mask)
        x = x.transpose(1, 2).reshape(batch_size, x_length, dim)
        return self.proj(x)


#---------------------------------------------------------------------------#
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
        # {num_patches_v} = lat_image_height/patch_size
        # {num_patches_h} = lat_image_width/patch_size
        # {num_patches}   = num_patches_v * num_patches_h
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
        self.y_proj = _MultilayerPerceptron(input_dim  = input_channels,
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
        positional_encodings = _generate_positional_encodings(timesteps, self.positional_channels, sincos=False, dtype=self.positional_dtype)
        embedding = self.mlp( positional_encodings.to(mlp_dtype) )
        return embedding


#---------------------------------------------------------------------------#
class PixArtBlock(nn.Module):
    """
    The PixArt transformer block that integrates visual, textual and
    temporal (timesteps) information.

    Args:
        inout_dim  (int) : Dimension of the input and output tensors.
        num_heads  (int) : Number of attention heads in the transformer encoder.
        mlp_ratio (float): Ratio of the hidden dimension to the input dimension in the MLP layer.
    """
    def __init__(self,
                 inout_dim: int   = 1152,
                 num_heads: int   =   16,
                 mlp_ratio: float =  4.0,
                 ):
        super().__init__()

        self.scale_shift_table = nn.Parameter(torch.randn(6, inout_dim) / inout_dim ** 0.5)

        self.norm1             = nn.LayerNorm(inout_dim,
                                              elementwise_affine = False,
                                              eps                = 1e-6)

        self.attn              = _MultiHeadSelfAttention(inout_dim,
                                                         num_heads = num_heads)

        self.cross_attn        = _MultiHeadCrossAttention(inout_dim,
                                                          num_heads)

        self.norm2             = nn.LayerNorm(inout_dim,
                                              elementwise_affine = False,
                                              eps                = 1e-6)

        self.mlp               = _MultilayerPerceptron(inout_dim,
                                                       hidden_dim = int(inout_dim * mlp_ratio),
                                                       output_dim = inout_dim)

    def forward(self, x, time6, caption: torch.Tensor, caption_mask: torch.Tensor = None):
        # REF:
        #  x             -> [batch_size, num_patches, inout_dim]
        #  time6         -> [batch_size,           6, inout_dim]
        #  caption       -> [batch_size,  seq_length, inout_dim]
        #  caption_mask  -> [batch_size,  seq_length]
        #  scale_*       -> [batch_size,           1, inout_dim]
        #  shift_*       -> [batch_size,           1, inout_dim]
        #  return        -> [batch_size, num_patches, inout_dim]
        shift_attn, scale_attn, gate_attn,  \
        shift_mlp , scale_mlp , gate_mlp  = \
            (self.scale_shift_table.unsqueeze(0) + time6).chunk(6, dim=1)

        residual = x
        x = _scale_and_shift(self.norm1(x), scale_attn, shift_attn)
        x = residual + gate_attn * self.attn( x )

        residual = x
        x = residual + self.cross_attn(x, caption, caption_mask)

        residual = x
        x = _scale_and_shift(self.norm2(x), scale_mlp, shift_mlp)
        x = residual + gate_mlp * self.mlp(x)

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

    def forward(self, x, time: torch.Tensor):
        # REF:
        #  x         -> [batch_size, seq_length, input_channels]
        #  time1     -> [batch_size,          1, input_channels]
        #  scale     -> [         1,          1, input_channels]
        #  shift     -> [         1,          1, input_channels]
        #  return    -> [batch_size, seq_length, patch_size * patch_size * output_channels]
        time1        = time.unsqueeze(1)
        shift, scale = (self.scale_shift_table.unsqueeze(0) + time1).chunk(2, dim=1)

        x = self.norm_final(x)
        x = _scale_and_shift(x, scale, shift)
        x = self.linear(x)
        return x


#===========================================================================#
#////////////////////////////// PIXART MODEL ///////////////////////////////#
#===========================================================================#

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
        self.out_channels     = latent_img_channels * 2
        self.internal_dim     = internal_dim
        self.patch_size       = patch_size
        self.num_heads        = num_heads
        self.depth            = depth
        self.pos_embeddings_cache_key = None
        self.pos_embeddings           = None

        if isinstance(device, str):
            device = torch.device(device)

        with torch.device(device):

            #- Embedder blocks ----------------------------

            self.x_embedder = PatchEmbedder(patch_size,
                                            latent_img_channels,
                                            internal_dim)

            self.y_embedder = CaptionEmbedder(caption_dim,
                                              internal_dim)

            self.t_embedder = TimestepEmbedder(internal_dim,
                                               positional_channels = 256,
                                               positional_dtype    = torch.float32)

            self.t_block    = nn.Sequential(nn.SiLU(),
                                            nn.Linear(internal_dim, 6 * internal_dim))

            #- Transformer blocks -------------------------

            self.blocks = nn.ModuleList([
                PixArtBlock(
                    internal_dim,
                    num_heads,
                    mlp_ratio = mlp_ratio
                ) for _ in range(depth)
            ])

            self.final_layer = PixArtFinalLayer(internal_dim,
                                                patch_size,
                                                self.out_channels)

            #----------------------------------------------


    def forward(self,
                x           : torch.Tensor,         # input latent images.
                timestep    : torch.Tensor,         # time-steps in the diffusion process.
                caption     : torch.Tensor = None,  # conditional caption info (from prompt).
                caption_mask: torch.Tensor = None,  # optional attention mask for the caption.
                return_epsilon: bool       = False, # if True, returns only the noise prediction (epsilon).
                                                    # otherwise return (eps + variance) as the original implementation.
                **kwargs
                ):

        # handle the case where caption is provided as parameter 'context' (e.g. in ComfyUI)
        if caption is None:
            caption = kwargs["context"] if "context" in kwargs else torch.zeros(1, 0, 4096, device=self.device)

        assert not self.training, \
            "This PixArtModel class can only be used for inference and should not be used during training."
        assert caption_mask is None or _is_valid_mask(caption_mask, caption), \
            "The provided `caption_mask` does not have the correct format for this forward method."

        # calculate some constants that will be used later
        batch_size, latent_height, latent_width = x.shape[0], x.shape[-2], x.shape[-1]
        height = latent_height // self.patch_size
        width  = latent_width  // self.patch_size

        # generate the embeddings
        tstep, tstep6 = self._cached_time_embeddings(timestep)
        pos           = self._cached_pos_embeddings(height, width, x.device, x.dtype)
        x             = self.x_embedder(x) + pos               # [batch_size, patches_count, internal_dim]
        caption       = self.y_embedder(caption)               # [batch_size,    seq_length, internal_dim]

        # apply transformer blocks
        for block in self.blocks:
            x = block(x, tstep6, caption, caption_mask)        # [batch_size, patches_count, internal_dim]
        x = self.final_layer(x, tstep)                         # [batch_size, patches_count, (patch_size^2)*(latent_channels*2)]

        assert x.shape[1] == height * width, \
            "The number of patches doesn't match the expected value"

        # unpatchify
        x = x.view(batch_size, height, width,                  # [batch_size, height, width, patch_size, patch_size, (latent_channels*2)]
                   self.patch_size, self.patch_size,
                   self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)                        # [batch_size, (latent_channels*2), patch_size, width, patch_size]
        x = x.reshape(batch_size, self.out_channels,           # [batch_size, (latent_channels*2), latent_height, latent_width]
                      latent_height, latent_width)

        # if only epsilon is required,
        # remove the second half of the channels (variance?)
        if return_epsilon:
            return x.chunk(2, dim=1)[0]                        # > [batch_size, latent_channels, latent_height, latent_width]
        else:
            return x                                           # > [batch_size, (latent_channels*2), latent_height, latent_width]


    def _cached_time_embeddings(self,
                                timesteps: torch.Tensor
                                ) -> torch.Tensor:
        # REF:
        #  timesteps -> [batch_size]
        #  t         -> [batch_size, internal_dim]
        #  t6        -> [batch_size, 6, internal_dim]
        batch_size = timesteps.shape[0]
        t  = self.t_embedder(timesteps)
        t6 = self.t_block(t).reshape(batch_size, 6, self.internal_dim)
        return t, t6

    def _cached_pos_embeddings(self,
                               height: int,
                               width : int,
                               device: torch.device,
                               dtype : torch.dtype
                               ) -> torch.Tensor:
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

        emb_h = _generate_positional_encodings(grid[0], embedding_dim // 2, sincos=True, dtype=torch.float64)  # [height*width, embedding_dim/2]
        emb_w = _generate_positional_encodings(grid[1], embedding_dim // 2, sincos=True, dtype=torch.float64)  # [height*width, embedding_dim/2]

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
