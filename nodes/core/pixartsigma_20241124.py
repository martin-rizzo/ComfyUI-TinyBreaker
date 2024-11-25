"""
File    : pixartsigma_20241124.py
Purpose : Old implementation of the PixArt model using PyTorch.
          This file is a snapshot at 2024/11/24 for my reference in future modifications.
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
from os          import PathLike
from typing      import Dict, Union
from safetensors import safe_open

from .blocks_pixart import \
    PatchEmbedder,         \
    CaptionEmbedder,       \
    TimestepEmbedder,      \
    PixArtMSBlock,         \
    PixArtFinalLayer

# Tabla de conversion para las keys utilizadas en los archivos .safetensors
DEPTH_TAG = "|depth|"
WB_TAG    = "|w,b|"
KV_TAG    = "|k+v|"
QKV_TAG   = "|q+k+v|"
STATE_DICT_TABLE_TEMPLATE = [
        #       PixArt Reference Code keys          |               HF Diffusers keys                    #
        #------------------------------------------------------------------------------------------------#
        # Patch embeddings                          |                                                    #
        ("x_embedder.proj.|w,b|"                    , "pos_embed.proj.|w,b|"                             ),
        # Caption projection                        |
        ("y_embedder.y_embedding"                   , "caption_projection.y_embedding"                   ),
        ("y_embedder.y_proj.fc1.|w,b|"              , "caption_projection.linear_1.|w,b|"                ),
        ("y_embedder.y_proj.fc2.|w,b|"              , "caption_projection.linear_2.|w,b|"                ),
        # AdaLN-single LN                           |
        ("t_embedder.mlp.0.|w,b|"                   , "adaln_single.emb.timestep_embedder.linear_1.|w,b|"),
        ("t_embedder.mlp.2.|w,b|"                   , "adaln_single.emb.timestep_embedder.linear_2.|w,b|"),
        # Shared norm                               |
        ("t_block.1.|w,b|"                          , "adaln_single.linear.|w,b|"                        ),
        # Final block                               |
        ("final_layer.linear.|w,b|"                 , "proj_out.|w,b|"                                   ),
        ("final_layer.scale_shift_table"            , "scale_shift_table"                                ),
        #--------------------------------- TRANSFORMER BLOCKS -------------------------------------------#
        ("blocks.|depth|.scale_shift_table"         , "transformer_blocks.|depth|.scale_shift_table"     ),
        # Projection                                |
        ("blocks.|depth|.attn.proj.|w,b|"           , "transformer_blocks.|depth|.attn1.to_out.0.|w,b|"  ),
        # Feed-forward                              |
        ("blocks.|depth|.mlp.fc1.|w,b|"             , "transformer_blocks.|depth|.ff.net.0.proj.|w,b|"   ),
        ("blocks.|depth|.mlp.fc2.|w,b|"             , "transformer_blocks.|depth|.ff.net.2.|w,b|"        ),
        # Cross-attention (proj)                    |
        ("blocks.|depth|.cross_attn.proj.|w,b|"     , "transformer_blocks.|depth|.attn2.to_out.0.|w,b|"  ),
        # Cross-attention                           |
        ("blocks.|depth|.cross_attn.q_linear.|w,b|" , "transformer_blocks.|depth|.attn2.to_q.|w,b|"      ),
        ("blocks.|depth|.cross_attn.kv_linear.|w,b|", "transformer_blocks.|depth|.attn2.to_|k+v|.|w,b|"  ),
        # Self-attention                            |
        ("blocks.|depth|.attn.qkv.|w,b|"            , "transformer_blocks.|depth|.attn1.to_|q+k+v|.|w,b|"),
]


#================================= Helpers =================================#

def _is_valid_mask(context_mask, context):
    """Checks if a given context mask is valid for a given context tensor.
    """
    batch_size = context.shape[0]
    seq_length = context.shape[1]
    return  isinstance(context_mask, torch.Tensor) and \
            context_mask.shape[0] == batch_size    and \
            context_mask.shape[1] == seq_length


def _find_tensor(state_dict: Dict, template_key: str, subkey: str = None):
    if subkey is None:
        return state_dict.get(template_key)
    else:
        parts = template_key.split("|")
        return state_dict.get(parts[0] + subkey + parts[2])


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


#===========================================================================#
# multi-scale sigma architecture
# sigma-512, sigma-1024 & sigma-2K
class PixArtSigma(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    # tablas inicializadas en el metodo `_init_state_dict_table()`
    # (sirver para convertir `state_dict` desde y hacia el formato de diffusers)
    STATE_DICT_TABLE = None
    DIFFUSERS_KEYS   = None
    PIXART_KEYS      = None

    def __init__(
            self,
            input_size       ,#=   32, # latent_size = image_size // 8
            pe_interpolation ,#=    1, # 512px =   1, 1024px =   1, 2048px = 3
            context_len      ,#=  120, # alpha = 120, sigma  = 300
            input_dim        ,#=    4, # canales en la imagen latente
            hidden_dim       ,#= 1152, # dimensionalidad usada internamente
            context_dim      ,#= 4096, # dimensionalidad del condicionante (prompt)
            patch_size       ,#=    2,
            num_heads        ,#=   16,
            depth            ,#=   28,
            mlp_ratio        =  4.0,
            device: Union[str, torch.device] = "cpu",
            #---- unsupported -----#
            drop_path : float  = 0,
            pred_sigma: bool   = True,
            qk_norm   : bool   = False,
            kv_compress_config = None,
            **kwargs,
    ):
        super().__init__()
        assert pe_interpolation in [1, 2, 3], "unicamente soporta 512px, 1024px y 2048px"
        assert drop_path == 0, "el argumento drop_path no está soportado"
        assert pred_sigma == True, "el argumento pred_sigma no está soportado"
        assert qk_norm   == False, "el argument qk_norm no está soportado"
        assert kv_compress_config == None, "el argumento kv_compress_config no está soportado"

        self.in_channels      = input_dim
        self.output_dim       = input_dim * 2
        self.patch_size       = patch_size
        self.num_heads        = num_heads
        self.pe_interpolation = pe_interpolation
        self.depth            = depth
        self.hidden_dim       = hidden_dim
        self.base_size        = input_size // self.patch_size
        self.pos_embeddings_cache_key = None
        self.pos_embeddings           = None

        if isinstance(device, str):
            device = torch.device(device)

        with torch.device(device):

            self.t_block = nn.Sequential(nn.SiLU(),
                                         nn.Linear(hidden_dim, 6 * hidden_dim)
                                         )
            self.x_embedder = PatchEmbedder(patch_size,
                                            input_dim,
                                            hidden_dim
                                            )
            self.t_embedder = TimestepEmbedder(hidden_dim,
                                               positional_dim   = 256,
                                               positional_dtype = torch.float32
                                               )
            self.y_embedder = CaptionEmbedder(context_dim,
                                              hidden_dim
                                              )
            self.blocks = nn.ModuleList([
                PixArtMSBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio = mlp_ratio
                    )
                for i in range(depth)
                ])
            self.final_layer = PixArtFinalLayer(hidden_dim,
                                                patch_size,
                                                self.output_dim
                                                )


    def forward(self,
                x,              # input images (in latent space).
                timesteps,      # time steps in the diffusion process.
                context,           # conditional info (from prompt).
                context_mask=None, # optional attention mask applied to conditional.
                return_eps_only=False, # if True then return only the noise prediction (eps), otherwise return the full output
                **kwargs):
        """
        x         = [batch_size, channels, H, W]
        timesteps = [batch_size]
        cond      = [batch_size, 1, cond_length, cond_dim]
        mask      = [batch_size, cond_length]
        """
        assert not self.training, \
            "This PixArtSigma class can only be used for inference, no ha sido probrada en training."
        assert context_mask is None or _is_valid_mask(context_mask, context), \
            "la context_mask suministrada en forward(..) tiene un formato inadecuado"

        dtype         = self.dtype
        batch_size    = x.shape[0]
        latent_height = x.shape[-2]
        latent_width  = x.shape[-1]
        height        = latent_height // self.patch_size
        width         = latent_width  // self.patch_size
        patch_size    = self.patch_size
        hidden_dim    = self.hidden_dim
        output_dim    = self.output_dim

        x         = x.to(dtype)
        timesteps = timesteps.to(dtype)
        context   = context.to(dtype)

        # forzar que `context` tenga 4 dimensiones, con un formato similar a:
        #   [batch_size, 1, seq_length, embedding_size]
        if len(context.shape) == 3:
            context = context.unsqueeze(1)

        pos_embed = self.cached_position_embeddings(height, width, x.device, x.dtype)
        x  = self.x_embedder(x) + pos_embed       #         [batch_size, patches, hidden_dim]
        t  = self.t_embedder(timesteps)           #                  [batch_size, hidden_dim]
        t6 = self.t_block(t).reshape(batch_size, 6, hidden_dim)#  [batch_size, 6, hidden_dim]
        context = self.y_embedder(context)        #   [batch_size, 1, seq_length, hidden_dim]

        # x = [batch_size, patches, hidden_dim]
        for block in self.blocks:
            x = block(x, t6, context, context_mask, **kwargs)

        # [batch_size, patches, (patch_size^2)*output_dim] <- [batch_size, patches, hidden_dim]
        x = self.final_layer(x, t.unsqueeze(1))

        # unpatchify
        # [batch_size, out_channels, lat_height, lat_width]
        #                <- [batch_size, patches, (patch_size^2)*output_dim]
        assert x.shape[1] == height * width
        x = x.view(batch_size, height, width, patch_size, patch_size, output_dim)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(batch_size, output_dim, latent_height, latent_width)

        if return_eps_only:
            x = x.chunk(2, dim=1)[0]
        return x


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
            torch.Tensor: A tensor of size (1, self.hidden_dim, height, width) containing the
                sinusoidal position embeddings.
        """
        cache_key = (height, width, str(device), str(dtype))

        if cache_key == self.pos_embeddings_cache_key:
            return self.pos_embeddings

        embedding_dim = self.hidden_dim
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
        return self.x_embedder.proj.weight.dtype


    @classmethod
    def from_predefined(cls,
                        model      : str, # [alpha, sigma]
                        image_size : int, # [256, 512, 1024, 2048]
                        ):
        assert model      in ["alpha", "sigma"]
        assert image_size in [256, 512, 1024, 2048]

        max_token_length = {"alpha": 120, "sigma": 300}
        pe_interpolation = {256: 0.5, 512: 1, 1024: 2, 2048: 4}

        kwargs = {
            "input_size"       : image_size // 8,
            "pe_interpolation" : pe_interpolation[image_size],
            "model_max_length" : max_token_length[model],
            "depth"            : 28,
            "hidden_dim"       : 1152,
            "patch_size"       : 2,
            "num_heads"        : 16,
            }
        if model == "alpha" and image_size == 1024:
            kwargs["micro_condition"] = True

        if image_size >= 512:
            model = PixArtSigma( **kwargs )
        else:
            #model = PixArt( **kwargs )
            model = None

        return model

    @classmethod
    def from_safetensors(cls,
                         safetensors_path : PathLike,
                         model            : str,
                         image_size       : int,
                         device           : str
                         ):

        model = cls.from_predefined(model, image_size)
        tensors = {}
        with safe_open(safetensors_path, framework="pt", device=device) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        state_dict, missing_keys = cls.get_pixart_state_dict( tensors )
        if len(missing_keys) > 0:
            print(f"## PixArt DiT conversion has {len(missing_keys)} missing keys!")
        model.load_state_dict(state_dict, strict=False, assign=False)
        return model

    # Freeze all params for inference.
    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()


    # Unfreeze all parameters for training.
    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        self.train()


    @classmethod
    def get_pixart_state_dict(cls, state_dict: Dict) -> Dict:

        cls._init_state_dict_table()
        is_pixart_state_dict = ("blocks.0.cross_attn.kv_linear.weight" in state_dict)

        # si state_dict tiene keys en formato del codigo de referencia de pixart
        # entonces no convertir y tomar state_dict como viene
        if is_pixart_state_dict:
            missing_keys = [ pkey for pkey in cls.PIXART_KEYS if pkey not in state_dict ]
            pixart_state_dict = state_dict

        # si state_dict tiene keys en formato diffusers
        # entonces convertir una x una las keys al formato de referencia de pixart
        else:
            missing_keys = [ dkey for dkey in cls.DIFFUSERS_KEYS if dkey not in state_dict ]
            pixart_state_dict = {}
            for pixart_key, diffusers_key in cls.STATE_DICT_TABLE:

                if "|" not in diffusers_key:
                    _tensor = _find_tensor(state_dict, diffusers_key)
                    if _tensor is not None:
                        pixart_state_dict[pixart_key] = _tensor

                elif "|k+v|" in diffusers_key:
                    _tensor_k = _find_tensor(state_dict, diffusers_key, "k")
                    _tensor_v = _find_tensor(state_dict, diffusers_key, "v")
                    if (_tensor_k is not None) and (_tensor_v is not None):
                        pixart_state_dict[pixart_key] = torch.cat((_tensor_k, _tensor_v))

                elif "|q+k+v|" in diffusers_key:
                    _tensor_q = _find_tensor(state_dict, diffusers_key, "q")
                    _tensor_k = _find_tensor(state_dict, diffusers_key, "k")
                    _tensor_v = _find_tensor(state_dict, diffusers_key, "v")
                    if (_tensor_q is not None ) and (_tensor_k is not None) and (_tensor_v is not None):
                        pixart_state_dict[pixart_key] = torch.cat((_tensor_q, _tensor_k, _tensor_v))

        return pixart_state_dict, missing_keys


    @classmethod
    def _init_state_dict_table(cls):
        if cls.STATE_DICT_TABLE is not None:
            return

        # generar `STATE_DICT_TABLE` desde la info del template STATE_DICT_TABLE_TEMPLATE
        cls.STATE_DICT_TABLE = [ (pkey, dkey) for pkey,dkey in STATE_DICT_TABLE_TEMPLATE if not "|depth|" in pkey ]
        for depth in range(28):
            for pixart_key, diffusers_key in STATE_DICT_TABLE_TEMPLATE:
                if DEPTH_TAG in pixart_key:
                    pixart_key    = pixart_key.replace(DEPTH_TAG, str(depth))
                    diffusers_key = diffusers_key.replace(DEPTH_TAG, str(depth))
                    cls.STATE_DICT_TABLE += [ (pixart_key, diffusers_key) ]
        _sdmap_ = cls.STATE_DICT_TABLE
        cls.STATE_DICT_TABLE = [ ]
        for pixart_key, diffusers_key in _sdmap_:
            if WB_TAG in pixart_key:
                cls.STATE_DICT_TABLE += [
                    ( pixart_key.replace(WB_TAG,"weight"), diffusers_key.replace(WB_TAG,"weight") ),
                    ( pixart_key.replace(WB_TAG,"bias"  ), diffusers_key.replace(WB_TAG,"bias")   )
                    ]
            else:
                cls.STATE_DICT_TABLE.append( (pixart_key, diffusers_key) )

        # generar "DIFFUSERS_KEYS" con el listado de keys necesarias en el formato DIFFUSERS
        cls.DIFFUSERS_KEYS = [ ]
        for _, diffusers_key in cls.STATE_DICT_TABLE:
            if "|" not in diffusers_key:
                cls.DIFFUSERS_KEYS.append(diffusers_key)
            elif KV_TAG in diffusers_key:
                cls.DIFFUSERS_KEYS.append( diffusers_key.replace(KV_TAG, "k") )
                cls.DIFFUSERS_KEYS.append( diffusers_key.replace(KV_TAG, "v") )
            elif QKV_TAG in diffusers_key:
                cls.DIFFUSERS_KEYS.append( diffusers_key.replace(QKV_TAG, "q") )
                cls.DIFFUSERS_KEYS.append( diffusers_key.replace(QKV_TAG, "k") )
                cls.DIFFUSERS_KEYS.append( diffusers_key.replace(QKV_TAG, "v") )

        # generar "PIXART_KEYS" con el listado de keys necesarias en el formato PIXART
        cls.PIXART_KEYS = [pixart_key for pixart_key, _ in cls.STATE_DICT_TABLE]

