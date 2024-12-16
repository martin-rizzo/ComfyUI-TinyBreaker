"""
File    : pixart_model_ex.py
Purpose : Extension of the PixArtModel class to include additional functionality.
          Includes autodetection of model comfiguation and HF diffusers format support.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 15, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
from .pixart_model import PixArtModel


def _detect_prefix(state_dict: dict, key_1:str, key_2:str) -> str:
    """
    Detects the model prefix based on the presence of two keys.
    """
    for key in state_dict.keys():
        if key.endswith(key_1):
            prefix = key[:-len(key_1)]
            if f"{prefix}{key_2}" in state_dict:
                return prefix
        elif key.endswith(key_2):
            prefix = key[:-len(key_2)]
            if f"{prefix}{key_1}" in state_dict:
                return prefix
    return ""

def _find_tensor(state_dict: dict, template_key: str, subkey: str = None) -> torch.Tensor:
    """
    Finds a tensor in the state_dict using a template key.
    """
    if subkey is None:
        return state_dict.get(template_key)
    else:
        parts = template_key.split("|")
        return state_dict.get(parts[0] + subkey + parts[2])


#----------------------- HF DIFFUSERS FORMAT SUPPORT -----------------------#

class _DiffusersFormatConverter:
    """
    Converts a state dictionary from Hugging Face Diffusers format to standard PixArt format.

    This converter handles specific mappings between Hugging Face Diffusers model
    parameters and standard PixArt parameters. It also handles specific cases where
    certain parameters need to be renamed or modified.

    Args:
        state_dict (dict): The state dictionary from Hugging Face Diffusers format.
        prefix     (str) : A prefix to filter keys in the state dictionary.
                           If "*", it will automatically detect the correct prefix.
    Returns:
        A tuple containing the converted state dictionary and the updated prefix.
    """
    def __call__(self, state_dict: dict, prefix: str) -> tuple[dict, str]:
        KEY_1, KEY_2 = self.KEY_1, self.KEY_2
        KV_TAG, QKV_TAG  = self.KV_TAG, self.QKV_TAG
        original_state_dict, original_prefix = state_dict, prefix

        # auto-detect prefix if the special prefix "*" is used
        if prefix == "*":
            prefix = _detect_prefix(state_dict, KEY_1, KEY_2)

        # ensure that prefix always ends with a period '.'
        if prefix and not prefix.endswith('.'):
            prefix += '.'

        # if the characteristic tensors of this format are not present
        # then it is not HF diffusers PixArt model, so we return the original values
        if (f"{prefix}{KEY_1}" not in state_dict) or (f"{prefix}{KEY_2}" not in state_dict):
            return original_state_dict, original_prefix

        # remove prefix and generate a pixart compatible state dict
        if prefix:
            state_dict = {key[len(prefix):]: tensor for key, tensor in state_dict.items() if key.startswith(prefix)}
            prefix     = ""
        pixart_state_dict = {}
        for pixart_key, diffusers_key in self.get_convertion_table():

            if "|" not in diffusers_key:
                _tensor = _find_tensor(state_dict, diffusers_key)
                if _tensor is not None:
                    pixart_state_dict[pixart_key] = _tensor

            elif KV_TAG in diffusers_key:
                _tensor_k = _find_tensor(state_dict, diffusers_key, "k")
                _tensor_v = _find_tensor(state_dict, diffusers_key, "v")
                if (_tensor_k is not None) and (_tensor_v is not None):
                    pixart_state_dict[pixart_key] = torch.cat((_tensor_k, _tensor_v))

            elif QKV_TAG in diffusers_key:
                _tensor_q = _find_tensor(state_dict, diffusers_key, "q")
                _tensor_k = _find_tensor(state_dict, diffusers_key, "k")
                _tensor_v = _find_tensor(state_dict, diffusers_key, "v")
                if (_tensor_q is not None ) and (_tensor_k is not None) and (_tensor_v is not None):
                    pixart_state_dict[pixart_key] = torch.cat((_tensor_q, _tensor_k, _tensor_v))

        return pixart_state_dict, prefix

    # these are characteristic tensor names for the HF Diffusers model
    KEY_1 = "adaln_single.emb.timestep_embedder.linear_2.bias"
    KEY_2 = "scale_shift_table"

    # tags used to map tensor names in the conversion table template
    DEPTH_TAG = "|depth|"
    WB_TAG    = "|w,b|"
    KV_TAG    = "|k+v|"
    QKV_TAG   = "|q+k+v|"

    # this template is used to create the conversion table
    CONVERSION_TABLE_TEMPLATE = [
            #         PixArt Reference keys             |               HF Diffusers keys                    #
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

    def get_conversion_table(self) -> list[tuple[str, str]]:
        DEPTH_TAG, WB_TAG  = self.DEPTH_TAG, self.WB_TAG

        # if `_conversion_table` is already generated, return it directly
        if hasattr(self, "_conversion_table"):
            return self._conversion_table

        # generate `_conversion_table` from the info of the template CONVERSION_TABLE_TEMPLATE
        table = [ (pkey, dkey) for pkey,dkey in self.CONVERSION_TABLE_TEMPLATE if not DEPTH_TAG in pkey ]
        for depth in range(28):
            for pixart_key, diffusers_key in self.CONVERSION_TABLE_TEMPLATE:
                if DEPTH_TAG in pixart_key:
                    pixart_key    =    pixart_key.replace(DEPTH_TAG, str(depth))
                    diffusers_key = diffusers_key.replace(DEPTH_TAG, str(depth))
                    table += [ (pixart_key, diffusers_key) ]
        _sdmap_ = table
        table = [ ]
        for pixart_key, diffusers_key in _sdmap_:
            if WB_TAG in pixart_key:
                table += [
                    ( pixart_key.replace(WB_TAG,"weight"), diffusers_key.replace(WB_TAG,"weight") ),
                    ( pixart_key.replace(WB_TAG,"bias"  ), diffusers_key.replace(WB_TAG,"bias")   )
                    ]
            else:
                table.append( (pixart_key, diffusers_key) )

        self._conversion_table = table
        return self._conversion_table




class PixArtModelEx(PixArtModel):

    @property
    def dtype(self):
        """Returns the data type of the model parameters."""
        return self.x_embedder.proj.weight.dtype

    @property
    def device(self):
        """Returns the device on which the model parameters are located."""
        return self.x_embedder.proj.weight.device

    @classmethod
    def from_state_dict(cls,
                        state_dict       : dict,
                        config           : dict = None,
                        prefix           : str  = None,
                        format_converters: list = None,
                        ) -> "PixArtModel":
        """
        Creates a PixArtModel instance from a state dictionary.

        Args:
            state_dict: A dictionary containing the model's state.
                        The keys represent the parameter names, and the values are
                        the corresponding tensors.
            config    : A dictionary containing the model's configuration.
                        If None, the configuration is inferred from the state dictionary.
            prefix    : An optional prefix used to filter the state dictionary keys.
                        If "?", an attempt is made to auto-detect the prefix.
            format_converters: A list of format converters. Each converter should be a
                               callable that takes a state dictionary as input and returns
                               a modified state dictionary.
                               These converters are used to adapt different state dictionary
                               formats to the expected format of PixArtModel.
        """
        # first try with any of the format converters (e.g., converter for diffusers format)
        if format_converters:
            for converter in format_converters:
                state_dict, prefix = converter(state_dict, prefix)

        # the "*" prefix is a special case that means the prefix should be detected automatically
        if prefix == "*":
            prefix = _detect_prefix(state_dict, "t_embedder.mlp.0.weight", "final_layer.scale_shift_table")

        # filter the state dictionary by prefix
        # (this allows loading of partial models or sub-models from a larger state dictionary)
        if prefix:
            prefix_length = len(prefix)
            state_dict = {key[prefix_length:]: tensor for key, tensor in state_dict.items() if key.startswith(prefix)}

        # if no config was provided then try to infer it automatically from the keys of the state_dict
        if not config:
            config = cls.infer_model_config(state_dict)

        model = cls( **config )
        model.load_state_dict(state_dict, string=False, assign=False)
        return model


    @classmethod
    def infer_model_config(cls, state_dict: dict, prefix = "", resolution: int = 1024) -> dict:
        assert resolution==2048 or resolution==1024 or resolution==512 or resolution==256, f"PixArtModelEx: Unsupported resolution: {resolution}px"

        # TODO: implement this method to infer the model configuration from a state dictionary
        config = {
            "latent_img_size"     : int(resolution//8), # 64 = 512x512px || 128 = 1024x1024px || 256 = 2048x2048px
            "latent_img_channels" :    4, # number of channels in the latent image
            "internal_dim"        : 1152, # internal dimensionality used
            "caption_dim"         : 4096, # dimensionality of the caption input (T5 encoded prompt)
            "patch_size"          :    2, # size of each patch (in latent blocks)
            "num_heads"           :   16, # number of attention heads in the transformer
            "depth"               :   28, # number of layers in the transformer
            "mlp_ratio"           :  4.0, # ratio of the hidden dimension to the mlp dimension
        }
        return config


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
