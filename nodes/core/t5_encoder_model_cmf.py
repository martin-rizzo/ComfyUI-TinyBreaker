"""
File    : t5_encoder_model_cmf.PY
Purpose : A T5EncoderModel subclass that is fully compatible with ComfyUI.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Feb 26, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
from .t5_encoder_model import T5EncoderModel

class T5EncoderModelCmf(T5EncoderModel):

    def __init__(self,
                 config_dict: dict,
                 dtype      : str | torch.dtype,
                 device     : str | torch.device,
                 operations : type
                 ):

        # comfyui needs the 'self.num_layers' attribute
        self.num_layers = config_dict.get("num_layers", 24)

        # call the parent class constructor using the `config_dict`
        super().__init__(
            d_ff               = config_dict.get("d_ff"              , 10240),
            d_kv               = config_dict.get("d_kv"              ,    64),
            d_model            = config_dict.get("d_model"           ,  4096),
            dense_act_fn       = config_dict.get("dense_act_fn"      , "gelu_pytorch_tanh"),
            layer_norm_epsilon = config_dict.get("layer_norm_epsilon",  1e-6),
            model_type         = config_dict.get("model_type"        ,  "t5"),
            num_heads          = config_dict.get("num_heads"         ,    64),
            num_layers         = self.num_layers,
            is_gated_act       = config_dict.get("is_gated_act"      ,  True),
            vocab_size         = config_dict.get("vocab_size"        , 32128),
            dropout_rate       = config_dict.get("dropout_rate"      ,   0.1),
            relative_attention_num_buckets  = config_dict.get("relative_attention_num_buckets",   32),
            relative_attention_max_distance = config_dict.get("relative_attention_max_distance", 128),
            dtype  = dtype,
            device = device,
            nn     = None,
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask               : torch.Tensor,
                intermediate_output          : int         = None,
                final_layer_norm_intermediate: bool        = False,
                dtype                        : torch.dtype = None,
                ) -> tuple[torch.Tensor]:
        dtype = dtype or torch.float32

        if intermediate_output:
            # when comfyui requests the output of some intermediate layer
            x, intermediate = super().forward(
                input_ids                       = input_ids,
                attention_mask                  = attention_mask,
                return_intermediate             = True,
                intermediate_index              = intermediate_output,
                intermediate_must_be_normalized = final_layer_norm_intermediate,
                dtype = dtype
                )
            return x, intermediate

        else:
            # when comfyui requires the output of the inference only (no intermediate layers)
            x = super().forward(input_ids      = input_ids,
                                attention_mask = attention_mask,
                                dtype = dtype
                                )
            return x, x
