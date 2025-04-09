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
from torch                import Tensor
from comfy                import model_management
from .t5_encoder_model    import Custom_nn
from .t5_encoder_model_ex import T5EncoderModelEx


#--------------------------------- HELPERS ---------------------------------#

class CmfCustom_nn(Custom_nn):
    """A adaptation of `Custom_nn` to use with comfyui.
    Defines a custom `Embedding` that can be forced to generate embeddings with a specific dtype.
    """
    class Embedding(Custom_nn.Embedding):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.forced_dtype = None # <- this attribute can be set externally to force the inference dtype

        def forward(self, input: Tensor, dtype: torch.dtype = None, out_dtype: torch.dtype = None) -> Tensor:
            return super().forward(input, dtype=self.forced_dtype or dtype or out_dtype)


#===========================================================================#
#////////////////////////// T5 ENCODER MODEL CMFY //////////////////////////#
#===========================================================================#

def get_t5_encoder_custom_class(load_device: str | torch.device,
                                inference_device: torch.device,
                                inference_dtype : torch.dtype,
                                precharge_depth : int = 2
                                ) -> type:
    """
    Returns a custom `T5EncoderModelEx` class customized for ComfyUI.

    This function creates a customized `T5EncoderModelEx` class that can be used
    with ComfyUI and will respond to the configuration provided by the parameters.

    Args:
        inference_device: The device on which to perform the inference.
        inference_dtype : The data type to use for performing the inference
        precharge_depth : Number of layers to pre-charge in GPU when performing inference with dynamic GPU allocation.
    """


    class T5EncoderModelCmfy(T5EncoderModelEx):
        """
        A ComfyUI compatible version of `T5EncoderModelEx`.
        Args:
            config_dict: A dictionary containing the configuration of the T5 model.
            dtype      : The data type used to store the model parameters.
            device     : The device used to store the model parameters.
            operations : A ComfyUI custom class that overrides the `torch.nn` modules.
        """
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
                nn     = CmfCustom_nn,
            )

        def forward(self,
                    input_ids                    : torch.Tensor,
                    attention_mask               : torch.Tensor = None,
                    embeds                       : torch.Tensor = None,
                    num_tokens                   : int          = None,
                    intermediate_output          : int          = None,
                    final_layer_norm_intermediate: bool         = False,
                    dtype                        : torch.dtype  = None,
                    ) -> tuple[torch.Tensor]:

            dtype            = dtype or torch.float32
            memory_required  = 2024*1024*1024
            model_management.free_memory(memory_required, model_management.get_torch_device())

            # if comfyui only requests the output of the inference (no intermediate layers)
            if intermediate_output is None:
                x = super().forward(
                        input_ids,
                        input_embeds    = embeds,
                        attention_mask  = attention_mask,
                        precharge_depth = precharge_depth,
                        device = inference_device,
                        dtype  = inference_dtype or dtype
                        )
                x = x.to(dtype)
                return x, x

            # if comfyui requests the output of some intermediate layer
            else:
                x, intermediate = super().forward(
                        input_ids,
                        input_embeds                    = embeds,
                        attention_mask                  = attention_mask,
                        return_intermediate             = True,
                        intermediate_index              = intermediate_output,
                        intermediate_must_be_normalized = final_layer_norm_intermediate,
                        precharge_depth                 = precharge_depth,
                        device = inference_device,
                        dtype  = inference_dtype or dtype
                        )
                x            = x.to(dtype)
                intermediate = intermediate.to(dtype)
                return x, intermediate


        def get_input_embeddings(self):
            # ATTENTION: this code force data type of embeddings to be the same
            #            as the `inference_dtype` parameter.
            #
            # ComfyUI uses `get_input_embeddings()` method to build the embeddings
            # since version 0.3.23 (commit 85ef295):
            #  https://github.com/comfyanonymous/ComfyUI/blob/v0.3.23/comfy/sd1_clip.py#L195
            #
            self.shared.forced_dtype = inference_dtype
            return self.shared


    return T5EncoderModelCmfy

