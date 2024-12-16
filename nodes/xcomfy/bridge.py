"""
File     : comfy_bridge.py
Purpose  : A layer of compatibility between ComfyUI and xPixArt nodes.
Author   : Martin Rizzo | <martinrizzo@gmail.com>
Date     : May 10, 2024
Repo     : https://github.com/martin-rizzo/ComfyUI-xPixArt
License  : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import os
import torch
from safetensors            import safe_open
from comfy.model_base       import BaseModel, ModelType
from comfy                  import supported_models_base
from comfy                  import model_management
from comfy                  import latent_formats
from comfy                  import conds
from ..core.pixart_model_ex import PixArtModelEx as PixArtModel
from ..utils.safetensors    import load_safetensors_header, estimate_model_params
from ..utils.system         import logger


#============================================================================
# Clases de configuracion similares a las definidas in `/comfy/supported_models.py`
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/supported_models.py

class PixArt_Config(supported_models_base.BASE):

    unet_config = {
        "latent_img_size"    :    -1, #  <---- sera configurado en __init__
        "latent_img_channels":     4, # number of channels in the latent image
        "internal_dim"       :  1152, # internal dimensionality used
        "caption_dim"        :  4096, # dimensionality of the caption input (T5 encoded prompt)
        "patch_size"         :     2, # size of each patch (in latent blocks)
        "num_heads"          :    16, # number of attention heads in the transformer
        "depth"              :    28, # number of layers in the transformer
        #--- old // to delete ---#
        "input_size"      :    -1, #  <---- sera configurado en __init__
        "pe_interpolation":    -1, #  <---- sera configurado en __init__
        "context_len"     :   300, #  300 tokens maximo en el prompt
        "input_dim"       :     4, #    4 channels in latent image
        "hidden_dim"      :  1152, # 1152 channels usados internamente
        "context_dim"     :  4096, # 4096 features por cada prompt token
        }

    unet_extra_config = {
        }

    sampling_settings = {
        "beta_schedule" : "sqrt_linear",
        "linear_start"  : 0.0001,
        "linear_end"    : 0.02,
        "timesteps"     : 1000,
        }

    latent_format = latent_formats.SDXL
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    def __init__(self, image_size):
        super().__init__( self.__class__.unet_config )
        self.unet_config["latent_img_size"]  = image_size//8
        self.unet_config["input_size"]       = image_size//8
        self.unet_config["pe_interpolation"] = image_size//512

    def get_model(self, state_dict, prefix="", device=None):
        out = PixArt(self, device=device)
        return out

    def process_unet_state_dict(self, state_dict):
        # state_dict, missing_keys = PixArtOldModel.get_pixart_state_dict(state_dict)
        # if len(missing_keys) > 0:
        #     logger.debug(f"PixArt DiT conversion has {len(missing_keys)} missing keys!")
        #     for i, key in enumerate(missing_keys):
        #         if i>4: logger.debug("    ....") ; break
        #         logger.debug(f"    - {key}")
        #     print()
        return state_dict


#===========================================================================#
# A class similar to the classes defined in `/comfy/model_base.py`
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_base.py

class PixArt(BaseModel):

    def __init__(self,
                 model_config,
                 model_type  : ModelType    = ModelType.EPS,
                 device      : torch.device = None
                 ):
        super().__init__(model_config, model_type, device=device, unet_model=PixArtModel)


    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        out["return_epsilon"] = conds.CONDConstant(True)

        cond_attn_mask = kwargs.get("cond_attn_mask", None)
        if cond_attn_mask is not None:
            out["context_mask"] = conds.CONDRegular(cond_attn_mask)

        return out


## ??? --------------------

def model_config_from_dit(state_dict, unet_key_prefix, use_sigma2K_if_no_match=False):
    model_config = PixArt_Config(image_size=1024)
    if model_config is None and use_sigma2K_if_no_match:
        return PixArt_Config(image_size=2048)
    else:
        return model_config



# pequeña emulacion al comportamiento de "load_unet_state_dict(sd)"
# - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd.py
def create_model_from_safetensors(safetensors_path,
                                  load_device,
                                  offload_device,
                                  prefix="",
                                  dtype=None
                                  ):

    header       = load_safetensors_header(safetensors_path)
    model_config = model_config_from_dit(header, prefix)

    # obtener informacion relacionada al modelo a cargar
    parameters          = estimate_model_params(safetensors_path, prefix)
    unet_dtype          = model_management.unet_dtype(model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
    initial_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
    manual_cast_dtype   = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)

    # preconfigurar los dtype del modelo
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    # obtener los parametros desde el archivo safetensors
    state_dict = {}
    safe_device = initial_load_device if isinstance(initial_load_device,str) else initial_load_device.type
    with safe_open(safetensors_path, framework="pt", device=safe_device) as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    # crear el model
    model = model_config.get_model(state_dict, prefix="", device=initial_load_device)

    # cargar los parametros dentro del model
    model.load_model_weights(state_dict, prefix)
    model.diffusion_model.to(unet_dtype)

    ## DEBUG
    logger.info(f"The model '{os.path.basename(safetensors_path)}' was loaded successfully.")
    logger.debug(f"Model details:")
    logger.debug(f" - parameters          : {parameters}"         )
    logger.debug(f" - unet_dtype          : {unet_dtype}"         )
    logger.debug(f" - initial_load_device : {initial_load_device}")
    logger.debug(f" - load_device         : {load_device}"        )
    logger.debug(f" - offload_device      : {offload_device}"     )
    return model;
