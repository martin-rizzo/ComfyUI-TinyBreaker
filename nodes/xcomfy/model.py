"""
File    : xconfy/model.py
Purpose : The standard MODEL object transmitted through ComfyUI's node system.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 10, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import comfy.model_detection
import torch
import comfy.utils
import comfy.model_patcher
from comfy                  import model_management, supported_models_base, latent_formats, conds
from comfy.model_base       import BaseModel, ModelType
from ..utils.system         import logger
from ..core.pixart_model_ex import PixArtModelEx


def normalize_prefix(prefix: str) -> str:
    """Normalize a given prefix"""
    prefix = prefix.strip()
    if prefix and prefix != "*" and not prefix.endswith('.'):
        prefix += '.'
    return prefix


#--------------------------------- PIXART ----------------------------------#

class _PixArtConfig(supported_models_base.BASE):
    """
    The configuration class for PixArt models used for compatibility with ComfyUI.
    This class emulates the standard configurations defined in `/comfy/supported_models.py`
    - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/supported_models.py
    """

    unet_extra_config = {
        }

    sampling_settings = {
        "beta_schedule" : "sqrt_linear",
        "linear_start"  : 0.0001,
        "linear_end"    : 0.02,
        "timesteps"     : 1000,
        }

    latent_format              = latent_formats.SDXL
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    def __init__(self, state_dict, prefix, resolution=1024):
        unet_config = PixArtModelEx.infer_model_config(state_dict, prefix=prefix, resolution=resolution)
        super().__init__( unet_config )

    def get_model(self, state_dict, prefix="", device=None):
        out = _PixArt(self, device=device)
        return out


class _PixArt(BaseModel):
    """
    A wrapper for the PixArt model used for compatibility with ComfyUI.
    This class emulates the standard model wrappers defined in `/comfy/model_base.py`
    - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_base.py
    """

    def __init__(self,
                 model_config: _PixArtConfig,
                 model_type  : ModelType    = ModelType.EPS,
                 device      : torch.device = None
                 ):
        super().__init__(model_config, model_type, device=device, unet_model=PixArtModelEx)
        self.diffusion_model.freeze()

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        out["return_epsilon"] = conds.CONDConstant(True)

        cond_attn_mask = kwargs.get("cond_attn_mask", None)
        if cond_attn_mask is not None:
            out["context_mask"] = conds.CONDRegular(cond_attn_mask)

        return out


def _model_config_from_unet(state_dict: dict,
                            prefix    : str,
                            use_base_if_no_match: bool = False,
                            resolution          : int  = 1024
                            ) -> tuple[supported_models_base.BASE, dict, str]:
    """
    Detects the model configuration from a given UNet state dictionary.
    It returns the appropriate configuration class and the normalized state dictionary and prefix.
    This function wraps the `model_config_from_unet` function defined in `/comfy/model_detection.py`.
     - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_detection.py
    """

    # check if PixArt is present in the state dict,
    # if so then execute the custom code for PixArt
    pixprefix = PixArtModelEx.detect_prefix(state_dict)
    print("##>> pxiprefix:", pixprefix)
    print("##>> prefix:", prefix)

    if PixArtModelEx.detect_prefix(state_dict, prefix) is not None:
        logger.info("Detected PixArt model")
        state_dict, prefix = PixArtModelEx.normalize_state_dict(state_dict, prefix)
        model_config       = _PixArtConfig(state_dict, prefix, resolution=resolution)
        return model_config, state_dict, prefix

    # by default use the normal detection of ComfyUI
    if prefix and not prefix.endswith("."):
        prefix += "."
    model_config = comfy.model_detection.model_config_from_unet(state_dict, prefix, use_base_if_no_match)
    return model_config, state_dict, prefix



#===========================================================================#
#////////////////////////////////// MODEL //////////////////////////////////#
#===========================================================================#

class Model(comfy.model_patcher.ModelPatcher):
    """This class represents any MODEL object transmitted through the ComfyUI's node system."""

    @classmethod
    def from_state_dict(cls,
                        state_dict: dict,
                        prefix    : str  = "",
                        resolution: int  = 1024,
                        ) -> "Model":
        """
        Create an instance of this class from the state dict of a PixArt model.
        Args:
            state_dict (dict): The state dictionary of the PixArt model.
            prefix      (str): The prefix used in the state dictionary.
            resolution  (int): The base resolution the model is intended to work at.
        """
        model_config, state_dict, prefix = _model_config_from_unet(state_dict, prefix, resolution=resolution)
        if model_config is None:
            raise ValueError("Unsupported model type")

        # get information related to the model to be loaded
        parameters          = comfy.utils.calculate_parameters(state_dict, prefix)
        unet_dtype          = model_management.unet_dtype(model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
        initial_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
        load_device         = model_management.get_torch_device()
        offload_device      = model_management.unet_offload_device()
        manual_cast_dtype   = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)

        # preconfigure the model dtype
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

        # create the model
        model = model_config.get_model(state_dict, prefix, device=initial_load_device)

        # load the parameters into the model
        model.load_model_weights(state_dict, prefix)
        model.diffusion_model.to(unet_dtype)

        # report the model details to the user
        logger.info(f"The model was loaded successfully.")
        logger.debug(f"Model details:")
        logger.debug(f" - parameters          : {parameters}"         )
        logger.debug(f" - unet_dtype          : {unet_dtype}"         )
        logger.debug(f" - initial_load_device : {initial_load_device}")
        logger.debug(f" - load_device         : {load_device}"        )
        logger.debug(f" - offload_device      : {offload_device}"     )

        return cls(model, load_device=load_device, offload_device=offload_device, size=0, weight_inplace_update=False)

