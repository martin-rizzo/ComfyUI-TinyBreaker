"""
File     : model_detection.py
Purpose  : A PixArt-adapted version of `comfy/model_detection.py` code,
           it offers similar functionality but with a different structure.
Author   : Martin Rizzo | <martinrizzo@gmail.com>
Date     : May 12, 2024
Repo     : https://github.com/martin-rizzo/ComfyUI-xPixArt
License  : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .supported_models import PixArt


def model_config_from_dit(state_dict, unet_key_prefix, use_sigma2K_if_no_match=False):
    model_config = PixArt(image_size=1024)
    if model_config is None and use_sigma2K_if_no_match:
        return PixArt(image_size=2048)
    else:
        return model_config
