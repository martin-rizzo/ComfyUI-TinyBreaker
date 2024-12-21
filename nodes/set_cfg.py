"""
File    : set_cfg.py
Desc    : Node that sets the Classifier-Free Guidance inside of `gparams`
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 21, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.gparams import GParams

class SetCFG:
    TITLE       = "xPixArt | Set Classifier-Free Guidance"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Allows to set the Classifier-Free Guidance as part of the generation parameters (gparams)."


    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gparams": ("GPARAMS", {"tooltip": "The generation parameters to modify."}),
                "prefix" : ("STRING" , {"default": "base", "tooltip": "The prefix used to identify the unpacked parameters."}),
                "cfg"    : ("FLOAT"  , {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
            }
        }

    #-- FUNCTION --------------------------------#
    FUNCTION = "set_cfg"
    RETURN_TYPES    = ("GPARAMS",)
    RETURN_NAMES    = ("gparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the Classifier-Free Guidance set.",)


    def set_cfg(self, gparams: GParams, prefix: str, cfg: float):
        KEY = "cfg"
        gparams = gparams.copy()
        gparams.set_prefixed(prefix, KEY, cfg)
        return (gparams,)
