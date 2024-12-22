"""
File    : set_float.py
Purpose : Node that sets a float value, packing it into the generation parameters.
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

class SetFloat:
    TITLE       = "xPixArt | Set Float"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Allows to set a float value within a group of generation parameters."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gparams": ("GPARAMS", {"tooltip": "The original generation parameters which will be updated."}),
                "key"    : ("STRING" , {"tooltip": "The key of the actual parameter to modify."}),
                "value"  : ("FLOAT"  , {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The float value to set."}),
            }
        }

    #-- FUNCTION --------------------------------#
    FUNCTION = "set_gparam"
    RETURN_TYPES    = ("GPARAMS",)
    RETURN_NAMES    = ("gparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated value.",)

    def set_gparam(self, gparams: GParams, key: str, value: float):
        gparams = gparams.copy()
        gparams.set(key, value)
        return (gparams,)
