"""
File    : set_noise_seed.py
Purpose : Node that sets the noise seed, packing it into the generation parameters.
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

class SetNoiseSeed:
    TITLE       = "xPixArt | Set Noise Seed"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Allows to set the noise seed within a group of generation parameters."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gparams"   : ("GPARAMS", {"tooltip": "The original generation parameters which will be updated."}),
                "key"       : ("STRING" , {"tooltip": "The full key of the parameter to modify. This key must include the prefix if any."}),
                "noise_seed": ("INT"    , {"tooltip": "The noise seed number to apply.", "default": 1, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    #-- FUNCTION --------------------------------#
    FUNCTION = "set_noise_seed"
    RETURN_TYPES    = ("GPARAMS",)
    RETURN_NAMES    = ("gparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated noise seed.",)

    def set_noise_seed(self, gparams: GParams, key: str, noise_seed: int):
        gparams = gparams.copy()
        gparams.set(key, noise_seed)
        return (gparams,)
