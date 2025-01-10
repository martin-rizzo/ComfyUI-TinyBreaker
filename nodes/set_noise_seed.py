"""
File    : set_noise_seed.py
Purpose : Node that sets the noise seed, packing it into the generation parameters.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 21, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.gen_params import GenParams

class SetNoiseSeed:
    TITLE       = "💪TB | Set Noise Seed"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Allows to set the noise seed within a group of generation parameters."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "genparams" : ("GENPARAMS", {"tooltip": "The original generation parameters which will be updated."}),
                "key"       : ("STRING"   , {"tooltip": "The full key of the parameter to modify. This key must include the prefix if any."}),
                "noise_seed": ("INT"      , {"tooltip": "The noise seed number to apply.", "default": 1, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "set_noise_seed"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated noise seed.",)

    def set_noise_seed(self, genparams: GenParams, key: str, noise_seed: int):
        genparams = genparams.copy()
        genparams.set(key, noise_seed)
        return (genparams,)
