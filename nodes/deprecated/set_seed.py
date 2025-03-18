"""
File    : set_seed.py
Purpose : Node to set a seed number, packing it into the generation parameters.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 21, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from ..core.genparams import GenParams

class SetSeed:
    TITLE       = "💪TB | Set Seed"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Allows to set a seed number, packing it into GenParams."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams":("GENPARAMS", {"tooltip": "The generation parameters to be updated.",
                                       }),
            "key"      :("STRING"   , {"tooltip": "The full key of the parameter to modify. This key must include the prefix if any.",
                                       "default": "base.noise_seed"
                                       }),
            "seed"     :("INT"      , {"tooltip": "The seed number to apply.",
                                       "default": 1, "min": 1, "max": 0xffffffffffffffff,
                                       }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "set_seed"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated seed number. (you can use this output to chain other genparams nodes)",)

    def set_seed(self, genparams: GenParams, key: str, seed: int):
        genparams = genparams.copy()
        genparams.set_int(f"denoising.{key}", seed)
        return (genparams,)
