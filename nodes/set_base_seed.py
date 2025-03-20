"""
File    : set_base_seed.py
Purpose : Node to set a seed number for the base model and integrate it into the generation parameters
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 17, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.genparams import GenParams

class SetBaseSeed:
    TITLE       = "💪TB | Set Base Seed"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Allows to set the noise seed number for the base model."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams":("GENPARAMS", {"tooltip": "The generation parameters to be updated.",
                                      }),
            "seed"     :("INT"      , {"tooltip": "The seed number to apply.",
                                       "default": 1, "min": 1, "max": 0xffffffffffffffff,
                                       "control_after_generate": True,
                                      }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "set_base_noise_seed"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated seed number. (you can use this output to chain other genparams nodes)",)

    def set_base_noise_seed(self, genparams: GenParams, seed: int):
        genparams = genparams.copy()
        genparams.set_int(f"denoising.base.noise_seed", seed)
        return (genparams,)
