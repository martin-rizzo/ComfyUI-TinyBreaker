"""
File    : set_float.py
Purpose : Node that sets a float value, packing it into the generation parameters.
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

class SetFloat:
    TITLE       = "💪TB | Set Float"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Allows to set a float value, packing it into GenParams."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "genparams":("GENPARAMS", {"tooltip": "The original generation parameters which will be updated."
                                       }),
            "key"      :("STRING"   , {"tooltip": "The full key of the parameter to modify. This key must include the prefix if any.",
                                       "default": "base.cfg"
                                       }),
            "value"    :("FLOAT"    , {"tooltip": "The float value to set.",
                                       "default": 8.0, "step":0.1, "round": 0.01
                                       }),
            }
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "set_float"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated value. (you can use this output to chain other genparams nodes)",)

    def set_float(self, genparams: GenParams, key: str, value: float):
        genparams = genparams.copy()
        genparams.set(key, float(value))
        return (genparams,)
