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
    DESCRIPTION = "Allows to set a float value within a group of generation parameters."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "genparams": ("GENPARAMS", {"tooltip": "The original generation parameters which will be updated."}),
                "key"      : ("STRING"   , {"tooltip": "The key of the actual parameter to modify."}),
                "value"    : ("FLOAT"    , {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The float value to set."}),
            }
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "set_float"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated value.",)

    def set_float(self, genparams: GenParams, key: str, value: float):
        genparams = genparams.copy()
        genparams.set(key, value)
        return (genparams,)
