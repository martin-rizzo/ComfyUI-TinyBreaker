"""
File    : set_cfg.py
Desc    : Node that sets the Classifier-Free Guidance inside of `genparams`
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

class SetCFG:
    TITLE       = "💪TB | Set Classifier-Free Guidance"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Allows to set the Classifier-Free Guidance as part of the generation parameters (genparams)."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "genparams": ("GENPARAMS", {"tooltip": "The generation parameters to modify."}),
                "prefix"   : ("STRING" , {"default": "base", "tooltip": "The prefix used to identify the unpacked parameters."}),
                "cfg"      : ("FLOAT"  , {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
            }
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "set_cfg"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the Classifier-Free Guidance set.",)


    #__ internal functions ________________________________

    @staticmethod
    def set_cfg(genparams: GenParams, prefix: str, cfg: float):
        KEY = "cfg"
        genparams = genparams.copy()
        genparams.set_prefixed(prefix, KEY, cfg)
        return (genparams,)
