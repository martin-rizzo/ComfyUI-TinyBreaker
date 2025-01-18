"""
File    : set_image_tweaks.py
Purpose : Node to fine-tune image parameters that achieve subtle improvements in the image.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 18, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.gen_params import GenParams

_NFACTORS_BY_REFINAMENT = {
    "disabled": -10000,
    "minimal" : -2,
    "low"     : -1,
    "normal"  : 0,
    "high"    : +1,
    "intense" : +2,
    "extreme" : +3,
}
_DEFAULT_REFINAMENT="normal"


class SetImageTweaks:
    TITLE       = "💪TB | Set Image Tweaks"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Fine-tune image parameters that achieve subtle improvements in the image."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "genparams" : ("GENPARAMS"      , {"tooltip": "The original generation parameters which will be updated."}),
                "variant"   : ("INT"            , {"tooltip": "The variant of the image to generate. This parameter allows you to choose between different versions of the same image.",
                                                   "default": 1, "min": 1, "max": 0xffffffffffffffff
                                                   }),
                "refinement": (cls.refinements(), {"tooltip": "The level of refinement for the image.",
                                                   "default": _DEFAULT_REFINAMENT
                                                   }),
                "cfg_correction": ("FLOAT"      , {"tooltip": "An adjustment applied to the classifier-free guidance value. Positive values increase prompt adherence; negative values allow more model freedom. A value of 0.0 uses the default setting.",
                                                   "default": 0.0, "min": -4.0, "max": 4.0, "step": 0.2, "round": 0.01
                                                   }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "set_image_tweaks"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the tweaks applied.",)

    def set_image_tweaks(self,
                         genparams     : GenParams,
                         variant       : int   = 1,
                         refinement    : str   = _DEFAULT_REFINAMENT,
                         cfg_correction: float = 0.0,
                         ):
        genparams = genparams.copy()
        genparams.set("refiner.noise_seed"   ,int(   variant                             ))
        genparams.set("refiner.steps_nfactor",int(   _NFACTORS_BY_REFINAMENT[refinement] ))
        genparams.add("base.cfg"             ,float( cfg_correction                      ))
        return (genparams,)


    #__ internal functions ________________________________

    @staticmethod
    def refinements():
        return list(_NFACTORS_BY_REFINAMENT.keys())
