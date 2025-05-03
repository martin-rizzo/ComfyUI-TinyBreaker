"""
File    : set_image_details_v2.py
Purpose : Node to fine-tune image parameters that achieve subtle improvements in the image.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 18, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.genparams import GenParams
from ._common        import STEPS_NFACTOR_BY_NAME, DEFAULT_DETAIL_LEVEL


class SetImageDetailsV2:
    TITLE       = "💪TB | Set Image Details"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Fine-tune generation parameters that achieve subtle improvements in the image."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams"   :("GENPARAMS" , {"tooltip": "The generation parameters to be updated."
                                          }),
            "image_shift" :("INT"       , {"tooltip": "Adjustment to the refiner's noise seed. Experiment with different values to create slight image variations without altering the overall composition",
                                           "default": 0, "min": 0, "max": 0xffffffffffffffff
                                          }),
            "cfg_shift"   :("INT"     ,   {"tooltip": "Adjustment to the Classifier-Free Guidance (CFG) scale. Positive values increase prompt adherence; negative values allow the model more freedom. A value of 0 uses the default recommendation from the current style.",
                                           "default": 0, "min": -20, "max": 20,
                                          }),
            "detail_level":(cls.levels(), {"tooltip": "The level of detail in the final image induced by the refiner.",
                                           "default": DEFAULT_DETAIL_LEVEL,
                                          }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "set_image_details"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the tweaks applied. You can use this output to chain to other genparams nodes.",)

    def set_image_details(self,
                          genparams   : GenParams,
                          image_shift : int,
                          cfg_shift   : int,
                          detail_level: str,
                          ):
        genparams = genparams.copy()
        genparams.set_int  ("denoising.refiner.noise_seed"   , image_shift     , as_delta = True   )
        genparams.set_float("denoising.base.cfg"             , cfg_shift * 0.2 , as_delta = True   )
        genparams.set_int  ("denoising.refiner.steps_nfactor", STEPS_NFACTOR_BY_NAME[detail_level] )
        return (genparams,)


    #__ internal functions ________________________________

    @staticmethod
    def levels():
        return list(STEPS_NFACTOR_BY_NAME.keys())

