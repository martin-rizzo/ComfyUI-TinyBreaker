"""
File    : set_image_and_cfg.py
Purpose : Node to set image attributes and CFG, packaging them into genparams
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 16, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.gen_params import GenParams
from .common import LANDSCAPE_SIZES_BY_ASPECT_RATIO, \
                    SCALES_BY_NAME,                  \
                    ORIENTATIONS,                    \
                    DEFAULT_ASPECT_RATIO,            \
                    DEFAULT_SCALE_NAME,              \
                    DEFAULT_ORIENTATION,             \
                    normalize_aspect_ratio


class SetImageCFGAndSeed(GenParams):
    TITLE       = "💪TB | Set Image, CFG and Seed"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Sets the image attributes, CFG and seed values, packaging them into the generation parameters."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "genparams"  : ("GENPARAMS" , {"tooltip": "The original generation parameters which will be updated."}),

                "orientation": (ORIENTATIONS, {"tooltip": "The orientation of the image.",
                                               "default": DEFAULT_ORIENTATION
                                               }),
                "ratio"      : (cls.ratios(), {"tooltip": "The aspect ratio of the image.",
                                               "default": DEFAULT_ASPECT_RATIO
                                               }),
                "size"       : (cls.scales(), {"tooltip": "The relative size for the image. ('Medium' is the size the model was trained on, but 'Large' is recommended)",
                                               "default": DEFAULT_SCALE_NAME
                                               }),
                "cfg_tweak"  : ("FLOAT"     , {"tooltip": "An adjustment to the default classifier-free guidance value. Positive values increase prompt adherence; negative values allow more model freedom. A value of 0.0 uses the default setting.",
                                               "default": 0.0, "min": -4.0, "max": 4.0, "step": 0.2, "round": 0.01
                                               }),
                "noise_seed" : ("INT"       , {"tooltip": "The pattern of the random noise to use as starting point for the image generation.",
                                               "default": 1, "min": 1, "max": 0xffffffffffffffff
                                               }),
                },
            }

    #__ FUNCTION __________________________________________
    FUNCTION = "set_image_attributes"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the image attributes and CFG values set.",)

    def set_image_attributes(self,
                             genparams  : GenParams,
                             orientation: str,
                             ratio      : str,
                             size       : str,
                             noise_seed : int,
                             cfg_tweak  : float,
                             ):
        genparams = genparams.copy()
        ratio = normalize_aspect_ratio(ratio, orientation=orientation)
        genparams.set("image.aspect_ratio", str(   ratio                         ))
        genparams.set("image.scale"       , float( SCALES_BY_NAME.get(size, 1.0) ))
        genparams.set("image.batch_size"  , int(   1                             ))
        genparams.set("base.noise_seed"   , int(   noise_seed                    ))
        genparams.add("base.cfg"          , float( cfg_tweak                     ))
        return (genparams,)


    #__ internal functions ________________________________

    @staticmethod
    def ratios():
        return list(LANDSCAPE_SIZES_BY_ASPECT_RATIO.keys())

    @staticmethod
    def scales():
        return list(SCALES_BY_NAME.keys())
