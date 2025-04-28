"""
File    : set_image_v2.py
Purpose : Node to set core image properties that drastically alter the overall format.
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
from ._common import LANDSCAPE_SIZES_BY_ASPECT_RATIO, \
                     SCALES_BY_NAME,                  \
                     ORIENTATIONS,                    \
                     DEFAULT_ASPECT_RATIO,            \
                     DEFAULT_SCALE,                   \
                     DEFAULT_ORIENTATION,             \
                     normalize_aspect_ratio


class SetImageV2:
    TITLE       = "💪TB | Set Image"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Sets core image attributes that drastically alter the overall format and integrate them into the generation parameters."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams"   :("GENPARAMS"  ,{"tooltip": "The generation parameters to be updated.",
                                          }),
            "ratio"       :(cls.ratios() ,{"tooltip": "The aspect ratio of the image.",
                                           "default": DEFAULT_ASPECT_RATIO
                                          }),
            "orientation" :(ORIENTATIONS ,{"tooltip": "The orientation of the image. (landscape or portrait)",
                                           "default": DEFAULT_ORIENTATION
                                          }),
            "size"        :(cls.scales() ,{"tooltip": 'The relative size for the image. ("medium" is the size the model was trained on, but "large" is recommended)',
                                           "default": DEFAULT_SCALE
                                          }),
            "batch_size"  :("INT"        ,{"tooltip": "The number of images to generate in a single batch.",
                                           "default": 1, "min": 1, "max": 4096
                                          }),
            "upscale"     :("BOOLEAN"    ,{"tooltip": "If true, the image will be upscaled to improve its quality.",
                                           "default": False,
                                          }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "set_image"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters updated with the new image attributes. (you can use this output to chain other genparams nodes)",)

    def set_image(self,
                  genparams  : GenParams,
                  ratio      : str,
                  orientation: str,
                  size       : str,
                  batch_size : int,
                  upscale    : bool,
                  ):
        genparams = genparams.copy()
        genparams.set_str  ( "image.aspect_ratio"   , normalize_aspect_ratio(ratio) )
        genparams.set_str  ( "image.orientation"    , orientation                   )
        genparams.set_float( "image.scale"          , SCALES_BY_NAME.get(size, 1.0) )
        genparams.set_bool ( "image.enable_upscaler", upscale                       )
        genparams.set_int  ( "image.batch_size"     , batch_size                    )
        return (genparams,)


    #__ internal functions ________________________________

    @staticmethod
    def ratios():
        return list(LANDSCAPE_SIZES_BY_ASPECT_RATIO.keys())

    @staticmethod
    def scales():
        return list(SCALES_BY_NAME.keys())
