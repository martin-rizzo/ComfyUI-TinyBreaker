"""
File    : set_image.py
Purpose : Node to set the attributes and seed for the initial noisy image.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 16, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.genparams import GenParams
from ._common import LANDSCAPE_SIZES_BY_ASPECT_RATIO, \
                     SCALES_BY_SIZE,                  \
                     ORIENTATIONS,                    \
                     DEFAULT_ASPECT_RATIO,            \
                     DEFAULT_SIZE,                    \
                     DEFAULT_ORIENTATION,             \
                     normalize_aspect_ratio


class SetImage:
    TITLE       = "💪TB | Set Image"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Sets the attributes and seed for the initial noisy image, packing them into GenParams."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams"  :("GENPARAMS" , {"tooltip": "The generation parameters to be updated.",
                                          }),
            "seed"       :("INT"       , {"tooltip": "The pattern of the random noise to use as starting point for the image generation. (different seed numbers will generate totally different images)",
                                          "default": 1, "min": 1, "max": 0xffffffffffffffff
                                          }),
            "ratio"      :(cls.ratios(), {"tooltip": "The aspect ratio of the image.",
                                          "default": DEFAULT_ASPECT_RATIO
                                          }),
            "orientation":(ORIENTATIONS, {"tooltip": "The orientation of the image. (landscape or portrait)",
                                          "default": DEFAULT_ORIENTATION
                                          }),
            "size"       :(cls.sizes() , {"tooltip": 'The relative size for the image. ("medium" is the size the model was trained on, but "large" is recommended)',
                                          "default": DEFAULT_SIZE
                                          }),
            "batch_size" :("INT"       , {"tooltip": "The number of images to generate in a single batch.",
                                          "default": 1, "min": 1, "max": 4096
                                          }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "set_image_attributes"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters updated with the new image attributes. (you can use this output to chain other genparams nodes)",)

    def set_image_attributes(self,
                             genparams  : GenParams,
                             seed       : int,
                             ratio      : str,
                             orientation: str,
                             size       : str,
                             batch_size : int
                             ):
        genparams = genparams.copy()
        genparams.set_str  ( "image.aspect_ratio"       , normalize_aspect_ratio(ratio) )
        genparams.set_str  ( "image.orientation"        , orientation                   )
        genparams.set_float( "image.scale"              , SCALES_BY_SIZE.get(size, 1.0) )
        genparams.set_int  ( "image.batch_size"         , batch_size                    )
        genparams.set_int  ( "denoising.base.noise_seed", seed                          )
        return (genparams,)


    #__ internal functions ________________________________

    @staticmethod
    def ratios():
        return list(LANDSCAPE_SIZES_BY_ASPECT_RATIO.keys())

    @staticmethod
    def sizes():
        return list(SCALES_BY_SIZE.keys())
