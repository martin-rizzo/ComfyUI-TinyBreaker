"""
File    : set_noisy_image.py
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
from .core.gen_params import GenParams
from .common import LANDSCAPE_SIZES_BY_ASPECT_RATIO, \
                    SCALES_BY_NAME,                  \
                    ORIENTATIONS,                    \
                    DEFAULT_ASPECT_RATIO,            \
                    DEFAULT_SCALE_NAME,              \
                    DEFAULT_ORIENTATION,             \
                    normalize_aspect_ratio


class SetNoisyImage:
    TITLE       = "💪TB | Set Noisy Image"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Sets the attributes and seed for the initial noisy image, packing them into a GenParams."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "genparams"  : ("GENPARAMS" , {"tooltip": "The generation parameters which will be updated.",
                                               }),
                "seed"       : ("INT"       , {"tooltip": "The pattern of the random noise to use as starting point for the image generation. (different seed numbers will generate totally different images)",
                                               "default": 1, "min": 1, "max": 0xffffffffffffffff
                                               }),
                "ratio"      : (cls.ratios(), {"tooltip": "The aspect ratio of the image.",
                                               "default": DEFAULT_ASPECT_RATIO
                                               }),
                "orientation": (ORIENTATIONS, {"tooltip": "The orientation of the image. (landscape or portrait)",
                                               "default": DEFAULT_ORIENTATION
                                               }),
                "size"       : (cls.scales(), {"tooltip": 'The relative size for the image. ("medium" is the size the model was trained on, but "large" is recommended)',
                                               "default": DEFAULT_SCALE_NAME
                                               }),
                "batch_size" : ("INT"       , {"tooltip": "The number of images to generate in a single batch.",
                                               "default": 1, "min": 1, "max": 4096}),
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
                             orientation: str,
                             ratio      : str,
                             size       : str,
                             batch_size : int
                             ):
        genparams = genparams.copy()
        ratio = normalize_aspect_ratio(ratio, orientation=orientation)
        genparams.set("image.aspect_ratio", str(   ratio                         ))
        genparams.set("image.scale"       , float( SCALES_BY_NAME.get(size, 1.0) ))
        genparams.set("image.batch_size"  , int(   batch_size                    ))
        genparams.set("base.noise_seed"   , int(   seed                          ))
        return (genparams,)


    #__ internal functions ________________________________

    @staticmethod
    def ratios():
        return list(LANDSCAPE_SIZES_BY_ASPECT_RATIO.keys())

    @staticmethod
    def scales():
        return list(SCALES_BY_NAME.keys())
