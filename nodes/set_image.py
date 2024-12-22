"""
File    : set_image.py
Purpose : Node that sets the image attributes, packaging them into the generation parameters.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 21, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.gparams import GParams

_LANDSCAPE_SIZE_BY_ASPECT_RATIO = {
    "1:1 square"      : (1024.0, 1024.0),
    "4:3 tv"          : (1182.4,  886.8),
    "48:35 (35 mm)"   : (1199.2,  874.4),
    "71:50 ~IMAX"     : (1220.2,  859.3),
    "3:2 photo"       : (1254.1,  836.1),
    "16:10 wide"      : (1295.3,  809.5),
    "16:9 hdtv"       : (1365.3,  768.0),
    "2:1 mobile"      : (1448.2,  724.0),
    "21:9 ultrawide"  : (1564.2,  670.4),
    "12:5 anamorphic" : (1586.4,  661.0),
    "70:27 cinerama"  : (1648.8,  636.0),
    "32:9 s.ultrawide": (1930.9,  543.0),
}
_DEFAULT_RATIO = "1:1 square"



class SetImage:
    TITLE       = "xPixArt | Set Image"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Sets the image attributes, packaging them into the generation parameters."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gparams"   : ("GPARAMS"    , {"tooltip": "The original generation parameters which will be updated."}),

                "landscape" : ("BOOLEAN"    , {"tooltip": "Set the image to landscape orientation. True for landscape, False for portrait.",
                                               "default": True}),
                "ratio"     : (cls._ratios(), {"tooltip": "Select the aspect ratio for the image.",
                                               "default": _DEFAULT_RATIO}),
                "scale"     : ("FLOAT"      , {"tooltip": "Scale factor for the image size. Adjust to increase or decrease resolution, a value of 1.0 is the default size for the model.",
                                               "default": 1.0, "min": 0.90, "max": 1.24, "step": 0.02}),
                "batch_size": ("INT"        , {"tooltip": "Number of images to generate per prompt. Adjust this value for batching.",
                                               "default": 1, "min": 1, "max": 4096})
                },
            }

    #-- FUNCTION --------------------------------#
    FUNCTION = "set_image_attributes"
    RETURN_TYPES    = ("GPARAMS",)
    RETURN_NAMES    = ("gparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated noise seed.",)

    def set_image_attributes(self,
                             gparams   : GParams,
                             landscape : bool,
                             ratio     : str,
                             scale     : float,
                             batch_size: int
                             ):
        gparams = gparams.copy()

        # get the actual aspect ratio from the input string,
        # stripping any additional text after the first space
        ratio = ratio.strip().split(' ',1)[0]

        # if not landscape, swap the width and height in the ratio string
        if not landscape and ':' in ratio:
            _parts = ratio.split(':')
            ratio = f"{_parts[1]}:{_parts[0]}"

        gparams.set("image.scale"       , float(scale)   )
        gparams.set("image.aspect_ratio", str(ratio)     )
        gparams.set("image.batch_size"  , int(batch_size))
        return (gparams,)



    @classmethod
    def _ratios(self):
        return list(_LANDSCAPE_SIZE_BY_ASPECT_RATIO.keys())