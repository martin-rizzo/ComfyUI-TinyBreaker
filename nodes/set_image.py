"""
File    : set_image.py
Purpose : Node that sets the image attributes, packaging them into the generation parameters.
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

_PREDEFINED_SCALES = {
    "Small"  : 0.82,
    "Medium" : 1.0,
    "Large"  : 1.22,
}

_DEFAULT_RATIO = "1:1 square"
_DEFAULT_SCALE = "Large"


class SetImage:
    TITLE       = "💪TB | Set Image"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Sets the image attributes, packaging them into the generation parameters."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "genparams" : ("GENPARAMS"  , {"tooltip": "The original generation parameters which will be updated."}),

                "landscape" : ("BOOLEAN"    , {"tooltip": "Set the image to landscape orientation. 'True' for landscape, 'False' for portrait.",
                                               "default": True}),
                "ratio"     : (cls._ratios(), {"tooltip": "The aspect ratio of the image.",
                                               "default": _DEFAULT_RATIO}),
                "scale"     : (cls._scales(), {"tooltip": "The scale factor for the image. ('Medium' is the size the model was trained on, but 'Large' is recommended)",
                                               "default": _DEFAULT_SCALE}),
                "batch_size": ("INT"        , {"tooltip": "Number of images to generate per prompt. Adjust this value for batching.",
                                               "default": 1, "min": 1, "max": 4096})
                },
            }

    #__ FUNCTION __________________________________________
    FUNCTION = "set_image_attributes"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated noise seed.",)

    def set_image_attributes(self,
                             genparams : GenParams,
                             landscape : bool,
                             ratio     : str,
                             scale     : str,
                             batch_size: int
                             ):
        genparams = genparams.copy()

        # get the actual aspect ratio from the input string,
        # stripping any additional text after the first space
        ratio = ratio.strip().split(' ',1)[0]

        # if not landscape, swap the width and height in the ratio string
        if not landscape and ':' in ratio:
            _parts = ratio.split(':')
            ratio = f"{_parts[1]}:{_parts[0]}"

        genparams.set("image.scale"       , float(_PREDEFINED_SCALES[scale]))
        genparams.set("image.aspect_ratio", str(ratio)                      )
        genparams.set("image.batch_size"  , int(batch_size)                 )
        return (genparams,)


    #__ internal functions ________________________________

    @staticmethod
    def _ratios():
        return list(_LANDSCAPE_SIZE_BY_ASPECT_RATIO.keys())

    @staticmethod
    def _scales():
        return list(_PREDEFINED_SCALES.keys())

