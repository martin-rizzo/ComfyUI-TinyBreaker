"""
File    : unpack_float.py
Desc    : Node that unpack a float value from a GenParams object.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Apr 25, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.genparams import GenParams

class UnpackFloat:
    TITLE = "💪TB | Unpack Float"
    CATEGORY = "TinyBreaker"
    DESCRIPTION = "Unpacks a float value from GenParams."


    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams"    :("GENPARAMS" ,{"tooltip": "The generation parameters containing the float value to be unpacked.",
                                          }),
            "param_name"   :("STRING"    ,{"tooltip": "The name of the float parameter to be unpacked.",
                                           "default": ""
                                          }),
            "default_value":("FLOAT"     ,{"tooltip": "The default value to be used when the parameter is not found.",
                                           "default": 0.0, "step": 0.1
                                          }),
            }
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "unpack_float_value"
    RETURN_TYPES = ("FLOAT", )
    OUTPUT_TOOLTIPS = ("The unpacked float value extracted from the `GenParams` input or the specified default.",)

    def unpack_float_value(self,
                           genparams    : GenParams,
                           param_name   : str,
                           default_value: float
                           ):
        value = genparams.get_float(param_name, default_value)
        return (value, )

