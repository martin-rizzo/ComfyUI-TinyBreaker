"""
File    : unpack_boolean.py
Desc    : Node that unpack a boolean value from the `GenParams` line.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Apr 18, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.genparams import GenParams

class UnpackBoolean:
    TITLE = "💪TB | Unpack Boolean"
    CATEGORY = "TinyBreaker"
    DESCRIPTION = "Unpacks a boolean value from GenParams."


    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams"  :("GENPARAMS" ,{"tooltip": "The generation parameters containing the boolean to be unpacked.",
                                        }),
            "param_name" :("STRING"    ,{"tooltip": "The name of the boolean parameter to be unpacked. If not found, it will return False.",
                                         "default": ""
                                        }),
            }
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "unpack_boolean_value"
    RETURN_TYPES = ("BOOLEAN",)
    OUTPUT_TOOLTIPS = ("The boolean value extracted from the `GenParams` line or False if not found.",)

    def unpack_boolean_value(self,
                             genparams : GenParams,
                             param_name: str
                             ):
        value = genparams.get_bool(param_name, False)
        return (value, )


