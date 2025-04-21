"""
File    : multiline_text.py
Purpose : Node to enter a text that contains multiple lines.
          Necessary because the primitive nodes of some versions of ComfyUI
          do not expand correctly to multiline.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Apr 19, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""


class MutilineText:
    TITLE       = "💪TB | Multiline Text"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Allows entering a text that contains multiple lines. (Note: implemented as a temporary solution because in some versions of ComfyUI, primitive nodes do not expand correctly to multiline.)"


    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "string" :("STRING" ,{"tooltip": "The multiline text you want to enter.",
                                  "multiline"     :  True,
                                  "dynamicPrompts": False,
                                  "default"       :    "",
                                 }),
            }
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "output_string"
    RETURN_TYPES    = ("STRING",)
    #RETURN_NAMES   = ("string",)
    OUTPUT_TOOLTIPS = ("The string you entered.",)


    def output_string(self, string):
        return (string, )

