"""
File    : save_anything.py
Purpose : Node for saving anything to disk.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 14, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
_INPUT_TYPES = "*" # "LATENT,IMAGE,STRING,CONDITIONING"

class SaveAnything:
    TITLE       = "💪TB | Save Anything"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Save anything to disk."
    OUTPUT_NODE = True

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "anything"       : (_INPUT_TYPES, {"tooltip": "The input to save to disk. This may be an image, a latent, or any other type of data that can be saved to disk."
                                              }),
            "filename_prefix": ("STRING"    , {"tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd%.",
                                               "default": "SaveAnything"
                                              }),

            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "save"
    RETURN_TYPES = ()

    def save(self,
             anything,
             filename_prefix: str,
             ):

        # if user didn't specify a file name, don't do anything
        if not filename_prefix:
            return {}

        # TODO: implement detecting the type of input and saving it to disk.
        print(anything)
        return {}

