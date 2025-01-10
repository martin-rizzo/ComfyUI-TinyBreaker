"""
File    : load_style.py
Desc    : Node that 
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 19, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.style_dict import StyleDict
from .utils.directories import PROJECT_DIR

style_dict = StyleDict.from_file( PROJECT_DIR.get_full_path("STYLES.cfg") )


class LoadStyle:
    TITLE       = "xPixArt | Load Style"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Loads a style for image generation"

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_name": (style_dict.list_styles(), {"tooltip": "The name of the style to use."}),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_style"
    RETURN_TYPES = ("GENPARAMS",)
    RETURN_NAMES = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the style loaded.",)

    def load_style(self, style_name):
        genparams = style_dict.get_style_params(style_name)
        return (genparams,)


