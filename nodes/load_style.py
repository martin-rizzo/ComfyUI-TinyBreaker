"""
File    : load_style.py
Desc    : Node that 
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 19, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.style_dict import StyleDict
from .utils.directories import PROJECT_DIR

style_dict = StyleDict.from_file( PROJECT_DIR.get_full_path("STYLES.cfg") )



class LoadStyle:
    TITLE       = "xPixArt | Load Style"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Loads a style for image generation"

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_name": (style_dict.list_styles(), {"tooltip": "The name of the style to use."}),
            },
        }

    #-- FUNCTION --------------------------------#
    FUNCTION = "load_style"
    RETURN_TYPES = ("GPARAMS",)
    RETURN_NAMES = ("gparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the style loaded.",)

    def load_style(self, style_name):
        gparams = style_dict.get_style_params(style_name)
        return (gparams,)


