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

    def load_style(self, style_name):
        """
        Loads a style for image generation.

        Args:
            style_name (str): The name of the style to apply.

        Returns:
            dict: A dictionary containing the loaded style parameters.
        """
        # gparams = {
        #     "base.positive"       : "An iPhone photo. The heroic leader, Master Chief with battle rifle and glowing green visor, at a fruit selling market in Ecuador. In the background there are many surprised people.",
        #     "base.negative"       : "macrophoto, bokeh, out of focus",
        #     "base.steps"          : 12,
        #     "base.cfg"            : 3.4,
        #     "base.sampler_name"   : "uni_pc",
        #     "base.scheduler"      : "simple",
        #     "base.start_at_step"  : 2,

        #     "refiner.positive"     : "The heroic leader, Master Chief with battle rifle and glowing green visor, at a fruit selling market in Ecuador. In the background there are many surprised people.",
        #     "refiner.negative"     : "(worst quality, low quality:1.8)",
        #     "refiner.steps"        : 11,
        #     "refiner.cfg"          : 2.0,
        #     "refiner.sampler_name" : "deis",
        #     "refiner.scheduler"    : "ddim_uniform",
        #     "refiner.start_at_step": 6,
        # }

        gparams = style_dict.get_style_params(style_name)
        print()
        print("##>> gparams:", gparams)
        print()
        return (gparams,)


