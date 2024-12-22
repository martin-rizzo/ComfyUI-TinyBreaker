"""
File    : unified_prompt_editor.py
Purpose : Node that unifies positive/negative prompts and parameters into a single text input.
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

class UnifiedPromptEditor:
    TITLE       = "xPixArt | Unified Prompt Editor"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Allows to write positive/negative prompts and parameters in a single text input."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gparams": ("GPARAMS", {"tooltip": "The original generation parameters which will be updated."}),
                "text"   : ("STRING" , {"tooltip": "The text input containing positive/negative prompts and parameters.", "multiline": True, "dynamicPrompts": True}),
            },
        }

    #-- FUNCTION --------------------------------#
    FUNCTION = "parse_text"
    RETURN_TYPES    = ("GPARAMS",)
    RETURN_NAMES    = ("gparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated values.",)

    def parse_text(self, gparams, text):
        template_params = gparams
        data_params = GParams()
        data_params.set("base.prompt"   , text)
        data_params.set("refiner.prompt", text)

        gparams = GParams.from_template_and_data(template_params, data_params)
        return (gparams,)
