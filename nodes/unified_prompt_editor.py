"""
File    : unified_prompt_editor.py
Purpose : Node that unifies positive/negative prompts and parameters into a single text input.
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

class UnifiedPromptEditor:
    TITLE       = "💪TB | Unified Prompt Editor"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Allows to write positive/negative prompts and parameters in a single text input."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "genparams": ("GENPARAMS", {"tooltip": "The original generation parameters which will be updated."}),
                "text"     : ("STRING"   , {"tooltip": "The text input containing positive/negative prompts and parameters.", "multiline": True, "dynamicPrompts": True}),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "parse_text"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated values.",)

    def parse_text(self, genparams, text):
        template_params = genparams
        data_params = GenParams()
        data_params.set("base.prompt"   , text)
        data_params.set("refiner.prompt", text)

        genparams = GenParams.from_template_and_data(template_params, data_params)
        return (genparams,)
