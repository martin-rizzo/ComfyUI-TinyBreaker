"""
File    : unified_prompt_input.py
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

class UnifiedPromptInput:
    TITLE       = "💪TB | Unified Prompt Input"
    CATEGORY    = "TinyBreaker/genparams"
    DESCRIPTION = "Allows to write positive/negative prompts and parameters in a single text input."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams":("GENPARAMS", {"tooltip": "The generation parameters to be updated."
                                       }),
            "text"     :("STRING"   , {"tooltip": "The text input containing positive/negative prompts and parameters.",
                                       "multiline": True, "dynamicPrompts": True
                                       }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "parse_text"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated values. (you can use this output to chain other genparams nodes)",)

    def parse_text(self, genparams: GenParams, text: str):
        template_params = genparams

        prompt   = text
        negative = ""

        # create a new GenParams applying the user's parameters to the template
        data_params = GenParams()
        data_params.set("base.prompt"     , prompt  )
        data_params.set("base.negative"   , negative)
        data_params.set("refiner.prompt"  , prompt  )
        data_params.set("refiner.negative", negative)
        genparams = GenParams.from_template_and_data(template_params, data_params)

        # before returning, store the original user's prompts for future reference
        genparams.set("user.prompt"  , prompt)
        genparams.set("user.negative", negative)
        return (genparams,)
