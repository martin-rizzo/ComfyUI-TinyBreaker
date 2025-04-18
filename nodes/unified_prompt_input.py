"""
File    : unified_prompt_input.py
Purpose : Node that unifies positive/negative prompts and parameters into a single text input.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 21, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import time
from .core.genparams             import GenParams
from .core.genparams_from_prompt import genparams_from_prompt


class UnifiedPromptInput:
    TITLE       = "💪TB | Unified Prompt Input"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Allows to write positive/negative prompts and parameters in a single text input."
    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        return self.get_state_hash(*args, **kwargs)


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
    FUNCTION = "parse_prompt"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated values. (you can use this output to chain other genparams nodes)",)

    def parse_prompt(self, text: str, genparams: GenParams) -> GenParams:
        genparams = genparams_from_prompt(text, template=genparams)
        genparams.set_str("user.prompt"  , text)
        genparams.set_str("user.negative", ""  )
        return (genparams,)


    @staticmethod
    def get_state_hash(text: str, genparams: GenParams) -> str:
        """
        Generates a unique hash representing the current state of the node.

        This hash is used to identify and track the state of the node. If the
        input text contains the word "random", a timestamp is appended to the
        hash to ensure uniqueness and force a recalculation, preventing caching.
        Args:
            The same parameters as `parse_prompt`
        """
        if "random" in text:
            return text + str(time.time())
        return text
