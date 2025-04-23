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

    #__ PARAMETERS ________________________________________
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return cls.get_state_hash(*args, **kwargs)

    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams" :("GENPARAMS" ,{"tooltip": "The generation parameters to be updated."
                                       }),
            "prompt"    :("STRING"    ,{"tooltip": "The text input containing positive/negative prompts and parameters.",
                                        "multiline"     : True,
                                        "dynamicPrompts": True
                                       }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "parse_prompt"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the updated values. (you can use this output to chain other genparams nodes)",)

    def parse_prompt(self, prompt: str, genparams: GenParams) -> GenParams:
        prompt    = self._filter_prompt(prompt, remove_comments=False)
        genparams = genparams_from_prompt(prompt, template=genparams)
        genparams.set_str("user.prompt"  , prompt)
        genparams.set_str("user.negative", ""    )
        return (genparams,)


    @staticmethod
    def get_state_hash(prompt: str, genparams: GenParams) -> str:
        """
        Generates a unique hash representing the current state of the node.
        This hash is used by ComfyUI to track changes for its caching mechanism.

        In this case, if the input prompt contains the word "random", a timestamp
        is appended to the hash to ensure uniqueness and force a recalculation,
        preventing caching.

        Args:
            The same parameters as `parse_prompt`
        """
        if "random" in prompt:
            return prompt + str(time.time())
        return prompt



    #__ internal functions ________________________________

    @staticmethod
    def _filter_prompt(prompt: str, /,*, remove_comments=False) -> str:
        """Filter the input prompt by removing leading/trailing whitespaces and comments."""
        lines = prompt.strip().splitlines()
        lines = [line.strip() for line in lines]
        if remove_comments:
            lines = [line for line in lines if not line.startswith("#")]
        return "\n".join(lines)
