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
import re
from .functions.genparams import GenParams
from ._common             import genparams_from_arguments


class UnifiedPromptInput:
    TITLE       = "💪TB | Unified Prompt Input"
    CATEGORY    = "TinyBreaker"
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

    def parse_text(self, genparams: GenParams, text: str) -> GenParams:

        # parse the arguments entered by the user
        args = self._parse_args(text)

        # extract the style name from the arguments and apply it to `genparams`
        style_name = args.pop("style", None)
        if style_name is not None:
            genparams = genparams.copy()
            # copy parameters from "styles.<style_name>.*" -> "denoising.*"
            count = genparams.copy_parameters( target="denoising", source=f"styles.{style_name}", valid_subkeys=["base", "refiner"])
            if count > 0:
                genparams.set_str("user.style", style_name)

        # build `genparams` from the parsed arguments (using the node input as template)
        genparams_output = genparams_from_arguments(args, template=genparams)
        genparams_output.set_str("user.prompt"  , text)
        genparams_output.set_str("user.negative", ""  )
        return (genparams_output,)


    #__ internal functions ________________________________

    @staticmethod
    def _parse_args(text: str) -> dict[str, str]:
        """Parse the text input into a dictionary of arguments."""
        prompt = ""

        # split the text into arguments by "--"
        arg_list = text.split("--")

        # the first element is the positive prompt, the rest are arguments
        if len(arg_list) > 0:
            prompt = arg_list.pop(0).strip()

        # parse the arguments into a dictionary
        # each argument is of the form "key value" or just "key" (for boolean values)
        arg_dict = {}
        for arg in arg_list:
            key, _, value = arg.partition(' ')
            key   = key.lower().strip()
            value = value.strip()
            if key:
                arg_dict[key] = value

        arg_dict["prompt"] = prompt
        return arg_dict

