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
import re
from .core.genparams import GenParams

# regular expression pattern to match arguments in the format "--key value"
# the pattern supports not having a space between the key and value
_ARGS_PATTERN = r"--([a-z]+)(.+?)(?=\s--|$)"


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

        # 
        style_name = args.pop("style", None)
        if style_name is not None:
            genparams = genparams.copy()
            count = genparams.copy_parameters( target="sampler", source=f"styles.{style_name}", valid_subkeys=["base", "refiner"])

        # build `genparams` from the parsed arguments (using the node input as template)
        genparams_output = GenParams.from_arguments(args, template=genparams)
        genparams_output.set_str("user.prompt"  , text)
        genparams_output.set_str("user.negative", ""  )
        return (genparams_output,)


    #__ internal functions ________________________________

    @staticmethod
    def _parse_args(text: str) -> tuple[str, dict]:
        prompt, _, str_args = text.partition("--")
        matches = re.findall(_ARGS_PATTERN, "--"+str_args)
        args    = {param.strip(): rest.strip() for param, rest in matches}
        args["prompt"] = prompt.strip()
        return args

