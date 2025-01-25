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
        genparams_input = genparams

        # parse the arguments entered by the user
        args = self._parse_args(text)

        style = args.pop("style", None)
        # if style is not None:
        #     genparams_input = self._apply_style(style, genparams_input)

        # build `genparams` from the parsed arguments (using the node input as template)
        genparams_input["refiner.seed"] = 100
        genparams_output = GenParams.from_arguments(args, template=genparams_input)
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


    # @staticmethod
    # def _split_prompt_and_args(text: str) -> tuple[str, list]:
    #     """Parses the text input and returns a tuple with the prompt and a list of arguments."""
    #     prompt, _, args_text = text.partition("--")
    #     args = args_text.replace('\n', " ").replace('\r', " ").split(" --")
    #     args = [a.strip() for a in args]
    #     return prompt.strip(), ["--"+a for a in args if len(a) > 0]


    # @staticmethod
    # def _get_style(args: list) -> str:
    #     for arg in args:
    #         if arg.startswith("--style "):
    #             return arg.split(' ',1)[1].upper()
    #     return ""

    # @staticmethod
    # def _apply_args_to_genparams(genparams: GenParams,
    #                              args: list,
    #                              *,# keyword-only args #
    #                              prompt: str = ""
    #                              ):
    #     genparams.set_str("base.prompt"     , prompt  )
    #     genparams.set_str("refiner.prompt"  , prompt  )

    #     for arg in args:
    #         if arg.startswith("--style "): continue
    #         if arg.startswith("--no "):
    #             _negative = arg.split(' ',1)[1].strip()
    #             genparams.set_str("base.negative"   , _negative)
    #             genparams.set_str("refiner.negative", _negative)

