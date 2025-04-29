"""
File    : genparams_from_prompt.py
Purpose : Functions to create `GenParams` from a prompt string.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Apr 11, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import random
from .genparams import GenParams
from .._common  import normalize_aspect_ratio,   \
                       NFACTORS_BY_DETAIL_LEVEL, \
                       SCALES_BY_NAME,           \
                       UPSCALE_NOISE_BY_LEVEL


def genparams_from_prompt(prompt: str, /,*, template: GenParams = None) -> GenParams:
    """
    Creates a `GenParams` object from a given prompt string.

    The prompt is parsed for arguments in the format `--<name> [value]`.
    For example: "A dog on the beach --aspect 3:2 --portrait --seed 1 --upscale on"

    Args:
        prompt         (str): The prompt string to parse arguments from.
        template (GenParams): An existing `GenParams` object to use as a base.

    Returns:
        A `GenParams` object configured with the parsed arguments.
    """
    # parse the arguments from the prompt
    prompt_args = _parse_args_from_prompt(prompt)
    return genparams_from_prompt_args( prompt_args, template=template )


def genparams_from_prompt_args(args: dict, /,*, template: GenParams = None) -> GenParams:
    """
    Create a `GenParams` object from the given arguments.
    Args:
        args          (dict): A dictionary of argument names and their values.
        template (GenParams): A template GenParams object to be used as a base for the new GenParams object.
    """
    # base/refiner prefixes constants
    BASE = "denoising.base."
    RE__ = "denoising.refiner."
    UP__ = "denoising.upscaler."

    # start with genparams being a copy of template
    genparams = GenParams(template) if template else GenParams()

    # if the user specifies a style in the prompt,
    # apply the style (overwritting all denoising values)
    style_name = _pop_str(args, "style")
    if style_name:
        apply_style(genparams, style_name)

    # --prompt <text>
    # "base.prompt", << "refiner.prompt" >>
    prompt = _pop_str(args, "prompt") or ""
    genparams.set_str(f"{BASE}prompt", prompt, use_template=True)

    # --no, --negative <text>
    # "base.negative", "refiner.negative"
    value = _pop_str(args, "no", "negative") or ""
    genparams.set_str(f"{BASE}negative", value, use_template=True)
    genparams.set_str(f"{RE__}negative", value, use_template=True)

    # --refine <text>
    # << "refiner.prompt" >>
    value = _pop_str(args, "refine") or ""
    if value and not value.endswith('.'):
        value += '.'
    if value.startswith('!'):
        genparams.set_str(f"{RE__}prompt", f"{value[1:]}", use_template=True)
    else:
        genparams.set_str(f"{RE__}prompt", f"{value}{prompt}", use_template=True)

    # --c, --cfg-shift <int>
    # "base.cfg"
    value = _pop_int_or_str(args, "c", "cfg-shift")
    if isinstance(value, int):
        genparams.set_float(f"{BASE}cfg", value * 0.2, as_delta=True)

    # --i, --image-shift <int>
    # "refiner.noise_seed"
    value = _pop_int_or_str(args, "i", "image-shift")
    if isinstance(value, int):
        genparams.set_int(f"{RE__}noise_seed", value, as_delta=True) # as_delta=False?

    # --u, --upscale <bool>
    # "image.enable_upscaler"
    value = _pop_bool_or_str(args, "u", "upscale")
    if isinstance(value, bool):
        genparams.set_bool("image.enable_upscaler", value)

    # --upscale-noise <float>
    # "upscaler.extra_noise"
    value = _pop_float_or_str(args, "upscale-noise")
    if isinstance(value, float):
        genparams.set_float(f"{UP__}extra_noise", value, as_delta=True)
    elif value in UPSCALE_NOISE_BY_LEVEL:
        genparams.set_float(f"{UP__}extra_noise", UPSCALE_NOISE_BY_LEVEL[value], as_delta=True)

    # --d, --detail-level <level>
    # "refiner.steps_nfactor"
    value = _pop_str(args, "d", "detail-level")
    if value in NFACTORS_BY_DETAIL_LEVEL:
        genparams.set_int(f"{RE__}steps_nfactor", NFACTORS_BY_DETAIL_LEVEL[value] )

    # --s, --seed <int> | "random"
    # "base.noise_seed"
    value = _pop_int_or_str(args, "s", "seed")
    if isinstance(value, int):
        genparams.set_int(f"{BASE}noise_seed", value)
    elif isinstance(value, str) and value.lower() == "random":
        genparams.set_int(f"{BASE}noise_seed", random.randint(0, 1<<24))

    # --a, --aspect <width:height>
    # "image.aspect_ratio"
    value = _pop_str(args, "a", "aspect")
    if isinstance(value, str):
        ratio = normalize_aspect_ratio(value)
        genparams.set_str(f"image.aspect_ratio", ratio)

    # --landscape / --portrait
    # "image.orientation"
    value = _pop_option(args, "landscape", "portrait")
    if isinstance(value, str):
        genparams.set_str(f"image.orientation", value)

    # --small / --medium / --large
    # "image.scale"
    value = _pop_option(args, "small", "medium", "large")
    if isinstance(value, str):
        genparams.set_float(f"image.scale", SCALES_BY_NAME[value])

    # --b, --batch <int>
    # "image.batch_size"
    value = _pop_int_or_str(args, "b", "batch")
    if isinstance(value, int):
        genparams.set_int("image.batch_size", value, as_delta=False)

    #-- DEPRECATED ------------------------------------------------#
    # --v, --variant <int>
    # "refiner.noise_seed"
    value = _pop_int_or_str(args, "variant")
    if isinstance(value, int):
        genparams.set_int(f"{RE__}noise_seed", value, as_delta=False)
    # --c, --cfg-adjust <float>
    # "base.cfg"
    value = _pop_float_or_str(args, "cfg-adjust")
    if isinstance(value, float):
        genparams.set_float(f"{BASE}cfg", value, as_delta=True)
    #--------------------------------------------------------------#

    return genparams


def split_prompt_and_args(text: str) -> tuple[str,list]:
    """
    Splits a string into a prompt and a list of arguments.

    The result of this function can be reconstructed
    using `join_prompt_and_args(prompt, args)`.

    Args:
        text (str): The input string containing the prompt and arguments.

    Returns:
        A tuple containing the prompt (string) and a list of arguments (strings).
    """
    args   = text.split("--")
    prompt = args.pop(0) if len(args)>0 else ""
    return prompt, args


def join_prompt_and_args(prompt: str, args: list[str]) -> str:
    """
    Joins a prompt and a list of arguments into a single string.

    This function is the inverse of `split_prompt_and_args(text)`.
    It takes the prompt and the list of arguments and can be used
    to reconstruct the original text after it has been split.

    Args:
        prompt (str): The prompt.
        args  (list): A list of strings containing the arguments.

    Returns:
        The concatenated string containing the prompt and arguments.
    """
    return prompt + "--" + "--".join(args) if args else prompt


def apply_style(genparams: GenParams, style_name:str) -> None:
    """
    Applies a predefined style to the GenParams object.

    The specified style must be defined within the GenParams under the key "styles.<style_name>".
    This function copies the 'base', 'refiner', and 'upscaler' subkeys from the style definition
    to the root level of the GenParams under the key "denoising".

    Args:
        genparams (GenParams): The GenParams object to modify.
        style_name      (str): The name of the style to apply.

    Returns:
        None, the GenParams object is modified in place.
    """
    count = genparams.copy_parameters( target="denoising", source=f"styles.{style_name}", valid_subkeys=["base", "refiner", "upscaler"])
    if count > 0:
        genparams.set_str("user.style", style_name)



#============================ INTERNAL HELPERS =============================#

def _parse_args_from_prompt(text: str) -> dict[str, str]:
    """Parse the text input into a dictionary of arguments."""

    # split the text into arguments by "--"
    # the first element is the positive prompt, the rest are arguments
    prompt, arg_list = split_prompt_and_args(text)
    prompt = prompt.strip()

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

def _pop_str(args: dict, *keys) -> str:
    """Pop a string value from the dictionary of arguments."""
    for key in keys:
        value = args.pop(key, None)
        if value is not None:
            return str(value)
    return None

def _pop_option(args: dict, *keys) -> str | None:
    """Pop an option from the dictionary of arguments."""
    option = None
    for key in keys:
        if args.pop(key, None) is not None:
            option = key
    return option

def _pop_float_or_str(args: dict, *keys) -> float | str | None:
    """Pop a float or string value from the dictionary of arguments."""
    str_value = _pop_str(args, *keys)
    if str_value is None:
        return None
    try:
        value = float(str_value)
        return value
    except ValueError:
        return str_value

def _pop_int_or_str(args: dict, *keys) -> int | str | None:
    """Pop an integer or string value from the dictionary of arguments."""
    str_value = _pop_str(args, *keys)
    if str_value is None:
        return None
    try:
        value = int(str_value)
        return value
    except ValueError:
        return str_value

def _pop_bool_or_str(args: dict, *keys) -> bool | str | None:
    """Pop a boolean or string value from the dictionary of arguments."""
    str_value = _pop_str(args, *keys)
    if str_value is None:
        return None
    str_value = str_value.lower()
    if str_value in ["true", "yes", "on"]:
        return True
    elif str_value in ["false", "no", "off"]:
        return False
    else:
        return str_value



