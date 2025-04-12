"""
File    : genparams_ex.py
Purpose : `GenParams` extension that supports parsing arguments from the prompt.
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
import re
from .genparams import GenParams
from .._common import normalize_aspect_ratio, NFACTORS_BY_DETAIL_LEVEL, SCALES_BY_NAME


class GenParamsEx(GenParams):

    @classmethod
    def from_prompt(cls, prompt: str, /,*, template: GenParams = None) -> GenParams:

        # parse the arguments from the prompt
        args       = _parse_args(prompt)
        style_name = args.pop("style", None)

        # if the user specifies a style in the prompt, overwrite the template denoising values
        # ( copy template parameters from `styles.{style_name}.*` to `denoising.*` )
        if style_name:
            template = template.copy()
            count = template.copy_parameters( target="denoising", source=f"styles.{style_name}", valid_subkeys=["base", "refiner", "upscaler"])
            if count > 0:
                template.set_str("user.style", style_name)

        return cls.from_prompt_args( args, template=template )


    @classmethod
    def from_prompt_args(cls, args: dict, /,*, template: GenParams = None) -> GenParams:
        """
        Create a GenParams object from the given arguments.
        Args:
            args          (dict): A dictionary of argument names and their values.
            template (GenParams): A template GenParams object to be used as a base for the new GenParams object.
        """
        genparams = cls(template) if template else cls()

        # base/refiner prefixes
        BASE = "denoising.base."
        RE__ = "denoising.refiner."

        # --prompt <text>
        # "base.prompt", << "refiner.prompt" >>
        prompt = _pop_str_value(args, "prompt") or ""
        genparams.set_str(f"{BASE}prompt", prompt, use_template=True)

        # --no, --negative <text>
        # "base.negative", "refiner.negative"
        value = _pop_str_value(args, "no", "negative") or ""
        genparams.set_str(f"{BASE}negative", value, use_template=True)
        genparams.set_str(f"{RE__}negative", value, use_template=True)

        # --refine <text>
        # << "refiner.prompt" >>
        value = _pop_str_value(args, "refine") or ""
        if value and not value.endswith('.'):
            value += '.'
        if value.startswith('!'):
            genparams.set_str(f"{RE__}prompt", f"{value[1:]}", use_template=True)
        else:
            genparams.set_str(f"{RE__}prompt", f"{value}{prompt}", use_template=True)

        # --c, --cfg-shift <int>
        # "base.cfg"
        value, as_delta = _pop_int_value(args, "c", "cfg-shift")
        if value is not None:
            genparams.set_float(f"{BASE}cfg", value * 0.2, as_delta=True)

        # --i, --image-shift <int>
        # "refiner.noise_seed"
        value, as_delta = _pop_int_value(args, "i", "image-shift")
        if value is not None:
            genparams.set_int(f"{RE__}noise_seed", value, as_delta=False)

        # --u, --upscale <float>
        # "image.upscale_factor"
        print("##>>>>============================")
        value, as_delta =  _pop_floatX_or_str(args, "u", "upscale")
        if isinstance(value, float):
            genparams.set_float(f"image.upscale_factor", value, as_delta=False)

        # --d, --detail-level <level>
        # "refiner.steps_nfactor"
        value = _pop_str_value(args, "d", "detail-level")
        if value in NFACTORS_BY_DETAIL_LEVEL:
            genparams.set_int(f"{RE__}steps_nfactor", NFACTORS_BY_DETAIL_LEVEL[value] )

        # --s, --seed <int>
        # "base.noise_seed"
        value, as_delta = _pop_int_value(args, "s", "seed")
        if value is not None:
            genparams.set_int(f"{BASE}noise_seed", value, as_delta=False)

        # --a, --aspect <width:height>
        # "image.aspect_ratio"
        value = _pop_str_value(args, "a", "aspect")
        if value:
            ratio = normalize_aspect_ratio(value)
            genparams.set_str(f"image.aspect_ratio", ratio)

        # --landscape / --portrait
        # "image.orientation"
        value = _pop_option(args, "landscape", "portrait")
        if value:
            genparams.set_str(f"image.orientation", value)

        # --small / --medium / --large
        # "image.scale"
        value = _pop_option(args, "small", "medium", "large")
        if value:
            genparams.set_float(f"image.scale", SCALES_BY_NAME[value])

        # --b, --batch <int>
        # "image.batch_size"
        value, as_delta = _pop_int_value(args, "b", "batch")
        if value is not None:
            genparams.set_int("image.batch_size", value, as_delta=False)

        #-- DEPRECATED ------------------------------------------------#
        # --v, --variant <int>
        # "refiner.noise_seed"
        value, as_delta = _pop_int_value(args, "variant")
        if value is not None:
            genparams.set_int(f"{RE__}noise_seed", value, as_delta=False)
        # --c, --cfg-adjust <float>
        # "base.cfg"
        value, as_delta = _pop_float_value(args, "cfg-adjust")
        if value is not None:
            genparams.set_float(f"{BASE}cfg", value, as_delta=True)
        #--------------------------------------------------------------#


        return genparams


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



def _pop_str_value(args: dict, *keys) -> str:
    for key in keys:
        value = args.pop(key, None)
        if value is not None:
            return str(value)
    return None

def _pop_option(args: dict, *keys) -> str | None:
    option = None
    for key in keys:
        value = args.pop(key, None)
        if value is not None:
            option = key
    return option

def _pop_int_value(args: dict, *keys) -> tuple[int | None, bool]:
    str_value = _pop_str_value(args, *keys)
    if str_value is None:
        return None, False
    value, as_delta = _parse_int(str_value)
    if value is None:
        return None, False
    return value, as_delta

def _pop_float_value(args: dict, *keys) -> tuple[float | None, bool]:
    str_value = _pop_str_value(args, *keys)
    if str_value is None:
        return None, False
    value, as_delta = _parse_float(str_value)
    if value is None:
        return None, False
    return value, as_delta

def _pop_floatX_or_str(args: dict, *keys) -> tuple[float | str | None, bool]:
    str_value = _pop_str_value(args, *keys)
    if str_value is None:
        return None, False
    str_value = str_value.lower()
    value = str_value.removesuffix('x')
    try:
        value = float(value)
        return value, False
    except ValueError:
        return str_value, False


def _parse_float(str_value: str,
                 *,
                 step     : float =  0.1,
                 min_value: float = -float('inf'),
                 max_value: float =  float('inf'),
                 ) -> tuple[float | None, bool]:
    """
    Parses a float value from a string with the format <number><brackets>.

    The number part is a standard float representation.
    The brackets part consists of '[' and ']' characters which increment
    and decrement the number by the given step, respectively.

    Args:
        str_value: The string to parse.
        step     : The increment/decrement value for each bracket.
        min_value: The minimum allowed value.
        max_value: The maximum allowed value.

    Returns:
        A tuple containing:
        - The parsed float value, or None if parsing fails.
        - A boolean indicating if the parsed value is a delta value.
    """
    if not str_value:
        return None, False

    # extract the number and the rest of the string
    match = re.match(r"^([-+]?\d*\.\d+?|)(.*)$", str_value.strip())
    if not match:
        return None, False

    number_str, remaining = match.groups()
    number   = float(number_str) if number_str else 0.0
    as_delta = not bool(number_str)

    # adjust the number based in the number of brackets in the remaining string
    for char in remaining.strip():
        if   char == "[":  number += step
        elif char == "]":  number -= step
        else:
            return None, False

    # if everything is OK,
    # return the number and a flag indicating if it's an offset
    number = max(min_value, min(number, max_value))
    return number, as_delta


def _parse_int(str_value: str,
               *,
               step       : int  =    1,
               min_value  : int  = None,
               max_value  : int  = None,
               should_wrap: bool = False,
               ) -> tuple[int | None, bool]:
    """
    Parses an integer value from a string with the format <number><brackets>.

    The number part is a standard integer representation.
    The brackets part consists of '[' and ']' characters which increment
    and decrement the number by the given step, respectively.

    Args:
        str_value  : The string to parse.
        step       : The increment/decrement value for each bracket.
        min_value  : The minimum allowed value.
        max_value  : The maximum allowed value.
        should_wrap: Whether to wrap around when reaching the minimum or maximum value.

    Returns:
        A tuple containing:
        - The parsed integer value, or None if parsing fails.
        - A boolean indicating if the parsed value is a delta value.
    """
    if not str_value:
        return None, False
    if min_value is None or max_value is None:
        should_wrap = False

    # extract the number and the rest of the string
    match = re.match(r"^([-+]?\d*)(.*)$", str_value.strip())
    if not match:
        return None, False

    number_str, remaining = match.groups()
    number   = int(number_str) if number_str else 0
    as_delta = not bool(number_str)

    # adjust the number based in the number of brackets in the remaining string
    for char in remaining.strip():
        if   char == "[":  number += int(step)
        elif char == "]":  number -= int(step)
        else:
            return None, False

    # ajust the value to be within min/max limits
    # if `should_wrap` is True then wrap around min/max limits
    if min_value is not None and number < min_value:
        number = (max_value + (number-min_value) + 1) if should_wrap else min_value
    if max_value is not None and number > max_value:
        number = (min_value + (number-max_value) - 1) if should_wrap else max_value

    # if everything is OK,
    # return the number and a flag indicating if it's an offset
    return number, as_delta

