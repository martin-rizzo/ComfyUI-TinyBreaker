"""
File    : common.py
Purpose : Common functions and constants that can be used by any node
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 16, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import re
from .core.genparams import GenParams


#----------------------- STRING MANIPULATION --------------------------#

def ireplace(text: str, old: str, new: str, count: int = -1) -> str:
    """
    Replaces all occurrences of `old` in `text` with `new`, case-insensitive.
    If count is given, only the first `count` occurrences are replaced.
    """
    lower_text , lower_old = text.lower(), old.lower()
    index_start, index_end = 0, lower_text.find(lower_old, 0)
    if index_end == -1 or len(lower_text) != len(text):
        return text

    output = ""
    lower_old_length = len(lower_old)
    while index_end != -1 and count != 0:
        output += text[index_start:index_end] + new
        index_start = index_end + lower_old_length
        index_end   = lower_text.find(lower_old, index_start)
        if count > 0:
            count -= 1
    return output + text[index_start:]


#-------------------------- IMAGE SCALES & RATIOS --------------------------#

LANDSCAPE_SIZES_BY_ASPECT_RATIO = {
    "1:1  (square)"      : (1024.0, 1024.0),
    "4:3  (tv)"          : (1182.4,  886.8),
    "3:2  (photo)"       : (1254.1,  836.1),
    "16:10  (wide)"      : (1295.3,  809.5),
    "16:9  (hdtv)"       : (1365.3,  768.0),
    "2:1  (mobile)"      : (1448.2,  724.0),
    "21:9  (ultrawide)"  : (1564.2,  670.4),
    "12:5  (anamorphic)" : (1586.4,  661.0),
    "70:27  (cinerama)"  : (1648.8,  636.0),
    "32:9  (s.ultrawide)": (1930.9,  543.0),
    # "48:35  (35 mm)"     : (1199.2,  874.4),
    # "71:50  (~imax)"     : (1220.2,  859.3),
}
SCALES_BY_NAME = {
    "small"  : 0.82,
    "medium" : 1.0,
    "large"  : 1.22,
}
ORIENTATIONS = [
    "landscape",
    "portrait"
]
NFACTORS_BY_DETAIL_LEVEL = {
    "none"     : -10000,
    "minimal"  : -2,
    "low"      : -1,
    "normal"   :  0,
    "high"     : +1,
    "veryhigh" : +2,
    "maximum"  : +3,
}

DEFAULT_ASPECT_RATIO = "1:1  (square)"
DEFAULT_SIZE         = "large"
DEFAULT_ORIENTATION  = "landscape"
DEFAULT_DETAIL_LEVEL = "normal"


def normalize_aspect_ratio(aspect_ratio: str, *, force_orientation: str = None) -> str:
    """
    Normalize aspect ratio to a string of the form 'width:height'.
    Args:
      aspect_ratio      (str): The landscape aspect ratio to normalize.
      force_orientation (str): The orientation to force the aspect ratio to. ("landscape" or "portrait")
    """
    width, height = 1, 1

    # handle aspect ratio as a string of the form 'width:height <some text>'
    if isinstance(aspect_ratio, str) and ':' in aspect_ratio:
        ratio    , _, _          = aspect_ratio.strip().partition(' ')
        width_str, _, height_str = ratio.partition(':')
        if width_str.isdigit() and height_str.isdigit():
            width, height  = int(width_str), int(height_str)

    # handle aspect ratio as a tuple/list of two integers
    elif (isinstance(aspect_ratio, tuple) or isinstance(aspect_ratio, list)) and len(aspect_ratio) == 2:
        if isinstance(aspect_ratio[0], int) and isinstance(aspect_ratio[1], int):
            width, height = aspect_ratio[0], aspect_ratio[1]

    # don't allow infinite and negative aspect ratios
    if width < 1 or height <= 1:
        width, height = 1, 1

    # don't allow ratios greater than 6 between width/height
    width  = min(width , 6 * height)
    height = min(height, 6 * width )

    # if orientation is forced, make sure it is respected
    if force_orientation:
        orientation = force_orientation.lower()
        if (orientation == "landscape" and width < height) or (orientation == "portrait" and width > height):
            width, height = height, width

    return f"{width}:{height}"


#--------------------------- GENPARAMS FROM ARGS ---------------------------#

def genparams_from_arguments(args: dict,
                             *,# keyword-only arguments #
                             template: GenParams = None
                             ) -> GenParams:
    """
    Create a GenParams object from the given arguments.
    Args:
        args          (dict): A dictionary of argument names and their values.
        template (GenParams): A template GenParams object to be used as a base for the new GenParams object.
    """
    genparams = template.copy() if template else GenParams()

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

    # --i, --img-shift <int>
    # "refiner.noise_seed"
    value, as_delta = _pop_int_value(args, "i", "img-shift")
    if value is not None:
        genparams.set_int(f"{RE__}noise_seed", value, as_delta=False)

    # --c, --cfg-shift <int>
    # "base.cfg"
    value, as_delta = _pop_int_value(args, "c", "cfg-shift")
    if value is not None:
        genparams.set_float(f"{BASE}cfg", value * 0.2, as_delta=True)

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

