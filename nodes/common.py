"""
File    : common.py
Purpose : Common functions and constants that can be used by any node
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 16, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import json
import time
from PIL.PngImagePlugin import PngInfo
from .core.gen_params   import GenParams

#--------------------------- VARIABLE EXPANSION ----------------------------#

def find_next_variable(string: str) -> tuple[str, str, str]:
    """
    Finds the next variable in a string.

    This function searches for the next variable in a string and returns a
    tuple containing the text before, the variable name and the text after.
    If the variable name returned is `END` then there are no more variables
    and the parsing is complete.

    Args:
        string: The string to search for the next variable.
    Returns:
        A tuple containing the text before, the variable name and the text after.
    """
    text    , _ , remainder = string.partition("%")
    var_name, ct, remainder = remainder.partition("%")
    return (text, var_name, remainder) if ct else (string, "END", "")


def expand_variables(template  : str,
                     time      : time.struct_time = None,
                     extra_vars: dict             = None
                     ) -> str:
    """
    Returns a string that is the copy of `template` but with its variables expanded
    Args:
        template  : The string containing the variables to expand.
        time      : The current time.
        extra_vars: A dictionary of additional expandable variables with their values.
    """
    TIME_VARS = ("year", "month", "day", "hour", "minute", "second")
    output = ""

    while True:
        value = None
        text, var, template = find_next_variable(template)

        # if `%END%` (or no more text to parse) then exit the loop
        if var == "END":
            output += text
            break
        # try to resolve time variables
        if (time is not None) and (var in TIME_VARS):
            if   var == "year"  : value = str(time.tm_year)
            elif var == "month" : value = str(time.tm_mon ).zfill(2)
            elif var == "day"   : value = str(time.tm_mday).zfill(2)
            elif var == "hour"  : value = str(time.tm_hour).zfill(2)
            elif var == "minute": value = str(time.tm_min ).zfill(2)
            elif var == "second": value = str(time.tm_sec ).zfill(2)
        # try to resolve full date variable
        elif (time is not None) and var.startswith("date:"):
            value = var[5:]
            value = value.replace("yyyy", str(time.tm_year))
            value = value.replace("yy"  , str(time.tm_year)[-2:])
            value = value.replace("MM"  , str(time.tm_mon ).zfill(2))
            value = value.replace("dd"  , str(time.tm_mday).zfill(2))
            value = value.replace("hh"  , str(time.tm_hour).zfill(2))
            value = value.replace("mm"  , str(time.tm_min ).zfill(2))
            value = value.replace("ss"  , str(time.tm_sec ).zfill(2))
        # try to resolve extra variables
        elif (extra_vars is not None) and (var in extra_vars):
            value = str(extra_vars[var])[:16]

        # if a value was found, add it to the output,
        # otherwise add the variable name without modification
        output += f"{text}{value}" if value is not None else f"{text}%{var}%"

    return output

#-------------------------- IMAGE SCALES & RATIOS --------------------------#

LANDSCAPE_SIZES_BY_ASPECT_RATIO = {
    "1:1 square"      : (1024.0, 1024.0),
    "4:3 tv"          : (1182.4,  886.8),
    "48:35 (35 mm)"   : (1199.2,  874.4),
    "71:50 ~IMAX"     : (1220.2,  859.3),
    "3:2 photo"       : (1254.1,  836.1),
    "16:10 wide"      : (1295.3,  809.5),
    "16:9 hdtv"       : (1365.3,  768.0),
    "2:1 mobile"      : (1448.2,  724.0),
    "21:9 ultrawide"  : (1564.2,  670.4),
    "12:5 anamorphic" : (1586.4,  661.0),
    "70:27 cinerama"  : (1648.8,  636.0),
    "32:9 s.ultrawide": (1930.9,  543.0),
}
SCALES_BY_NAME = {
    "Small"  : 0.82,
    "Medium" : 1.0,
    "Large"  : 1.22,
}
ORIENTATIONS = ["Landscape", "Portrait"]

DEFAULT_ASPECT_RATIO = "1:1 square"
DEFAULT_SCALE_NAME   = "Large"
DEFAULT_ORIENTATION  = "Landscape"


def normalize_aspect_ratio(aspect_ratio: str, *, orientation: str = DEFAULT_ORIENTATION) -> str:
    """
    Normalize aspect ratio to a string of the form 'width:height'.
    Args:
      aspect_ratio (str): The landscape aspect ratio to normalize.
      portrait    (bool): If True, swap the width and height.
    """
    width, height = 1, 1

    # handle aspect ratio as a string of the form 'width:height <some text>'
    if isinstance(aspect_ratio, str) and ':' in aspect_ratio:
      ratio, _, _      = aspect_ratio.strip().partition(' ')
      width, _, height = ratio.partition(':')
      if not width.isdigit() or not height.isdigit():
         width, height = 1, 1

    # handle aspect ratio as a tuple/list of two integers
    elif (isinstance(aspect_ratio, tuple) or isinstance(aspect_ratio, list)) and len(aspect_ratio) == 2:
      width, height = str[aspect_ratio[0]], str[aspect_ratio[1]]

    # if portrait mode, swap the width and height
    if orientation.lower() == "portrait":
        width, height = height, width

    return f"{width}:{height}"

#------------------------------ PNG METADATA -------------------------------#

def create_a1111_params(genparams   : GenParams | None,
                        image_width : int,
                        image_height: int
                        ) -> str:
    """
    Return a string containing generation parameters in A1111 format.
    Args:
        genparams   : A GenParams dictionary containing the original generation parameters.
        image_width : The width of the generated image.
        image_height: The height of the generated image.
    """
    if not genparams:
        return ""

    def a1111_normalized_string(text: str) -> str:
        return text.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # extract and clean up parameters from the GenParams dictionary
    positive      = a1111_normalized_string( genparams.get("user.prompt"  , "") )
    negative      = a1111_normalized_string( genparams.get("user.negative", "") )
    base_steps    = max(0, genparams.get("base.steps"   ,0) - genparams.get("base.start_at_step"   ,0))
    refiner_steps = max(0, genparams.get("refiner.steps",0) - genparams.get("refiner.start_at_step",0))
    sampler       = genparams.get("base.sampler_name")
    cfg_scale     = genparams.get("base.cfg")
    seed          = genparams.get("base.noise_seed")
    width         = image_width
    height        = image_height

    # build A1111 params string
    a1111_params = f"{positive}\nNegative prompt: {negative}\nSteps: {base_steps + refiner_steps}, "
    if sampler:
        a1111_params += f"Sampler: {sampler}, "
    if cfg_scale:
        a1111_params += f"CFG scale: {cfg_scale}, "
    if seed:
        a1111_params += f"Seed: {seed}, "
    if width and height:
        a1111_params += f"Size: {image_width}x{image_height}, "

    # remove the trailing comma and return it
    a1111_params = a1111_params.strip().rstrip(",")
    return a1111_params


def create_png_info(*,# keyword-only arguments #
                    prompt       : dict,
                    extra_pnginfo: dict = None,
                    a1111_params : str  = None
                    ) -> PngInfo | None:
    """
    Return a PngInfo object containing the provided generation parameters.
    PngInfo is a class that is used to store metadata in PNG files.
    It is used by this node to embed generation parameters in PNG files.
    """
    if not any([prompt, extra_pnginfo, a1111_params]):
        return None

    metadata = PngInfo()
    if a1111_params:
        metadata.add_text("parameters", a1111_params)
    if prompt:
        metadata.add_text("prompt", json.dumps(prompt))
    if extra_pnginfo:
        for key, content_dict in extra_pnginfo.items():
            metadata.add_text(key, json.dumps(content_dict))
    return metadata

