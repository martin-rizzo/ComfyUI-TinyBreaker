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
import math
from .core.system    import logger
from .core.genparams import GenParams

#----------------------- IMAGE SCALE/RATIO CONSTANTS -----------------------#

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
    "small"  : 1.0,
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

DEFAULT_ASPECT_RATIO   = "1:1  (square)"
DEFAULT_SCALE          = "large"
DEFAULT_ORIENTATION    = "landscape"
DEFAULT_DETAIL_LEVEL   = "normal"
DEFAULT_VAE_PATCH_SIZE = 8
DEFAULT_RESOLUTION     = 1024

_MIN_IMAGE_AREA = 256  * 256
_MAX_IMAGE_AREA = 8192 * 8192


#------------------------------ ASPECT RATIO -------------------------------#

def normalize_aspect_ratio(aspect_ratio: str | tuple,
                           /,*,
                           force_orientation: str | None = None) -> str:
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
    orientation = force_orientation.lower() if isinstance(force_orientation, str) else ""
    if (orientation == "portrait" and width > height) or (orientation == "landscape" and width < height):
        width, height = height, width

    return f"{width}:{height}"


#------------------------------- IMAGE SIZE --------------------------------#


def get_image_size_from_genparams(genparams: GenParams) -> tuple:
    """Calculates the image size based on a set of generation parameters."""

    resolution   = genparams.get    ( "modelspec.resolution", 1024                 )
    scale        = genparams.get    ( "image.scale"         , DEFAULT_SCALE        )
    aspect_ratio = genparams.get_str( "image.aspect_ratio"  , DEFAULT_ASPECT_RATIO )
    orientation  = genparams.get_str( "image.orientation"   , None                 )

    image_width, image_height = calculate_image_size(
                                    resolution,
                                    aspect_ratio = aspect_ratio,
                                    scale        = scale,
                                    orientation  = orientation,
                                    block_size   = DEFAULT_VAE_PATCH_SIZE,
                                    )
    return image_width, image_height



def calculate_image_size(resolution  : str | float | int,
                         aspect_ratio: str | tuple | None = None,
                         scale       : str | float | None = None,
                         orientation : str | None         = None,
                         block_size  : int                = 1,
                         ) -> tuple:
    """Calculates the image size based on a resolution, aspect ratio and scale."""

    # parse the aspect ratio and scale factors
    aspect_ratio = normalize_aspect_ratio(aspect_ratio or "1:1", force_orientation=orientation)
    ratio_numerator, ratio_denominator = _parse_ratio(aspect_ratio)
    scale                              = _parse_scale(scale or 1.0)

    # calculate the image dimensions based on the resolution and aspect ratio
    desired_area   = _parse_image_area(resolution)
    desired_area   = min(max(_MIN_IMAGE_AREA, desired_area), _MAX_IMAGE_AREA)
    desired_width  = math.sqrt(desired_area * ratio_numerator / ratio_denominator)
    desired_height = desired_width * ratio_denominator / ratio_numerator

    # round to nearest block size
    width  = int( desired_width  * scale // block_size ) * block_size
    height = int( desired_height * scale // block_size ) * block_size
    return width, height


def _parse_image_area(resolution: str | float | int) -> int:
    """Calculates the area of an image based on a resolution attribute.
    Args:
        resolution: The resolution attribute, this can be provided as:
            - An integer or float: Interpreted as a side length (e.g., 1024).
                                   The area is calculated as side * side.
            - A string: Can be an integer string ("1024"), interpreted as a side length,
                        or a "widthxheight" string ("1024x768").
    """
    if isinstance(resolution, (float,int)):
        return int(resolution) * int(resolution)

    if isinstance(resolution, str):
        resolution = resolution.lower().removesuffix("px")

        if 'x' in resolution:
            width, _, height = resolution.partition('x')
            try:
                width  = int(width.strip())
                height = int(height.strip())
                return width * height
            except ValueError:
                pass
        else:
            try:
                size = int(resolution)
                return size * size
            except ValueError:
                pass

    # if none of the above conditions are met, use the default resolution
    logger.debug(f"Invalid resolution: '{resolution}'. Using default value.")
    return DEFAULT_RESOLUTION * DEFAULT_RESOLUTION


def _parse_ratio(ratio: str | float) -> tuple:
    """Returns a tuple of two integers (numerator, denominator) from a ratio."""

    # asume the string is a fraction with the format "numerator:denominator"
    if isinstance(ratio, str) and ':' in ratio:
        numerator, denominator = ratio.split(':',1)
        try:
            return int(numerator), int(denominator)
        except ValueError as e:
            pass

    # asume the string is a decimal number representing the width/height ratio
    elif isinstance(ratio, str) and '.' in ratio:
        try:
            return float(ratio), 1
        except ValueError as e:
            pass

    # asume the ratio is a float number representing the width/height ratio
    elif isinstance(ratio, float):
        return ratio, 1

    # if none of the above conditions are met, use the default value
    logger.debug(f"Invalid aspect ratio: '{ratio}'. Using default value.")
    return 1, 1


def _parse_scale(scale: str | float) -> float:
    """Returns the scale factor from a string or number."""

    if isinstance(scale, float):
        return scale

    elif scale in SCALES_BY_NAME:
        return SCALES_BY_NAME[scale]

    elif isinstance(scale, str):
        try:
            return float(scale)
        except ValueError:
            pass

    # if none of the above conditions are met, use the default value
    logger.debug(f"Invalid scale factor: '{scale}'. Using default value.")
    return 1.0



#--------------------------- STRING MANIPULATION ---------------------------#

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


