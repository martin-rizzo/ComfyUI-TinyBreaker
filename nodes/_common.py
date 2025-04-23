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

DEFAULT_ASPECT_RATIO = "1:1  (square)"
DEFAULT_SIZE         = "large"
DEFAULT_ORIENTATION  = "landscape"
DEFAULT_DETAIL_LEVEL = "normal"


#------------------------------ ASPECT RATIO -------------------------------#

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


