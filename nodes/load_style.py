"""
File    : load_style.py
Desc    : Node that loads a pre-defined style from the STYLES.cfg file.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 19, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.style_dict   import StyleDict
from .utils.directories import PROJECT_DIR
from .utils.system      import logger
_PREDEFINED_STYLE_FILE = "STYLES.cfg"

# load the pre-defined styles logging any errors that may occur
try:
    _PREDEFINED_STYLE_DICT = StyleDict.from_file( PROJECT_DIR.get_full_path(_PREDEFINED_STYLE_FILE) )
except FileNotFoundError:
    logger.error(f"The file {_PREDEFINED_STYLE_FILE} was not found.")
    _PREDEFINED_STYLE_DICT = StyleDict()
except PermissionError:
    logger.error(f"You do not have permission to read the file {_PREDEFINED_STYLE_FILE}.")
    _PREDEFINED_STYLE_DICT = StyleDict()
except IsADirectoryError:
    logger.error(f"{_PREDEFINED_STYLE_FILE} is a directory, not a file (!!!).")
    _PREDEFINED_STYLE_DICT = StyleDict()



# TODO: rename to SelectStyle (?)
class LoadStyle:
    TITLE       = "💪TB | Load Style"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Loads a style with parameters for image generation."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selected_style": (cls.style_names(), {"tooltip": "The name of the style to use."}),
            },
            "optional": {
                "custom_styles": ("STRING", {"tooltip": "A string containing a list of custom styles that override the pre-defined styles.",
                                             "default": "", "multiline": True, "forceInput": True}),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_style"
    RETURN_TYPES = ("GENPARAMS",)
    RETURN_NAMES = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters with the selected style loaded.",)


    def load_style(self, selected_style, custom_styles=None):
        # try to load the user styles from string (if any)
        custom_style_dict = StyleDict.from_string(custom_styles) if custom_styles else None

        # load the selected style from pre-defined styles and combine it with the user style
        genparams = _PREDEFINED_STYLE_DICT.get_style_genparams(selected_style)
        if custom_style_dict and selected_style != "none":
            custom_genparams = custom_style_dict.get_style_genparams(selected_style)
            genparams.update(custom_genparams)

        return (genparams,)


    #__ internal functions ________________________________

    @classmethod
    def style_names(cls):
        return _PREDEFINED_STYLE_DICT.names()
