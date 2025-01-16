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
from .core.style_dict   import StyleDict, load_all_style_dict_versions
from .utils.directories import PROJECT_DIR
from .utils.system      import logger
_STYLES_DIR = PROJECT_DIR.get_full_path("styles")

# load all versions of the pre-defined styles
_PREDEFINED_STYLE_DICTS_BY_VERSION = load_all_style_dict_versions(dir_path=_STYLES_DIR)


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
                "style_name": (cls.style_names(), {"tooltip": "The name of the style to use."}),
                "version"   : (cls.versions()   , {"tooltip": "The version of the file with the pre-defined styles."}),
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


    def load_style(self, version: str, style_name: str, custom_styles=None):

        # load the pre-defined style
        predefined_style_dict = _PREDEFINED_STYLE_DICTS_BY_VERSION[version]
        genparams = predefined_style_dict.get_style_genparams(style_name)

        # try to load the user styles from string (if any)
        custom_style_dict = StyleDict.from_string(custom_styles) if custom_styles else None
        if custom_style_dict and style_name != "none":
            genparams = genparams.copy()
            genparams.update( custom_style_dict.get_style_genparams(style_name) )

        return (genparams,)


    #__ internal functions ________________________________

    @classmethod
    def versions(cls):
        return list( _PREDEFINED_STYLE_DICTS_BY_VERSION.keys() )


    @classmethod
    def style_names(cls):
        # return the list of style of the first valid version
        # (versions are ordered by number, from most recent to oldest)
        for style_dict in _PREDEFINED_STYLE_DICTS_BY_VERSION.values():
            names = style_dict.names()
            if len(names) > 0:
                return names
        return []

