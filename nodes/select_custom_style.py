"""
File    : select_style_v2.py
Desc    : Allows the user to select a custom style and apply it to generation parameters
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Aug 11, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.styles                import Styles, load_all_styles_versions
from .core.genparams             import GenParams
from .core.directories           import PROJECT_DIR


# load all versions of the pre-defined styles
_PREDEFINED_STYLES_BY_VERSION, _VERSIONS, _STYLES = \
    load_all_styles_versions(dir_path = PROJECT_DIR.get_full_path("styles"))
_LAST_VERSION = _VERSIONS[0]


class SelectCustomStyle:
    TITLE       = "💪TB | Select Custom Style"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Select the custom style that be used to generate the images."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams"   :("GENPARAMS",{"tooltip": "The generation parameters to be updated."
                                        }),
            "custom_style_index":("INT",{"tooltip": "The index of the custom style to be used.",
                                         "default": 1, "min": 1, "max": 9
                                        }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "select_style"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters updated with the selected custom style. (you can use this output to chain other genparams nodes)",)

    def select_style(self,
                     genparams         : GenParams,
                     custom_style_index: int,
                     ):
        style_name = f"CUSTOM{custom_style_index}"
        genparams  = genparams.copy()

        # if the "CUSTOM1" style does not exist, then initialize it with the PREDEFINED style
        if not genparams.has_style( "CUSTOM1" ):
            # TODO: implement a function to load the pre-defined style in `genparams`
            pass

        # if the selected style by the user does not exist, then use "CUSTOM1"
        if not genparams.has_style( style_name ):
            style_name = "CUSTOM1"

        # apply the style (overwritting all denoising values)
        genparams.apply_style( style_name )
        return (genparams,)


