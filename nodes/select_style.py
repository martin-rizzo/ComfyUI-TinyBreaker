"""
File    : load_style.py
Desc    : Node to select a pre-defined style and load it into the generation parameters.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 19, 2024
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


class SelectStyle:
    TITLE       = "💪TB | Select Style"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Select a predefined style from a list of available options, packing it into GenParams. Additionally, you can also provide custom definitions to create your own unique styles."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams" :("GENPARAMS"   , {"tooltip": "The generation parameters to be updated."
                                          }),
            "style_name":(cls.styles()  , {"tooltip": "The name of the style to use."
                                          }),
            "version"   :(cls.versions(), {"tooltip": "The version of the styles file to use. By selecting a particular version, you're choosing a snapshot of the style at that point in time."
                                          }),
            },
        "optional": {
            "custom_definitions":("STRING", {"tooltip": "A string containing a list of custom styles that override the pre-defined styles.",
                                             "multiline"     :  True,
                                             "dynamicPrompts": False,
                                             "forceInput"    :  True,
                                             "default"       :    "",
                                             }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_style"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters updated with the selected style. (you can use this output to chain other genparams nodes)",)

    def load_style(self,
                   genparams         : GenParams,
                   version           : str,
                   style_name        : str,
                   custom_definitions: str = None
                   ):
        genparams = genparams.copy()

        # replace "last" with the last available version
        if version == "last":
            version = _LAST_VERSION

        # load the pre-defined styles in `genparams` using the keys "styles.<style_name>"
        predefined_styles = _PREDEFINED_STYLES_BY_VERSION[version]
        genparams.add_styles( predefined_styles )

        # try to load the user styles from string (if any)
        if custom_definitions:
            custom_styles = Styles.from_string(custom_definitions)
            genparams.add_styles( custom_styles )

        # apply the style (overwritting all denoising values)
        genparams.apply_style( style_name )

        return (genparams,)


    #__ internal functions ________________________________

    @classmethod
    def versions(cls) -> list[str]:
        return [ "last", *_VERSIONS ]


    @classmethod
    def styles(cls) -> list[str]:
        return _STYLES

