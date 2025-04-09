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
from .functions.styles    import Styles, load_all_styles_versions
from .functions.genparams import GenParams
from .utils.directories   import PROJECT_DIR


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
                                             "default": "", "multiline": True, "forceInput": True
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
        genparams    = genparams.copy()

        # replace "last" with the last available version
        if version == "last":
            version = _LAST_VERSION

        print()
        print("##>> LAST_VERSION:", _LAST_VERSION)
        print()

        # load the pre-defined styles in `genparams` using the keys "styles.<style_name>"
        predefined_styles = _PREDEFINED_STYLES_BY_VERSION[version]
        genparams.update( predefined_styles.to_genparams(prefix_to_add="styles") )

        # try to load the user styles from string (if any)
        custom_styles = Styles.from_string(custom_definitions) if custom_definitions else None
        if custom_styles:
            genparams.update( custom_styles.to_genparams(prefix_to_add="styles") )

        # copy parameters from "styles.<style_name>.*" -> "denoising.*"
        count = genparams.copy_parameters( target="denoising", source=f"styles.{style_name}", valid_subkeys=["base", "refiner"])
        if count > 0:
            genparams.set_str("user.style", style_name)

        return (genparams,)


    #__ internal functions ________________________________

    @classmethod
    def versions(cls) -> list[str]:
        return [ "last", *_VERSIONS ]


    @classmethod
    def styles(cls) -> list[str]:
        return _STYLES

