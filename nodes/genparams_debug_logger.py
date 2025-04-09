"""
File    : genparams_debug_logger.py
Purpose : Node that outputs to console the content of GenParams
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 31, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.genparams import GenParams

_MODES = [
    "no logging",
    "user parameters",
    "short text values",
    "long text values",
    "key/value pairs",
]
_DEFAULT_MODE = "user parameters"

_DEFAULT_PARAMS = {
    "file.name"                  : "TinyBreaker_prototype0.safetensors",
    "file.date"                  : "2025-01-28",
    "modelspec.sai_model_spec"   : "1.0.0",
    "modelspec.title"            : "TinyBreaker prototype0",
    "modelspec.date"             : "2025-01-28",
    "modelspec.architecture"     : "TinyBreaker",
    "modelspec.description"      : "TinyBreaker is a hybrid model",
    "modelspec.resolution"       : "1024x1024",
    "modelspec.implementation"   : "reference",
    "modelspec.author"           : "Martin Rizzo",
    "modelspec.license"          : "MIT",
    "image.aspect_ratio"         : "1:1",
    "image.scale"                : 1.22,
    "image.batch_size"           : 1,
    "denoising.base.prompt"        : "A detailed iPhone photo. A cute corgi, smiling with great joy, flying high above the river bank in Yosemite National Park, wearing a Superman cape. Bokeh",
    "denoising.base.negative"      : "macrophoto, bokeh, out of focus, draw",
    "denoising.base.sampler"       : "uni_pc",
    "denoising.base.scheduler"     : "simple",
    "denoising.base.steps"         : 12,
    "denoising.base.steps_start"   : 2,
    "denoising.base.cfg"           : 3.4,
    "denoising.base.noise_seed"    : 1,
    "denoising.refiner.prompt"     : "A cute corgi, smiling with great joy, flying high above the river bank in Yosemite National Park, wearing a Superman cape. Bokeh",
    "denoising.refiner.negative"   : "(draw, worst quality, low quality:1.8)",
    "denoising.refiner.sampler"    : "deis",
    "denoising.refiner.scheduler"  : "ddim_uniform",
    "denoising.refiner.steps"      : 11,
    "denoising.refiner.steps_start": 6,
    "denoising.refiner.cfg"        : 2.0,
    "denoising.refiner.noise_seed" : 1,
}


class GenParamsDebugLogger:
    TITLE       = "💪TB | GenParams Debug Logger"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Allows you to see the content of the GenParams node in the console."
    OUTPUT_NODE = True

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "mode"     :(_MODES  , {"tooltip": "Select the logging mode.",
                                    "default": _DEFAULT_MODE
                                    }),
            "flag"     :("STRING", {"tooltip": "Flag to mark the logging.",
                                    "multiline": False
                                    }),
            "filter"   :("STRING", {"tooltip": "The prefix of the parameter to be logged, empty to log all parameters. e.g. image.aspect_ratio",
                                    "default": ""
                                    }),
            },
        "optional": {
            "genparams":("GENPARAMS", {"tooltip": "The generation parameters to be logged.",
                                       }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "log"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The same input genparams. (you can use this output to chain other genparams nodes)",)

    def log(self, flag: str, mode: str, filter: str = "", genparams: GenParams = None) -> GenParams:
        original_genparams = genparams or GenParams(_DEFAULT_PARAMS)

        # if a filter is provided,
        # filter out all keys that do not start with the filter prefix
        if filter:
            genparams = GenParams({key: value for key, value in genparams.items() if key.startswith(filter)})

        # if flag is provided, add a separator banner to the console output
        if flag:
            print(f"-[ 🚩 {flag.upper()} ]------------------------------------")

        # log the selected parameters based on the chosen logging mode
        if   mode == "no logging":
            pass
        elif mode == "user parameters":
            print( genparams.to_string(filter_prefixes=["user","image","denoising"]) )
        elif mode == "short text values":
            print( genparams.to_string(width=94) )
        elif mode == "long text values":
            print( genparams.to_string(width=-1) )
        elif mode == "key/value pairs":
            for key, value in genparams.items():
                print(f"{key}: {value}")

        # return the original genparams making the node a pass-through
        return (original_genparams,)

