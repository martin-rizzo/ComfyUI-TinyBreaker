"""
File    : genparams_debug_logger.py
Purpose : Node that outputs to console the content of GenParams
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 31, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.genparams import GenParams

_MODES = [
    "quiet",
    "user parameters only",
    "short values",
    "long values",
    "all keys with long values",
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
    "sampler.base.prompt"        : "A detailed iPhone photo. A cute corgi, smiling with great joy, flying high above the river bank in Yosemite National Park, wearing a Superman cape. Bokeh",
    "sampler.base.negative"      : "macrophoto, bokeh, out of focus, draw",
    "sampler.base.sampler"       : "uni_pc",
    "sampler.base.scheduler"     : "simple",
    "sampler.base.steps"         : 12,
    "sampler.base.steps_start"   : 2,
    "sampler.base.cfg"           : 3.4,
    "sampler.base.noise_seed"    : 1,
    "sampler.refiner.prompt"     : "A cute corgi, smiling with great joy, flying high above the river bank in Yosemite National Park, wearing a Superman cape. Bokeh",
    "sampler.refiner.negative"   : "(draw, worst quality, low quality:1.8)",
    "sampler.refiner.sampler"    : "deis",
    "sampler.refiner.scheduler"  : "ddim_uniform",
    "sampler.refiner.steps"      : 11,
    "sampler.refiner.steps_start": 6,
    "sampler.refiner.cfg"        : 2.0,
    "sampler.refiner.noise_seed" : 1,
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
            "flag"     :("STRING"   , {"tooltip": "Flag to mark the logging.",
                                       "multiline": False
                                       }),
            "mode"     :(_MODES     , {"tooltip": "Select the logging mode.",
                                       "default": _DEFAULT_MODE
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

    def log(self, flag: str, mode: str, genparams: GenParams = None) -> GenParams:

        if not genparams:
            genparams = GenParams(_DEFAULT_PARAMS)

        if mode == "quiet":
            return (genparams,)

        if flag:
            print(f"-- 🚩{flag} -------------------------------------")

        if   mode == "user parameters only":
            print( genparams.to_string(filter_prefixes=["image","sampler","user"]) )
        elif mode == "short values":
            print( genparams )
        elif mode == "long values":
            print( genparams.to_string() )
        elif mode == "all keys with long values":
            for key, value in genparams.items():
                print(f"{key}: {value}")

        return (genparams,)

