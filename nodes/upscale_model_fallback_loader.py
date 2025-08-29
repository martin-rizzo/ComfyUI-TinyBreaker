"""
File    : upscale_model_fallback_loader.py
Purpose : Node for loading an alternative upscale model when the input model fails
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Aug 29, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.directories                  import UPSCALE_MODELS_DIR
from .core.comfyui_bridge.upscale_model import UpscaleModel


class UpscaleModelFallbackLoader:
    TITLE       = "💪TB | Upscale Model Fallback Loader"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "This node loads an alternative upscale model when the input model fails to load or does not exist."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "optional": {
            "input_model":("UPSCALE_MODEL",  {"tooltip": "The optional model to check for errors or non-existence.",
                                             }),
            },
        "required": {
            "model_name":(cls.upscalers()  ,{"tooltip": "The fallback model to use when the input_model fails to load.",
                                            }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "load_if_failed"
    RETURN_TYPES    = ("UPSCALE_MODEL",)
    RETURN_NAMES    = ("output model",)
    OUTPUT_TOOLTIPS = ("The output upscale model to use.",)

    def load_if_failed(self, input_model=None, model_name=""):
        upscale_model = input_model
        if not upscale_model:
            state_dict    = UPSCALE_MODELS_DIR.load_state_dict_or_raise(model_name)
            upscale_model = UpscaleModel.from_state_dict(state_dict)
        if not upscale_model:
            raise Exception("Failed to load the fallback upscale model.")
        return (upscale_model, )


    #__ internal functions ________________________________

    @staticmethod
    def upscalers():
        return UPSCALE_MODELS_DIR.get_filename_list()


