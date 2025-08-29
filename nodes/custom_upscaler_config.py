"""
File    : custom_upscaler_config.py
Purpose : Node for configuring parameters for the TinyUpscaler
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Aug 24, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from comfy.samplers  import KSampler
from .core.styles    import Styles
from .core.genparams import GenParams

class CustomUpscalerConfig:
    TITLE       = "💪TB | Custom Upscaler Config"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Configure the parameters for the upscaler and associate it to a name."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams"    :("GENPARAMS",{"tooltip": "The generation parameters to be updated.",
                                         }),
            "upscaler_name":("STRING"   ,{"tooltip": "The name to be associated with this upscaler configuration. (It will be the name that will allow you to select it later within the custom style.",
                                          "multiline": False, "dynamicPrompts": False,
                                         }),
            "prompt"       :("STRING"   ,{"tooltip": "The text used as prompt during the upscale process. (It should be something generic.)",
                                          "multiline": True, "dynamicPrompts": True,
                                         }),
            "negative"     :("STRING"   ,{"tooltip": "The text used as negative prompt during the upscale process. (It should be something generic.)",
                                          "multiline": True, "dynamicPrompts": True,
                                         }),
            "sampler"  :(KSampler.SAMPLERS,  {"tooltip": "The algorithm used when sampling, this can affect the quality and speed of the upscale process.",
                                             }),
            "scheduler":(KSampler.SCHEDULERS,{"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
            "steps"    :("INT"              ,{"tooltip": "The total number of steps",
                                              "default": 20, "min": 1, "max": 10000
                                             }),
            "start_at_step":("INT"          ,{"tooltip": "The step where the denoising process should start. This allows you to select a range within the total number of steps at which the image will be denoised. Use 0 to start from the beginning.",
                                              "default": 0, "min": 0, "max": 10000
                                             }),
            "end_at_step": ("INT"           ,{"tooltip": "The step where the denoising process should end. This allows you to select a range within the total number of steps at which the image will be denoised. Use 10000 to end at the last step.",
                                              "default": 10000, "min": 0, "max": 10000
                                             }),
            "cfg"        :("FLOAT"          ,{"tooltip": "The Classifier-Free Guidance scale.",
                                              "default": 4.0, "min": 0.0, "max": 15.0, "step":0.5, "round": 0.01,
                                             }),
            "extra_noise":("FLOAT"          ,{"tooltip": "The noise level added to the image before it is denoised.",
                                              "default": 0.5, "min": 0.0, "max": 0.9, "step":0.1, "round": 0.01,
                                             }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "add_custom_upscaler"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters updated with the custom style. (you can use this output to chain other genparams nodes)",)

    @classmethod
    def add_custom_upscaler(self,
                            genparams: GenParams,
                            upscaler_name: str = "DEFAULT",
                            prompt:str="",
                            negative:str="",
                            sampler: str="euler",
                            scheduler: str="simple",
                            steps:int=20,
                            start_at_step:int=0,
                            end_at_step:int=1000,
                            cfg:float=4.0,
                            extra_noise:float=0.6,
                            ):

        # build a kv table with the upscaler parameters
        upscaler_raw_kv_params = {
            "prompt"      : prompt,
            "negative"    : negative,
            "sampler"     : sampler,
            "scheduler"   : scheduler,
            "steps"       : steps,
            "steps_start" : start_at_step,
            "steps_end"   : end_at_step,
            "cfg"         : cfg,
            #"noise_seed" : 1,
            "extra_noise" : extra_noise,
        }

        custom_upscaler = Styles()
        custom_upscaler.add_style(upscaler_name, upscaler_raw_kv_params)
        # custom_upscaler = Styles(upscaler_name, upscaler_raw_kv_params)

        # update the available upscaler-configs with the new one
        # (the available upscaler-configs are found in genparams under the prefix "upscalers.*")
        genparams = genparams.copy()
        genparams.update( custom_upscaler.to_genparams(prefix_to_add="upscalers") )
        return (genparams,)


    #__ internal functions ________________________________

