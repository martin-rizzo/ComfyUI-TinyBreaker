"""
File    : custom_style.py
Purpose : Node for configuring a custom style for TinyBreaker
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
from comfy.samplers  import KSampler
from .core.styles    import Styles
from .core.genparams import GenParams

class CustomStyle:
    TITLE       = "💪TB | Custom Style"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Configure a custom style"

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams"     :("GENPARAMS",     {"tooltip": "The generation parameters to be updated.",
                                               }),
            "custom_style_index":("INT"       ,{"tooltip": 'The index of the custom style to be configured. The index can be used to choose this style in the "Select Custom Style" node.',
                                                "default": 1, "min": 1, "max": 9
                                               }),
            "base_prompt"   :("STRING"        ,{"tooltip": "The text used as template for the prompt.",
                                                "multiline": True, "dynamicPrompts": True,
                                               }),
            "base_negative" :("STRING"        ,{"tooltip": "The text used as template for the negative prompt.",
                                                "multiline": True, "dynamicPrompts": True,
                                               }),
            "base_sampler"  :(cls.samplers(),  {"tooltip": "The algorithm used when sampling, this can affect the quality, speed and composition of images created with this style.",
                                               }),
            "base_scheduler":(cls.schedulers(),{"tooltip": "The scheduler controls how noise is gradually removed to form the image."
                                               }),
            "base_steps"    :("INT"           ,{"tooltip": "The total number of steps",
                                                "default": 20, "min": 1, "max": 10000
                                               }),
            "base_start_at_step":("INT"       ,{"tooltip": "The step where the denoising process should start. This allows you to select a range within the total number of steps at which the image will be denoised. Use 0 to start from the beginning.",
                                                "default": 0, "min": 0, "max": 10000
                                               }),
            "base_end_at_step"  :("INT"       ,{"tooltip": "The step where the denoising process should end. This allows you to select a range within the total number of steps at which the image will be denoised. Use 10000 to end at the last step.",
                                                "default": 10000, "min": 0, "max": 10000
                                               }),
            "base_cfg"      :("FLOAT"         ,{"tooltip": "The Classifier-Free Guidance scale balances creativity and prompt adherence. Higher values produce more rigid, artificial images while adhering more closely to the prompt. Lower values allow for more creative and natural results, but potentially with more errors and distortions.",
                                                "default": 4.0, "min": 0.0, "max": 15.0, "step":0.5, "round": 0.01,
                                               }),
            "refi_prompt"   :("STRING"        ,{"tooltip": "The text used as template for the prompt.",
                                                "multiline": True, "dynamicPrompts": True,
                                               }),
            "refi_negative" :("STRING"        ,{"tooltip": "The text used as template for the negative prompt.",
                                                "multiline": True, "dynamicPrompts": True,
                                               }),
            "refi_sampler"  :(cls.samplers(),  {"tooltip": "The algorithm used when sampling, this can affect the quality, speed and composition of images created with this style.",
                                               }),
            "refi_scheduler":(cls.schedulers(),{"tooltip": "The scheduler controls how noise is gradually removed to form the image."
                                               }),
            "refi_steps"    :("INT"           ,{"tooltip": "The total number of steps",
                                                "default": 20, "min": 1, "max": 10000
                                               }),
            "refi_start_at_step":("INT"       ,{"tooltip": "The step where the denoising process should start. This allows you to select a range within the total number of steps at which the image will be denoised. Use 0 to start from the beginning.",
                                                "default": 0, "min": 0, "max": 10000
                                               }),
            "refi_end_at_step"  :("INT"       ,{"tooltip": "The step where the denoising process should end. This allows you to select a range within the total number of steps at which the image will be denoised. Use 10000 to end at the last step.",
                                                "default": 10000, "min": 0, "max": 10000
                                               }),
            "refi_cfg"      :("FLOAT"         ,{"tooltip": "The Classifier-Free Guidance scale balances creativity and prompt adherence. Higher values produce more rigid, artificial images while adhering more closely to the prompt. Lower values allow for more creative and natural results, but potentially with more errors and distortions.",
                                                "default": 4.0, "min": 0.0, "max": 15.0, "step":0.5, "round": 0.01,
                                               }),
            "upscaler_name" :("STRING"        ,{"tooltip": "The name of a custom upscaler configuration to use with this style.",
                                                "multiline": False, "dynamicPrompts": False,
                                               }),
            },
        }


    #__ FUNCTION __________________________________________
    FUNCTION = "add_custom_style"
    RETURN_TYPES    = ("GENPARAMS",)
    RETURN_NAMES    = ("genparams",)
    OUTPUT_TOOLTIPS = ("The generation parameters updated with the custom style. (you can use this output to chain other genparams nodes)",)

    @classmethod
    def add_custom_style(self,
                         genparams: GenParams, custom_style_index: int = 1,
                         base_prompt:str="", base_negative:str="",
                         base_sampler: str="euler", base_scheduler: str="simple",
                         base_steps:int=20, base_start_at_step:int=0, base_end_at_step:int=1000, base_cfg:float=4.0,
                         refi_prompt:str ="", refi_negative:str="",
                         refi_sampler: str="euler", refi_scheduler: str = "simple",
                         refi_steps:int=20, refi_start_at_step:int=0, refi_end_at_step:int=1000, refi_cfg:float=4.0,
                         upscaler_name:str="",
                         ):
        style_name = f"CUSTOM{custom_style_index}"

        # build a kv table with the base/refiner model parameters
        style_raw_kv_params = {
            "base.prompt"     : base_prompt,
            "base.negative"   : base_negative,
            "base.sampler"    : base_sampler,
            "base.scheduler"  : base_scheduler,
            "base.steps"      : base_steps,
            "base.steps_start": base_start_at_step,
            "base.steps_end"  : base_end_at_step,
            "base.cfg"        : base_cfg,
            "refiner.prompt"     : refi_prompt,
            "refiner.negative"   : refi_negative,
            "refiner.sampler"    : refi_sampler,
            "refiner.scheduler"  : refi_scheduler,
            "refiner.steps"      : refi_steps,
            "refiner.steps_start": refi_start_at_step,
            "refiner.steps_end"  : refi_end_at_step,
            "refiner.cfg"        : refi_cfg,
        }
        if upscaler_name:
            style_raw_kv_params["upscaler.name"] = upscaler_name

        # create a new set of styles containing only one style with the given name
        custom_styles = Styles.from_single_style(style_name, style_raw_kv_params)

        # add the new custom style to the existing available styles
        # (the available style are stored in genparams under the prefix "styles.")
        genparams = genparams.copy().add_styles( custom_styles )
        return (genparams,)


    #__ internal functions ________________________________

    @staticmethod
    def samplers():
        return KSampler.SAMPLERS

    @staticmethod
    def schedulers():
        return KSampler.SCHEDULERS
