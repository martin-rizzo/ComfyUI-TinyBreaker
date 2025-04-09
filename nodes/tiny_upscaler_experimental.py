"""
File    : tiny_upscaler_experimental.py
Purpose : Node to upscale an image using an experimental method.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 24, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import comfy.samplers
from .core.tiny_upscale import tiny_upscale
from .xcomfy.vae        import VAE
from .xcomfy.model      import Model


class TinyUpscalerExperimental:
    TITLE       = "💪TB | Tiny Upscaler (Experimental)"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Upscale an image using an experimental method."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "image"    :("IMAGE"        ,{"tooltip": "The image to upscale.",
                                         }),
            "model"    :("MODEL"        ,{"tooltip": "The model to use for the upscale.",
                                         }),
            "positive" :("CONDITIONING" ,{"tooltip": "The positive conditioning to use for the upscale.",
                                         }),
            "negative" :("CONDITIONING" ,{"tooltip": "The negative conditioning to use for the upscale.",
                                         }),
            "vae"      :("VAE"          ,{"tooltip": "The VAE to use for the upscale.",
                                         }),
            "seed"     :("INT"          ,{"tooltip": "The random seed used for creating the noise.",
                                          "default": 0, "min": 0, "max": 0xffffffffffffffff,
                                          "control_after_generate": True,
                                         }),
            "steps"    :("INT"          ,{"tooltip": "The number of steps used in the denoising process.",
                                          "default": 20, "min": 1, "max": 1000,
                                         }),
            "start_at_step":("INT"      ,{"tooltip": "???.",
                                          "default": 11, "min": 1, "max": 50,
                                         }),
            "end_at_step":("INT"        ,{"tooltip": "???.",
                                          "default": 16, "min": 1, "max": 50,
                                         }),
            "cfg"      :("FLOAT"        ,{"tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
                                          "default": 4.0, "min": 0.0, "max": 15.0, "step":0.5, "round": 0.01,
                                         }),
            "sampler"  :(cls._samplers(),{"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output.",
                                          "default": "dpmpp_2m",
                                         }),
            "scheduler":(cls._schedulers(),{"tooltip": "The scheduler controls how noise is gradually removed to form the image.",
                                            "default": "karras",
                                         }),
            "extra_noise":("FLOAT"      ,{"tooltip": "The amount of extra noise to add to the image during the denoising process.",
                                          "default": 0.6, "min": 0.0, "max": 1.5, "step":0.1,
                                         }),
            "upscale_by":("FLOAT"       ,{"tooltip": "The upscale factor.",
                                          "default": 3.0, "min": 1.0, "max": 5.0, "step":0.5,
                                         }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "upscale"
    RETURN_TYPES    = ("IMAGE",)
    RETURN_NAMES    = ("image",)
    OUTPUT_TOOLTIPS = ("The upscaled image.",)

    def upscale(self,
                image             : torch.Tensor,
                model             : Model,
                vae               : VAE,
                positive          : list[ list[torch.Tensor, dict] ],
                negative          : list[ list[torch.Tensor, dict] ],
                seed              : int,
                steps             : int,
                start_at_step     : int,
                end_at_step       : int,
                cfg               : float,
                sampler           : str,
                scheduler         : str,
                extra_noise       : float,
                upscale_by        : float,
                tile_size         : int = 1024,
                overlap_percent   : int = 100,
                interpolation_mode: str = "bilinear" # "nearest"
                ):

        upscaled_image = tiny_upscale(image,
                                      model              = model,
                                      vae                = vae,
                                      positive           = positive,
                                      negative           = negative,
                                      seed               = seed,
                                      steps              = steps,
                                      start_at_step      = start_at_step,
                                      end_at_step        = end_at_step,
                                      cfg                = cfg,
                                      sampler            = sampler,
                                      scheduler          = scheduler,
                                      extra_noise        = extra_noise,
                                      upscale_by         = upscale_by,
                                      tile_size          = tile_size,
                                      overlap_percent    = overlap_percent,
                                      interpolation_mode = interpolation_mode,
                                      )
        return (upscaled_image, )


    #__ internal functions ________________________________

    @staticmethod
    def _samplers():
        return comfy.samplers.KSampler.SAMPLERS


    @staticmethod
    def _schedulers():
        return comfy.samplers.KSampler.SCHEDULERS





