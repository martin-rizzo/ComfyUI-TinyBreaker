"""
File    : tiny_upscaler_advanced.py
Purpose : Node to upscale an image.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Apr 18, 2025
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
from .core.system                      import logger
from .core.comfyui_bridge.vae          import VAE
from .core.comfyui_bridge.model        import Model
from .core.comfyui_bridge.progress_bar import ProgressBar
from .core.tiny_upscale                import tiny_upscale
_MODE_UPSCALER = "upscaler"
_MODE_ENHANCER = "enhancer"
_MODES = [ _MODE_UPSCALER ] # enhancer mode is not available yet


class TinyUpscalerAdvanced:
    TITLE       = "💪TB | Tiny Upscaler (advanced)"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Upscale an image using custom sampler and sigmas to improve the quality of the result."


    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "image"    :("IMAGE"        ,{"tooltip": "The image to upscale.",
                                         }),
            "model"    :("MODEL"        ,{"tooltip": "The diffusion model used to improve the upscaling quality.",
                                         }),
            "positive" :("CONDITIONING" ,{"tooltip": "The positive conditioning to use for the upscale.",
                                         }),
            "negative" :("CONDITIONING" ,{"tooltip": "The negative conditioning to use for the upscale.",
                                         }),
            "sampler"  :("SAMPLER"      ,{"tooltip": "The sampler to use for improving the upscaled image. The `KSamplerSelect` node from ComfyUI can be connected here.",
                                         }),
            "sigmas"   :("SIGMAS"       ,{"tooltip": "The sigmas of each step of the sampling process. Any sampler node from ComfyUI that has `sigmas` as output can be connected here.",
                                         }),
            "vae"      :("VAE"          ,{"tooltip": "The VAE used to encode the image for the model.",
                                         }),
            "cfg"      :("FLOAT"        ,{"tooltip": "The Classifier-Free Guidance scale balances creativity and prompt adherence. Higher values produce more rigid, artificial images while adhering more closely to the prompt. Lower values allow for more creative and natural results, but potentially with more errors and distortions.",
                                          "default": 4.0, "min": 0.0, "max": 15.0, "step":0.5, "round": 0.01,
                                         }),
            "seed"     :("INT"          ,{"tooltip": "The random seed used for creating the noise.",
                                          "default": 1, "min": 1, "max": 0xffffffffffffffff,
                                          "control_after_generate": True,
                                         }),
            "scale_by" :("FLOAT"        ,{"tooltip": "The factor by which to scale the image.",
                                          "default": 3, "min": 1.5, "max": 6.0, "step": 0.5
                                         }),
            "enabled"  :("BOOLEAN"      ,{"tooltip": "If disabled, this node will pass the image through without alteration",
                                          "default": True,
                                         }),
          # "mode"     :(cls.modes()    ,{"tooltip": "The upscaling mode (experimental).",
          #                              }),
            }
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "upscale_using_genparams"
    RETURN_TYPES    = ("IMAGE",)
    RETURN_NAMES    = ("image",)
    OUTPUT_TOOLTIPS = ("The upscaled image.",)

    def upscale_using_genparams(self,
                                image    : torch.Tensor,
                                model    : Model,
                                positive : list[ list[torch.Tensor, dict] ],
                                negative : list[ list[torch.Tensor, dict] ],
                                sampler  : comfy.samplers.KSAMPLER,
                                sigmas   : torch.Tensor,
                                cfg      : float,
                                seed     : int,
                                vae      : VAE,
                                scale_by : float,
                                enabled  : bool,
                                mode     : str = _MODE_UPSCALER,
                                ):

        # if upscaling is disabled by the user, skip the upscaling
        if not enabled:
            return (image, )

        # original pixart-sigma model cannot handle upscaling because missing Refiner model
        # old tinybreaker version cannot handle upscaling because missing VAE;
        if not model or not vae:
            logger.warning("No refiner Model or VAE available. Upscaling will be skipped.")
            return (image, )

        extra_noise        = 0.6
        tile_size          = 1024
        overlap_percent    = 100
        interpolation_mode = "bilinear"
        keep_original_size = (mode == _MODE_ENHANCER)

        # upscale the image
        upscaled_image = tiny_upscale(image,
                                      model              = model,
                                      vae                = vae,
                                      positive           = positive,
                                      negative           = negative,
                                      sampler_object     = sampler,
                                      sigmas             = sigmas,
                                      cfg                = cfg,
                                      noise_seed         = seed,
                                      extra_noise        = extra_noise,
                                      upscale_by         = scale_by,
                                      tile_size          = tile_size,
                                      overlap_percent    = overlap_percent,
                                      interpolation_mode = interpolation_mode,
                                      keep_original_size = keep_original_size,
                                      discard_last_sigma = False,
                                      progress_bar       = ProgressBar.from_comfyui(steps=100),
                                      )
        return (upscaled_image, )


    #__ internal functions ________________________________

    @staticmethod
    def modes():
        return _MODES


