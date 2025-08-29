"""
File    : tiny_upscaler_v2.py
Purpose : Node to upscale an image using the configuration provided by GenParams
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
import torch
from .core.system                       import logger
from .core.comfyui_bridge.clip          import CLIP
from .core.comfyui_bridge.vae           import VAE
from .core.comfyui_bridge.model         import Model
from .core.comfyui_bridge.upscale_model import UpscaleModel
from .core.comfyui_bridge.progress_bar  import ProgressBar
from .core.tiny_upscale                 import tiny_upscale
from .core.genparams                    import GenParams
from .core.denoising_params             import DenoisingParams
_MODES = [ "simple", "noisy", "upscaler", "enhancer" ]


class TinyUpscalerV2:
    TITLE       = "💪TB | Tiny Upscaler"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Upscale an image using the configuration provided by GenParams."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams"    :("GENPARAMS" ,{"tooltip": "The generation parameters containing the upscaling configuration.",
                                          }),
            "image"        :("IMAGE"     ,{"tooltip": "The image to upscale.",
                                          }),
            "refiner_model":("MODEL"     ,{"tooltip": "The diffusion model used to improve the upscaling quality.",
                                          }),
            "refiner_clip" :("CLIP"      ,{"tooltip": "The CLIP used to encode the prompt for the refiner model.",
                                          }),
            "refiner_vae"  :("VAE"       ,{"tooltip": "The VAE used to encode/decode the image for the refiner model.",
                                          }),
            "mode"         :(_MODES      ,{"tooltip": "Determines the upscale process\n  simple: basic upscale with no refinement\n  noisy: basic upscale + noise injection\n  upscaler: refined upscale using diffusion \n  enhancer: same as upscaler but without changing the size",
                                          }),
            "scale_by"     :("FLOAT"     ,{"tooltip": "The factor by which to scale the image.",
                                           "default": 3, "min": 1.5, "max": 6.0, "step": 0.1
                                          }),
            },
        "optional": {
            "upscale_model":("UPSCALE_MODEL" ,{"tooltip": "Optional upscaling model to use before refining. If not provided, the upscaling is done with a basic upscaling algorithm.",
                                              }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "upscale_using_genparams"
    RETURN_TYPES    = ("IMAGE",)
    RETURN_NAMES    = ("image",)
    OUTPUT_TOOLTIPS = ("The upscaled image.",)

    def upscale_using_genparams(self,
                                genparams    : GenParams,
                                image        : torch.Tensor,
                                refiner_model: Model,
                                refiner_clip : CLIP,
                                refiner_vae  : VAE,
                                scale_by     : float,
                                mode         : str,
                                upscale_model: UpscaleModel = None
                                ):
        # get denoising params from the genparams
        denoising = DenoisingParams.from_genparams(genparams, "denoising.upscaler",
                                                   model_to_sample        = refiner_model,
                                                   return_none_on_missing = True)
        extra_noise    = genparams.get_float("denoising.upscaler.extra_noise", 0.0)
        enable_upscale = genparams.get_bool("image.enable_upscaler", False)

        # if upscaling is disabled by the user, skip the upscaling
        if not enable_upscale:
            return (image, )

        # the upscaler denoising parameters may be missing if they
        # were not correctly configured in the styles or metadata of the model.
        if not denoising:
            logger.warning("No upscale denoising parameters found. Upscaling will be skipped.")
            return (image, )

        # old tinybreaker version cannot handle upscaling because missing VAE;
        # original pixart-sigma model cannot handle upscaling because missing Refiner model
        if not refiner_model or not refiner_vae:
            logger.warning("No refiner model or refiner VAE available. Upscaling will be skipped.")
            return (image, )

        positive, negative = self._encode(refiner_clip, denoising.positive, denoising.negative)
        tile_size          = 1024
        overlap_percent    = 100
        interpolation_mode = "bilinear"

        # upscale the image
        upscaled_image = tiny_upscale(image,
                                      model              = refiner_model,
                                      vae                = refiner_vae,
                                      positive           = positive,
                                      negative           = negative,
                                      sampler_object     = denoising.sampler_object,
                                      sigmas             = denoising.sigmas,
                                      cfg                = denoising.cfg,
                                      noise_seed         = denoising.noise_seed,
                                      extra_noise        = extra_noise,
                                      upscale_by         = scale_by,
                                      upscale_model      = upscale_model,
                                      tile_size          = tile_size,
                                      overlap_percent    = overlap_percent,
                                      mode               = mode,
                                      interpolation_mode = interpolation_mode,
                                      discard_last_sigma = False,
                                      progress_bar       = ProgressBar.from_comfyui(100),
                                      )
        return (upscaled_image, )


    #__ internal functions ________________________________

    @staticmethod
    def _encode(clip: CLIP,
                positive: str | torch.Tensor,
                negative: str | torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the given positive and negative text using CLIP."""

        if isinstance(positive,str):
            tokens   = clip.tokenize(positive)
            positive = clip.encode_from_tokens_scheduled(tokens)

        if isinstance(negative,str):
            tokens   = clip.tokenize(negative)
            negative = clip.encode_from_tokens_scheduled(tokens)

        return positive, negative
