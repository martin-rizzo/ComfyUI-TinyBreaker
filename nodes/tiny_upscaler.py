"""
File    : tiny_upscaler.py
Purpose : Experimental node to upscale an image using the configuration provided by GenParams
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Apr 10, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
from .core.comfyui_bridge.clip  import CLIP
from .core.comfyui_bridge.vae   import VAE
from .core.comfyui_bridge.model import Model
from .core.tiny_upscale         import tiny_upscale
from .core.genparams            import GenParams
from .core.denoising_params     import DenoisingParams


class TinyUpscaler:
    TITLE       = "💪TB | Tiny Upscaler"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Upscale an image using the configuration provided by GenParams."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "image"    :("IMAGE"     ,{"tooltip": "The image to upscale.",
                                      }),
            "genparams":("GENPARAMS" ,{"tooltip": "The generation parameters containing the upscaling configuration.",
                                      }),
            "model"    :("MODEL"     ,{"tooltip": "The diffusion model used to improve the upscaling quality.",
                                      }),
            "clip"     :("CLIP"      ,{"tooltip": "The CLIP used to encode the prompt for the model.",
                                      }),
            "vae"      :("VAE"       ,{"tooltip": "The VAE used to encode the image for the model.",
                                      }),
            }
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "upscale_using_genparams"
    RETURN_TYPES    = ("IMAGE",)
    RETURN_NAMES    = ("image",)
    OUTPUT_TOOLTIPS = ("The upscaled image.",)

    def upscale_using_genparams(self,
                                image    : torch.Tensor,
                                genparams: GenParams,
                                model    : Model,
                                vae      : VAE,
                                clip     : CLIP,
                                ):

        # get denoising params from the genparams
        denoising = DenoisingParams.from_genparams(genparams, "denoising.upscaler",
                                                   model_to_sample        = model,
                                                   return_none_on_missing = True)
        upscale_by = genparams.get_float("image.upscale_factor")

        if not denoising or not upscale_by:
            return image # no upscaling

        positive, negative = self._encode(clip, denoising.positive, denoising.negative)
        extra_noise        = 0.6
        tile_size          = 1024
        overlap_percent    = 100
        interpolation_mode = "bilinear"

        # upscale the image
        upscaled_image = tiny_upscale(image,
                                      model              = model,
                                      vae                = vae,
                                      positive           = positive,
                                      negative           = negative,
                                      sampler_object     = denoising.sampler_object,
                                      sigmas             = denoising.sigmas,
                                      cfg                = denoising.cfg,
                                      noise_seed         = denoising.noise_seed,
                                      extra_noise        = extra_noise,
                                      upscale_by         = upscale_by,
                                      tile_size          = tile_size,
                                      overlap_percent    = overlap_percent,
                                      interpolation_mode = interpolation_mode,
                                      discard_last_sigma = True,
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
