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
import torch.nn.functional as F
import comfy.utils
import comfy.samplers
from .xcomfy.helpers.sigmas import calculate_sigmas
from .xcomfy.helpers.images import normalize_images, tiny_encode, refine_latent_image
from .xcomfy.helpers.tiles  import apply_tiles_tlbr, apply_tiles_brtl
from .xcomfy.vae            import VAE
from .xcomfy.model          import Model


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
    RETURN_TYPES    = ("LATENT",)
    RETURN_NAMES    = ("latent",)
    OUTPUT_TOOLTIPS = ("The upscaled image in latent space.",)

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
                interpolation_mode: str = "bilinear" # "nearest"
                ):
        sampler = comfy.samplers.sampler_object(sampler)
        image   = normalize_images(image)
        _, image_height, image_width, _ = image.shape

        # calculate sigmas (old)
        # total_steps    = int(steps / strength)
        # steps_start    = total_steps - steps
        # steps_end      = 10000
        # model_sampling = model.get_model_object("model_sampling")
        # sigmas         = calculate_sigmas(model_sampling, sampler, scheduler, total_steps, steps_start, steps_end)

        # calculate sigmas (new)
        model_sampling = model.get_model_object("model_sampling")
        sigmas         = calculate_sigmas(model_sampling, sampler, scheduler, steps, start_at_step, end_at_step)

        # upscale the image using simple interpolation
        upscaled_width  = int( round(image_width  * upscale_by) )
        upscaled_height = int( round(image_height * upscale_by) )
        upscaled_image  = F.interpolate(image.transpose(1,-1),
                                        size = (upscaled_width, upscaled_height),
                                        mode = interpolation_mode).transpose(1,-1)

        # encode the image into latent space
        upscaled_latent = tiny_encode(upscaled_image,
                                      vae          = vae,
                                      tile_size    = tile_size,
                                      tile_padding = (tile_size/4),
                                      )

        # add extra noise to the latent image if requested
        if extra_noise > 0.0:
            upscaled_latent += torch.randn_like(upscaled_latent) * extra_noise


        sigmas = sigmas[:-1]
        number_of_steps = len(sigmas)-1
        pbar, pstep     = comfy.utils.ProgressBar(1000), int(1000 / number_of_steps)
        for step in range(number_of_steps):
            progress_bar = (pbar,step*pstep,pstep)

            # build the function (lambda) for refining tiles of the latent image
            #     latent       : the latent image to refine
            #     x, y         : the coordinates of the tile to refine
            #     width, height: the size of the tile to refine
            refine_tile = lambda latent, x, y, width, height: \
                refine_latent_image(latent,
                                    model      = model,
                                    add_noise  = (step==0),
                                    noise_seed = seed,
                                    cfg        = cfg,
                                    sampler    = sampler,
                                    sigmas     = torch.Tensor( [sigmas[step], sigmas[step+1]] ),
                                    positive   = positive,
                                    negative   = negative,
                                    tile_pos   = (x,y),
                                    tile_size  = (width,height),
                                    )

            # process the latent image in tiles using the refinement function,
            # alternating between top-left to bottom-right and
            # bottom-right to top-left directions
            if step % 2 == 0:
                apply_tiles_tlbr(upscaled_latent,
                                 create_tile_func = refine_tile,
                                 tile_size        = int(tile_size//8),
                                 progress_bar     = progress_bar)
            else:
                apply_tiles_brtl(upscaled_latent,
                                 create_tile_func = refine_tile,
                                 tile_size        = int(tile_size//8),
                                 progress_bar     = progress_bar)

        return ({"samples":upscaled_latent}, )


    #__ internal functions ________________________________

    @staticmethod
    def _samplers():
        return comfy.samplers.KSampler.SAMPLERS


    @staticmethod
    def _schedulers():
        return comfy.samplers.KSampler.SCHEDULERS





