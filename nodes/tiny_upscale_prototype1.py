"""
File    : tiny_upscale_prototype1.py
Purpose : Node to upscale an image using an experimental method (Prototype 1)
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 20, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import comfy.sample
import comfy.samplers
from .xcomfy.model          import Model
from .xcomfy.vae            import VAE
from .xcomfy.helpers.sigmas import calculate_sigmas, split_sigmas
from .xcomfy.helpers.images import normalize_images,                  \
                                   upscale_images,                    \
                                   prepare_image_for_superresolution, \
                                   generate_superresolution_image


class TinyUpscalePrototype1:
    TITLE       = "💪TB | Tiny Upscale (Prototype 1)"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Upscale an image using an experimental method."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "image"    :("IMAGE"         , {"tooltip": "The image to upscale.",
                                           }),
            "model"    :("MODEL"         , {"tooltip": "The model to use for the upscale.",
                                           }),
            "vae"      :("VAE"           , {"tooltip": "The VAE to use for the upscale.",
                                           }),
            "positive" :("CONDITIONING"  , {"tooltip": "The positive conditioning to use for the upscale.",
                                           }),
            "negative" :("CONDITIONING"  , {"tooltip": "The negative conditioning to use for the upscale.",
                                           }),
            "seed"     :("INT"           , {"tooltip": "The random seed used for creating the noise.",
                                            "default": 0, "min": 0, "max": 0xffffffffffffffff,
                                            "control_after_generate": True,
                                           }),
            "steps"    :("INT"           , {"tooltip": "The number of steps used in the denoising process.",
                                            "default": 5, "min": 1, "max": 50,
                                           }),
            "cfg"      :("FLOAT"         , {"tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
                                            "default": 3.0, "min": 0.0, "max": 15.0, "step":0.1, "round": 0.01,
                                           }),
            "sampler"  :(cls._samplers() , {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output.",
                                            "default": "dpmpp_2m",
                                           }),
            "scheduler":(cls._schedulers(),{"tooltip": "The scheduler controls how noise is gradually removed to form the image.",
                                            "default": "karras",
                                           }),
            "strength" :("FLOAT"         , {"tooltip": "The strength of the denoising process. Higher values result in more alterations to the image.",
                                            "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.02,
                                           }),
            "upscale_by":("FLOAT"        , {"tooltip": "The upscale factor.",
                                            "default": 2.0, "min": 1.0, "max": 5.0, "step":0.5,
                                           }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "upscale"
    RETURN_TYPES    = ("IMAGE", "IMAGE"        )
    RETURN_NAMES    = ("image", "lowfreq_guide")
    OUTPUT_TOOLTIPS = ("The upscaled image.", "The low-frequency guide image")

    def upscale(self,
                image,
                model     : Model,
                vae       : VAE,
                positive  : list[ list[torch.Tensor, dict] ],
                negative  : list[ list[torch.Tensor, dict] ],
                seed      : int,
                steps     : int,
                cfg       : float,
                sampler   : str,
                scheduler : str,
                strength  : float,
                upscale_by: float,
                ):
        patch_size = 8 # <- must match the VAE minimum block size

        print()
        print("##>> input image.dtype :", image.dtype)
        print("##>> input image.device:", image.device)


        # adjust the model sampling parameters to match the denoising strength
        total_steps = int(steps / strength)
        steps_start = total_steps - steps

        # calculate sigmas
        model_sampling = model.get_model_object("model_sampling")
        sigmas = calculate_sigmas(model_sampling, sampler, scheduler, total_steps, steps_start)

        # get sampler object
        sampler = comfy.samplers.sampler_object(sampler)

        # prepare image for super-resolution
        image, lowfreq_guide_image, upscaled_width, upscaled_height \
            = prepare_image_for_superresolution(image,
                                                patch_size = patch_size,
                                                upscale_by = upscale_by,
                                                )

        # create the empty upscaled image (in RED color)
        upscaled_image = torch.zeros(1, upscaled_width, upscaled_height, 3)
        upscaled_image[ : , : , : , 0 ] = 1.0


        # # upscale image
        # upscaled_tile = generate_superresolution_image(image,
        #                                                upscale_by = upscale_by,
        #                                                patch_size = patch_size,
        #                                                model    = model,
        #                                                vae      = vae,
        #                                                seed     = seed,
        #                                                cfg      = cfg,
        #                                                sampler  = sampler,
        #                                                sigmas   = sigmas,
        #                                                positive = positive,
        #                                                negative = negative,
        #                                                tile_pos = (x,y),
        #                                                tile_size = (width,height),
        #                                                )

        generate_tile_fn = lambda x, y, width, height: generate_superresolution_image(
                                                        image,
                                                        patch_size      = patch_size,
                                                        upscaled_width  = upscaled_width,
                                                        upscaled_height = upscaled_height,
                                                        model           = model,
                                                        vae             = vae,
                                                        seed            = seed,
                                                        cfg             = cfg,
                                                        sampler         = sampler,
                                                        sigmas          = sigmas,
                                                        positive        = positive,
                                                        negative        = negative,
                                                        tile_pos        = (x, y),
                                                        tile_size       = (width, height)
                                                        )

        #upscaled_image[ : , 8:8+upscaled_tile.shape[1], 8:8+upscaled_tile.shape[2], : ] = upscaled_tile[ : , :256 , :256, : ]
        #upscaled_image[ : , 8:8+256, 8:8+256, : ] = upscaled_tile[ : , :256 , :256, : ]

        tile_filling_zero(upscaled_image, generate_tile_fn)

        print("##>> upscaled_image.dtype:", upscaled_image.dtype)
        print("##>> upscaled_image.device:", upscaled_image.device)
        print("##>> lowfreq_guide_image.dtype:", lowfreq_guide_image.dtype)
        print("##>> lowfreq_guide_image.device:", lowfreq_guide_image.device)
        print()
        return (upscaled_image, lowfreq_guide_image)


    #__ internal functions ________________________________

    @staticmethod
    def _samplers():
        return comfy.samplers.KSampler.SAMPLERS


    @staticmethod
    def _schedulers():
        return comfy.samplers.KSampler.SCHEDULERS



def tile_filling_zero(canvas          : torch.Tensor,
                      generate_tile_fn: callable,
                      ):
    batch_size, canvas_width, canvas_height, channels = canvas.shape
    tile_width  = 768
    tile_height = 768

    for y in range(0,canvas_height,tile_height):
        for x in range(0,canvas_width,tile_width):
            draw_tile(canvas, x,y, generate_tile_fn(x,y,tile_width,tile_height))



def draw_tile(canvas: torch.Tensor,
              x: int,
              y: int,
              tile: torch.Tensor
              ):
    _, canvas_width, canvas_height, _ = canvas.shape
    _,        width,        height, _ = tile.shape
    width  = min( width , canvas_width -x )
    height = min( height, canvas_height-y )
    canvas[ : , x:x+width , y:y+width, : ] = tile[ : , :width , :height , : ]
