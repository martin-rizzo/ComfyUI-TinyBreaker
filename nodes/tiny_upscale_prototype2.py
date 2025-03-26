"""
File    : tiny_upscale_prototype2.py
Purpose : Node to upscale an image using an experimental method (Prototype 2)
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
from .xcomfy.helpers.images import normalize_images, tiny_encode
from .xcomfy.vae            import VAE


class TinyUpscalePrototype2:
    TITLE       = "💪TB | Tiny Upscale (Prototype 2)"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Upscale an image using an experimental method."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "image"     :("IMAGE"        ,{"tooltip": "The image to upscale.",
                                          }),
            "vae"       :("VAE"          ,{"tooltip": "The VAE to use for the upscale.",
                                          }),
            "upscale_by":("FLOAT"        ,{"tooltip": "The upscale factor.",
                                           "default": 2.0, "min": 1.0, "max": 5.0, "step":0.5,
                                          }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "upscale"
    RETURN_TYPES    = ("LATENT",)
    RETURN_NAMES    = ("latent",)
    OUTPUT_TOOLTIPS = ("The upscaled image in latent space.",)

    def upscale(self, image: torch.Tensor, vae: VAE, upscale_by: float):
        image = normalize_images(image)
        batch_size, image_height, image_width, channels = image.shape
        extra_noise = 0.8  # <- adjust this value to control the amount of extra noise to add
        tile_size =  512

        # upscale the image using simple bilinear interpolation
        upscaled_width  = int( round(image_width  * upscale_by) )
        upscaled_height = int( round(image_height * upscale_by) )
        upscaled_image  = F.interpolate(image.transpose(1,-1),
                                        size = (upscaled_width, upscaled_height),
                                        mode = "bilinear").transpose(1,-1)

        # encode the image into latent space
        upscaled_latent = tiny_encode(upscaled_image,
                                      vae          = vae,
                                      tile_size    = tile_size,
                                      tile_padding = (tile_size/4),
                                      )

        # # add extra noise to the latent image
        # upscaled_latent += torch.randn_like(upscaled_latent) * extra_noise


        # generate_super_details_fn = lambda x, y, width, height: \
        #     generate_super_details(upscaled_latent,
        #                            model     = model,
        #                            seed      = seed,
        #                            cfg       = cfg,
        #                            sampler   = sampler,
        #                            sigmas    = sigmas_step,
        #                            positive  = positive,
        #                            negative  = negative,
        #                            tile_pos  = (x, y),
        #                            tile_size = (width, height)
        #                            )

        # tile_filling_zero( generate_super_details_fn )

        return ({"samples":upscaled_latent}, )

