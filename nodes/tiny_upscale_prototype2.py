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
import torch.nn.functional as F
from .xcomfy.helpers.images import normalize_images, tiny_encode


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

    def upscale(self, image, vae, upscale_by):
        image = normalize_images(image)
        batch_size, image_height, image_width, channels = image.shape

        upscaled_width  = int( round(image_width  * upscale_by) )
        upscaled_height = int( round(image_height * upscale_by) )
        upscaled_image  = F.interpolate(image.transpose(1,-1),
                                        size = (upscaled_width, upscaled_height),
                                        mode = "bilinear").transpose(1,-1)

        upscaled_latent = tiny_encode(upscaled_image, vae, tile_size=64, tile_padding=6)
        return ({"samples":upscaled_latent}, )

