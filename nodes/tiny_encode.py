"""
File    : tiny_encode.py
Purpose : Node to encode images into a latent representation using the provided VAE.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Apr 5, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
from .xcomfy.vae            import VAE
from .xcomfy.helpers.images import tiny_encode, normalize_images


class TinyEncode:
    TITLE       = "💪TB | Tiny Encode"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Encode an image into a latent representation using the provided VAE."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "image": ("IMAGE" ,{"tooltip": "The image to be encoded.",
                               }),
            "vae"  : ("VAE"   ,{"tooltip": "The VAE model used for encoding.",
                               }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "encode"
    RETURN_TYPES    = ("LATENT",)
    RETURN_NAMES    = ("latent",)
    OUTPUT_TOOLTIPS = ("Latent representation of the input image.",)

    def encode(self, image: torch.Tensor, vae: VAE):
        tile_size    = 512
        tile_padding = 128
        image  = normalize_images(image)
        latent = tiny_encode(image,
                             vae          = vae,
                             tile_size    = tile_size,
                             tile_padding = tile_padding,
                             )
        return ({"samples": latent}, )
