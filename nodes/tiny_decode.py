"""
File    : tiny_decode.py
Purpose : Node to decode latent representations back into images using the provided VAE.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Apr 6, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.comfyui_bridge.vae             import VAE
from .core.comfyui_bridge.helpers.images  import normalize_images
from .core.tiny_encode_decode             import tiny_decode


class TinyDecode:
    TITLE       = "💪TB | Tiny Decode"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Decode a latent representation back into an image using the provided VAE."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "latent": ("LATENT" ,{"tooltip": "The latent representation to decode back into image.",
                                 }),
            "vae"   : ("VAE"    ,{"tooltip": "The VAE model used for decoding the latent representation.",
                                 }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "decode"
    RETURN_TYPES    = ("IMAGE",)
    RETURN_NAMES    = ("image",)
    OUTPUT_TOOLTIPS = ("The decoded image.",)

    def decode(self, latent, vae: VAE):
        tile_size    = 512
        tile_padding = 128
        image = tiny_decode(latent["samples"],
                            vae          = vae,
                            tile_size    = tile_size,
                            tile_padding = tile_padding,
                            )
        image = normalize_images(image)
        return (image, )
