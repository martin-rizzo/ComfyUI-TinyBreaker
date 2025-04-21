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
from .core.comfyui_bridge.vae             import VAE
from .core.comfyui_bridge.helpers.images  import normalize_images
from .core.tiny_encode_decode             import tiny_encode
_TILE_SIZES        = ["128px", "256px", "384px", "512px", "640px", "768px", "1024px"]
_DEFAULT_TILE_SIZE = "512px"
_OVERLAPS          = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
_DEFAULT_OVERLAP   = "100%"


class TinyEncode:
    TITLE       = "💪TB | Tiny Encode"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Encode an image into a latent representation using the provided VAE. This node optimizes memory usage by dividing the process into smaller tiles."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "image"    :("IMAGE"     ,{"tooltip": "The image to be encoded to a latent representation.",
                                      }),
            "vae"      :("VAE"       ,{"tooltip": "The VAE model used for encoding the image.",
                                      }),
            "tile_size":(_TILE_SIZES ,{"tooltip": "The size of the tiles used to divide the image into smaller regions for processing, expressed in pixels of the input image. A lower tile size reduces memory usage but may result in lower image quality.",
                                       "default": _DEFAULT_TILE_SIZE
                                      }),
            "overlap"  :(_OVERLAPS   ,{"tooltip": "The percentage of overlap between adjacent tiles.",
                                       "default": _DEFAULT_OVERLAP,
                                      }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "encode"
    RETURN_TYPES    = ("LATENT",)
    RETURN_NAMES    = ("latent",)
    OUTPUT_TOOLTIPS = ("Latent representation of the input image.",)

    def encode(self,
               image    : torch.Tensor,
               vae      : VAE,
               tile_size: str | int = 512,
               overlap  : str | int = 100,
               ) -> tuple:

        if isinstance(tile_size, str):
            tile_size = int(tile_size.removesuffix("px"))
        if isinstance(overlap, str):
            overlap = int(overlap.removesuffix("%"))

        latent = tiny_encode(normalize_images(image),
                             vae          = vae,
                             tile_size    = tile_size,
                             tile_padding = (tile_size*overlap//400),
                             )

        return ({"samples": latent}, )

