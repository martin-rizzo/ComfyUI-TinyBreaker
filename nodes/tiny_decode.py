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
_TILE_SIZES        = ["128px", "256px", "512px", "768px", "1024px"]
_DEFAULT_TILE_SIZE = "512px"


class TinyDecode:
    TITLE       = "💪TB | Tiny Decode"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Decode a latent representation back into an image using the provided VAE. This node optimizes memory usage by dividing the process into smaller tiles."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "latent"   :("LATENT"    ,{"tooltip": "The latent representation to decode back into image.",
                                      }),
            "vae"      :("VAE"       ,{"tooltip": "The VAE model used for decoding the latent representation.",
                                      }),
            "tile_size":(_TILE_SIZES ,{"tooltip": "The size of the tiles used to divide the input latent into smaller chunks for processing. The value is expressed in pixels of the output image.",
                                       "default": _DEFAULT_TILE_SIZE
                                      }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "decode"
    RETURN_TYPES    = ("IMAGE",)
    RETURN_NAMES    = ("image",)
    OUTPUT_TOOLTIPS = ("The decoded image.",)

    def decode(self,
               latent         : dict,
               vae            : VAE,
               tile_size      : str | int = 512,
               overlap_percent: str | int = 100,
               ) -> tuple:

        if isinstance(tile_size, str):
            tile_size = int(tile_size.removesuffix("px"))
        if isinstance(overlap_percent, str):
            overlap_percent = int(overlap_percent.removesuffix("%"))

        image = tiny_decode(latent["samples"],
                            vae          = vae,
                            tile_size    = tile_size,
                            tile_padding = (tile_size*overlap_percent//400),
                            )

        return (normalize_images(image), )


    #__ internal functions ________________________________

    @staticmethod
    def tile_sizes():
        return [str_size for str_size in map(str, _TILE_SIZES)]
