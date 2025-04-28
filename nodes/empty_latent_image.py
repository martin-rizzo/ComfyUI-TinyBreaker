"""
File    : empty_latent_image.py
Purpose : Create an empty latent image for use in ComfyUI with PixArt.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 22, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import comfy
from   .core.system     import logger
from   .core.genparams  import GenParams
from   ._common         import get_image_size_from_genparams, \
                               DEFAULT_VAE_PATCH_SIZE


class EmptyLatentImage:
    TITLE       = "💪TB | Empty Latent Image"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "genparams": ("GENPARAMS", {"tooltip": "The generation parameters ???"}),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "generate_latents"
    RETURN_TYPES    = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)

    def generate_latents(self, genparams: GenParams):

        image_width, image_height = get_image_size_from_genparams(genparams)
        batch_size                = genparams.get_int("image.batch_size", 1)

        # calculate the latent dimensions
        latent_width    = int( image_width  // DEFAULT_VAE_PATCH_SIZE )
        latent_height   = int( image_height // DEFAULT_VAE_PATCH_SIZE )
        latent_channels = 4

        # make sure the latent dimensions are even, as required by the model
        if latent_width % 2 != 0:
            latent_width += 1
        if latent_height % 2 != 0:
            latent_height += 1

        # report the calculated dimensions and create the batch of latents
        logger.debug(f"Empty latent dimensions: {latent_width}x{latent_height} ({image_width}x{image_height})")
        latents = torch.zeros([batch_size, latent_channels, latent_height, latent_width], device=self.device)

        return ({"samples":latents}, )

