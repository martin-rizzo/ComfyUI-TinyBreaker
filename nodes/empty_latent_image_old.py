"""
File    : empty_latent_image_old.py
Purpose : Create an empty latent image for use in ComfyUI with PixArt.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 14, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import comfy
from   .utils.system import logger
MAX_RESOLUTION=16384
LATENT_SCALE_FACTOR=8

# 1048576

LANDSCAPE_SIZE_BY_ASPECT_RATIO = {
    "1:1 square"      : (1024.0, 1024.0),
    "4:3 tv"          : (1182.4,  886.8),
    "48:35 (35 mm)"   : (1199.2,  874.4),
    "71:50 ~IMAX"     : (1220.2,  859.3),
    "3:2 photo"       : (1254.1,  836.1),
    "16:10 wide"      : (1295.3,  809.5),
    "16:9 hdtv"       : (1365.3,  768.0),
    "2:1 mobile"      : (1448.2,  724.0),
    "21:9 ultrawide"  : (1564.2,  670.4),
    "12:5 anamorphic" : (1586.4,  661.0),
    "70:27 cinerama"  : (1648.8,  636.0),
    "32:9 s.ultrawide": (1930.9,  543.0),
}

class EmptyLatentImageOld:
    TITLE       = "💪TB | Empty Latent Image [Old Version]"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."


    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "landscape" : ("BOOLEAN", {"default": True}),
                "ratio"     : (list(LANDSCAPE_SIZE_BY_ASPECT_RATIO.keys()),),
                "scale"     : ("FLOAT", {"default": 1.0, "min": 0.90, "max": 1.24, "step": 0.02}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
                },
            }

    #__ FUNCTION __________________________________________
    FUNCTION = "generate"
    RETURN_TYPES    = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)


    def generate(self, landscape, ratio, scale=1.0, batch_size=1):

        # get the aspect ratio dimensions from the dictionary
        width, height = LANDSCAPE_SIZE_BY_ASPECT_RATIO[ratio]

        # swap the width and height for portrait mode
        if not landscape:
            width, height = height, width

        # calculate the latent image dimensions based on the scale factor and aspect ratio
        latent_width  = int(  width * scale // LATENT_SCALE_FACTOR )
        latent_height = int( height * scale // LATENT_SCALE_FACTOR )

        # make sure the dimensions are even, as required by the model
        if latent_width % 2 != 0:
            latent_width += 1
        if latent_height % 2 != 0:
            latent_height += 1

        image_width, image_height = latent_width * LATENT_SCALE_FACTOR, latent_height * LATENT_SCALE_FACTOR
        logger.debug(f"Empty latent dimensions: {latent_width}x{latent_height} ({image_width}x{image_height})")

        # create a batch of empty latent images
        latents = torch.zeros([batch_size, 4, latent_height, latent_width],
                              device=self.device
                              )
        return ({"samples":latents}, )
