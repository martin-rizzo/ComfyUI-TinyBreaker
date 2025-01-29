"""
File    : empty_latent_image.py
Purpose : Create an empty latent image for use in ComfyUI with PixArt.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 22, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ConfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import math
import torch
import comfy
from   .utils.system   import logger
from   .core.genparams import GenParams, normalize_prefix
MAX_RESOLUTION=16384
LATENT_SCALE_FACTOR=8

_PREDEFINED_SCALES = {
    "small"  : 0.82,
    "normal" : 1.0,
    "large"  : 1.22,
}
_DEFAULT_SCALE        = 1.0
_DEFAULT_ASPECT_RATIO = "1:1"
_DEFAULT_BATCH_SIZE   = 1


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
        prefix = normalize_prefix("image")

        resolution  = 1024
        scale       = genparams.get    ( f"{prefix}scale"        , _DEFAULT_SCALE       )
        ratio       = genparams.get    ( f"{prefix}aspect_ratio" , _DEFAULT_ASPECT_RATIO)
        batch_size  = genparams.get_int( f"{prefix}batch_size"   , _DEFAULT_BATCH_SIZE  )
        orientation = genparams.get    ( f"{prefix}orientation"  , None                 )

        scale                              = self._parse_scale(scale)
        ratio_numerator, ratio_denominator = self._parse_ratio(ratio)

        # orientation overrides the aspect ratio dimensions
        if isinstance(orientation, str):
            orientation = orientation.lower()
            if   orientation == "portrait"  and ratio_numerator > ratio_denominator:
                ratio_numerator, ratio_denominator = ratio_denominator, ratio_numerator
            elif orientation == "landscape" and ratio_numerator < ratio_denominator:
                ratio_numerator, ratio_denominator = ratio_denominator, ratio_numerator

        latents = self._generate_latents(resolution, scale, ratio_numerator, ratio_denominator, batch_size, device=self.device)
        return ({"samples":latents}, )


    #__ internal functions ________________________________

    @staticmethod
    def _generate_latents(resolution, scale, ratio_numerator, ratio_denominator, batch_size, device):

        # calculate the image dimensions based on the resolution and aspect ratio
        desired_area = resolution * resolution
        width        = math.sqrt(desired_area * ratio_numerator / ratio_denominator)
        height       = width * ratio_denominator / ratio_numerator

        # calculate the latent dimensions
        latent_width    = int(  width * scale // LATENT_SCALE_FACTOR )
        latent_height   = int( height * scale // LATENT_SCALE_FACTOR )
        latent_channels = 4

        # make sure the latent dimensions are even, as required by the model
        if latent_width % 2 != 0:
            latent_width += 1
        if latent_height % 2 != 0:
            latent_height += 1

        # report the calculated dimensions and create the batch of latents
        image_width, image_height = latent_width * LATENT_SCALE_FACTOR, latent_height * LATENT_SCALE_FACTOR
        logger.debug(f"Empty latent dimensions: {latent_width}x{latent_height} ({image_width}x{image_height})")
        return torch.zeros([batch_size, latent_channels, latent_height, latent_width], device=device)


    @staticmethod
    def _parse_scale(scale):
        if isinstance(scale, float):
            return scale

        elif scale in _PREDEFINED_SCALES:
            return _PREDEFINED_SCALES[scale]

        elif isinstance(scale, str):
            try:
                return float(scale)
            except ValueError:
                pass

        # if none of the above conditions are met, use the default value
        logger.debug(f"Invalid scale factor: '{scale}'. Using default value.")
        return 1.0


    @staticmethod
    def _parse_ratio(ratio) -> tuple:

        # asume the string is a fraction with the format "numerator:denominator"
        if isinstance(ratio, str) and ':' in ratio:
            numerator, denominator = ratio.split(':',1)
            try:
                return int(numerator), int(denominator)
            except ValueError as e:
                pass

        # asume the string is a decimal number representing the width/height ratio
        elif isinstance(ratio, str) and '.' in ratio:
            try:
                return float(ratio), 1
            except ValueError as e:
                pass

        # asume the ratio is a float number representing the width/height ratio
        elif isinstance(ratio, float):
            return ratio, 1

        # if none of the above conditions are met, use the default value
        logger.debug(f"Invalid aspect ratio: '{ratio}'. Using default value.")
        return 1, 1
