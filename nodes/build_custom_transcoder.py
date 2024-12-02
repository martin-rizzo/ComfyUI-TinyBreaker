"""
File    : build_custom_transcoder.py
Purpose : Node for building custom transcoders from two VAEs.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 22, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
from .xcomfy.transcoder import Transcoder

ENHANCER_OPS = [
    "None",
    "Auto",
    "Blur",
    #"Color Balance",
]
    
class BuildCustomTranscoder:
    TITLE       = "xPixArt | Build Custom Transcoder"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Builds a custom transcoder from two VAEs."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoder"       : ("VAE"       , {"tooltip": "A VAE model used for decoding in the first step."}),
                "encoder"       : ("VAE"       , {"tooltip": "A VAE model used for encoding in the second step."}),
                "enhancer_op"   : (ENHANCER_OPS, {"default": "None"}, {"tooltip": "The operation to apply after decode but before encode."}),
                "enhancer_level": ("FLOAT"     , {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}, {"tooltip": "The strength of the enhancer operation"}),
            },
        }

    #-- FUNCTION --------------------------------#
    FUNCTION = "build_transcoder"
    RETURN_TYPES    = ("TRANSCODER",)
    OUTPUT_TOOLTIPS = ("A custom transcoder.",)

    @classmethod
    def build_transcoder(cls,  decoder, encoder, enhancer_op, enhancer_level):
        gaussian_blur_sigma = 0.0

        if enhancer_op == "Auto":
            # TODO: set blur 0.5 only when the encoder/decoder are `Tiny Autoencoders`
            gaussian_blur_sigma = 0.5

        elif enhancer_op == "Blur":
            gaussian_blur_sigma = enhancer_level

        transcoder = Transcoder.from_decoder_encoder(decoder = decoder,
                                                     encoder = encoder,
                                                     gaussian_blur_sigma = gaussian_blur_sigma)

        return (transcoder,)
