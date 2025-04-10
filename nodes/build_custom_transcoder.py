"""
File    : build_custom_transcoder.py
Purpose : Node for building custom transcoders from two VAEs.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 22, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.comfyui_bridge.transcoder import Transcoder

ENHANCER_OPS = [
    "None",
    "Auto",
    "Blur",
]

class BuildCustomTranscoder:
    TITLE       = "💪TB | Build Custom Transcoder"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Builds a custom transcoder using two Variational Autoencoders (VAEs). They are combined to create an integrated transcoder that can convert between different latent spaces."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_vae"    : ("VAE"       , {"tooltip": "VAE model of the source latent space in the conversion. (This VAE will be used as the decoder)"}),
                "target_vae"    : ("VAE"       , {"tooltip": "VAE model of the target latent space in the conversion. (This VAE will be used as the encoder)"}),
                "enhancer_op"   : (ENHANCER_OPS, {"default": "None"}, {"tooltip": "The operation to apply after decode but before encode."}),
                "enhancer_level": ("FLOAT"     , {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}, {"tooltip": "The strength of the enhancer operation"}),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "build_transcoder"
    RETURN_TYPES    = ("TRANSCODER",)
    OUTPUT_TOOLTIPS = ("A custom transcoder.",)

    @classmethod
    def build_transcoder(cls,  source_vae, target_vae, enhancer_op, enhancer_level):
        gaussian_blur_sigma = 0.0

        if enhancer_op == "Auto":
            # TODO: set blur 0.5 only when the encoder/decoder are `Tiny Autoencoders`
            gaussian_blur_sigma = 0.5

        elif enhancer_op == "Blur":
            gaussian_blur_sigma = enhancer_level

        transcoder = Transcoder.from_decoder_encoder(decoder = source_vae,
                                                     encoder = target_vae,
                                                     gaussian_blur_sigma = gaussian_blur_sigma)

        return (transcoder,)
