"""
File    : vae_transcode_two_steps.py
Purpose : Node for transcoding between different latent spaces in two steps (decoding+encoding)
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 21, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
from .xcomfy.transcoder import Transcoder
from .xcomfy.vae        import VAE


class VAETranscodeTwoSteps:
    TITLE       = "xPixArt | VAE Transcode in Two Steps"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Transcode a latent image from one latent space to another in two steps (decoding+encoding)."
    
    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "samples"    : ("LATENT", {"tooltip": "The latent to be transcoded."}), 
                "blur_level" : ("FLOAT" , {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}, {"tooltip": "The blur level to fix artifacts between decoding and encoding."}),
            },
            "optional": {
                "decoder"    : ("VAE"   , {"tooltip": "A VAE model used for decoding in the first step."}),
                "encoder"    : ("VAE"   , {"tooltip": "A VAE model used for encoding in the second step."}),
            }
        }

    #-- FUNCTION --------------------------------#
    FUNCTION = "transcode"
    RETURN_TYPES    = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The transcoded latent.",)

    @classmethod
    def transcode(cls,
                  samples   : dict,
                  blur_level: float,
                  decoder   : VAE = None,
                  encoder   : VAE = None
                  ) -> dict:

        if not decoder or not encoder:
            return samples

        transcoder = Transcoder.from_decoder_encoder(decoder, encoder, gaussian_blur_level=blur_level)
        samples    = transcoder( samples["samples"] )
        return ({"samples": samples}, )

