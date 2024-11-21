"""
File    : vae_transcoder.py
Purpose : Node to transcode between different latent spaces using VAE.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 21, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .xcomfy.objects import Transcoder, VAE


class VAETranscoder:
    TITLE       = "xPixArt | VAE Transcoder"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Transcode a latent image from one latent space to another."
    
    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "samples": ("LATENT", {"tooltip": "The latent to be transcoded."}), 
            },
            "optional": {
                "transcoder": ("Transcoder", {"tooltip": "A transcoder for direct single-step transcoding."}),
                "decoder"   : ("VAE"       , {"tooltip": "A VAE model for decoding, enabling two-step transcoding."}),
                "encoder"   : ("VAE"       , {"tooltip": "A VAE model for encoding, enabling two-step transcoding."})
            }
        }

    #-- FUNCTION --------------------------------#
    FUNCTION = "transcode"
    RETURN_TYPES    = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The transcoded latent.",)

    @classmethod
    def transcode(cls,
                  samples   : dict,
                  transcoder: Transcoder = None,
                  decoder   : VAE        = None,
                  encoder   : VAE        = None
                  ) -> dict:

        if transcoder is None and (decoder is None or encoder is None):
            return samples

        if transcoder is None:
            transcoder = Transcoder.from_decoder_encoder(decoder, encoder)

        samples = transcoder( samples["samples"] )
        return ({"samples": samples}, )

