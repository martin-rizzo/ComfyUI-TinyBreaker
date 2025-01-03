"""
File    : vae_transcode.py
Purpose : Node to transcode between different latent spaces.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 21, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .xcomfy.transcoder import Transcoder


class VAETranscode:
    TITLE       = "xPixArt | VAE Transcode"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Transcode a latent image from one latent space to another."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "samples"   : ("LATENT"    , {"tooltip": "The latent to be transcoded."}), 
            },
            "optional": {
                "transcoder": ("TRANSCODER", {"tooltip": "The transcoder to use for the processing."}),
            },
        }

    #-- FUNCTION --------------------------------#
    FUNCTION = "transcode"
    RETURN_TYPES    = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The transcoded latent.",)

    @classmethod
    def transcode(cls,
                  samples   : dict,
                  transcoder: Transcoder = None,
                  ) -> dict:

        if transcoder is None:
            return samples

        # `unet_compatible_latents=False` specifies that the latent images
        # passed to the transcoder are in their raw, unnormalized form.
        # This follows ComfyUI's convention where scaling and shifting
        # (using `scale_factor`/`shift_factor`) are applied by the
        # samplers **immediately before** the UNet processes the latents.
        latents = samples["samples"]
        latents = transcoder( latents )
        # latents = transcoder( latents, unet_compatible_latents=False )
        return ({"samples": latents}, )

