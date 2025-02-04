"""
File    : transcode_latent.py
Purpose : Node to transcode images between different latent spaces.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 21, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .xcomfy.transcoder import Transcoder


class TranscodeLatent:
    TITLE       = "💪TB | Transcode Latent"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Transcode a latent image from one latent space to another."

    #__ PARAMETERS ________________________________________
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

    #__ FUNCTION __________________________________________
    FUNCTION = "transcode"
    RETURN_TYPES    = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The transcoded latent.",)

    def transcode(self, samples: dict, transcoder: Transcoder = None) -> tuple:

        if transcoder is None:
            return samples

        latents = samples["samples"]
        latents = transcoder( latents )
        return ({"samples": latents}, )

