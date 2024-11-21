"""
File    : multisampler.py
Purpose : Node that group two ksamples into one.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 21, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""


class MultiSampler:
    TITLE       = "xPixArt | Multi Sampler"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Group two ksamples into one, both preconfigured with the best settings."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model"           : ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "positive"        : ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative"        : ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "transcoder"      : ("VAE", {"tooltip": "The VAE model used for transcoding the latent from model to refiner."}),
                "refiner_model"   : ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "refiner_positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "refiner_negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "seed"            : ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "cfg"             : ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "latent_image"    : ("LATENT", {"tooltip": "The latent image to denoise."}),
            },
        }

    #-- FUNCTION --------------------------------#
    FUNCTION = "sample"
    RETURN_TYPES    = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)