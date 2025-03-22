"""
File    : tiny_upscale_experimental.py
Purpose : Node to upscale an image
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 20, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import comfy.sample
import comfy.samplers
from .xcomfy.model          import Model
from .xcomfy.vae            import VAE
from .xcomfy.helpers.sigmas import calculate_sigmas, split_sigmas
from .xcomfy.helpers.images import normalize_images, upscale_images


class TinyUpscaleExperimental:
    TITLE       = "💪TB | Tiny Upscale (Experimental)"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Upscale an image using an experimental method."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "image"    :("IMAGE"         , {"tooltip": "The image to upscale.",
                                           }),
            "model"    :("MODEL"         , {"tooltip": "The model to use for the upscale.",
                                           }),
            "vae"      :("VAE"           , {"tooltip": "The VAE to use for the upscale.",
                                           }),
            "positive" :("CONDITIONING"  , {"tooltip": "The positive conditioning to use for the upscale.",
                                           }),
            "negative" :("CONDITIONING"  , {"tooltip": "The negative conditioning to use for the upscale.",
                                           }),
            "seed"     :("INT"           , {"tooltip": "The random seed used for creating the noise.",
                                            "default": 0, "min": 0, "max": 0xffffffffffffffff,
                                            "control_after_generate": True,
                                           }),
            "steps"    :("INT"           , {"tooltip": "The number of steps used in the denoising process.",
                                            "default": 5, "min": 1, "max": 50,
                                           }),
            "cfg"      :("FLOAT"         , {"tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
                                            "default": 3.0, "min": 0.0, "max": 15.0, "step":0.1, "round": 0.01,
                                           }),
            "sampler"  :(cls._samplers() , {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output.",
                                            "default": "dpmpp_2m",
                                           }),
            "scheduler":(cls._schedulers(),{"tooltip": "The scheduler controls how noise is gradually removed to form the image.",
                                            "default": "karras",
                                           }),
            "strength" :("FLOAT"         , {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.02,
                                           })

            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "upscale"
    RETURN_TYPES    = ("IMAGE","IMAGE")
    RETURN_NAMES    = ("image","preview")
    OUTPUT_TOOLTIPS = ("The upscaled image.","A preview of the upscaled image without denoising methods applied.")

    def upscale(self,
                image,
                model: Model,
                vae  : VAE,
                positive,
                negative,
                seed: int,
                steps: int,
                cfg: float,
                sampler: str,
                scheduler: str,
                strength: float
                ):

        # adjust the model sampling parameters to match the denoising strength
        total_steps = int(steps / strength)
        steps_start = total_steps - steps

        # calculate sigmas
        model_sampling = model.get_model_object("model_sampling")
        sigmas = calculate_sigmas(model_sampling, sampler, scheduler, total_steps, steps_start)

        # get sampler object
        sampler_object = comfy.samplers.sampler_object(sampler)

        upscaled_image, preview_image \
              = self._upscale(image,
                              scale_by       = 3,
                              model          = model,
                              vae            = vae,
                              sampler_object = sampler_object,
                              sigmas         = sigmas,
                              cfg            = cfg,
                              seed           = seed,
                              positive       = positive,
                              negative       = negative,
                              )

        return (upscaled_image, preview_image)


    #__ internal functions ________________________________

    @staticmethod
    def _samplers():
        return comfy.samplers.KSampler.SAMPLERS

    @staticmethod
    def _schedulers():
        return comfy.samplers.KSampler.SCHEDULERS

    @staticmethod
    def _upscale(images: torch.Tensor,
                 *,
                 scale_by      : int,
                 model         : Model,
                 vae           : VAE,
                 sampler_object: object,
                 sigmas        : torch.Tensor,
                 cfg           : float,
                 seed          : int,
                 positive,
                 negative,
                ) -> torch.Tensor:

        images = normalize_images(images)

        upscaled_width  = images.shape[-3] * scale_by
        upscaled_height = images.shape[-2] * scale_by
        upscaled_images = upscale_images(images, upscaled_width, upscaled_height)
        preview_images  = upscaled_images

        batch_indices = None
        latent_images = TinyUpscaleExperimental._vae_encode(vae, upscaled_images)

        noise = comfy.sample.prepare_noise(latent_images, seed, batch_indices)

        latent_images = comfy.sample.sample_custom(model,
                                                    noise,
                                                    cfg,
                                                    sampler_object,
                                                    sigmas,
                                                    positive,
                                                    negative,
                                                    latent_images,
                                                    noise_mask = None,
                                                    callback   = None,
                                                    disable_pbar = True,
                                                    seed         = seed)

        # for sigma_value in sigmas:
        #     print("##>> sigma_value", sigma_value)
        #     latent_images = comfy.sample.sample_custom(model,
        #                                                noise,
        #                                                cfg,
        #                                                sampler_object,
        #                                                torch.tensor([sigma_value]),
        #                                                positive,
        #                                                negative,
        #                                                latent_images,
        #                                                noise_mask = None,
        #                                                callback   = None,
        #                                                disable_pbar = True,
        #                                                seed         = seed)
        #     noise.fill_(0)


        upscaled_images = TinyUpscaleExperimental._vae_decode(vae, latent_images)
        return upscaled_images, preview_images


    @staticmethod
    def _vae_encode(vae   : VAE,
                    images: torch.Tensor
                    ):
        rgb_images    = images[ :, :, :, :3 ]
        latent_images = vae.encode(rgb_images)
        return latent_images


    @staticmethod
    def _vae_decode(vae          : VAE,
                    latent_images: torch.Tensor
                    ):
        images = vae.decode(latent_images)
        if len(images.shape) > 4:
            images = torch.flatten(images, start_dim=0, end_dim=-4)
        return images
