"""
File    : double_stage_sampler.py
Purpose : Node for denoising a latent image in two stages (base and refinement).
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
import torch
import comfy.utils
import comfy.sample
import comfy.samplers
import latent_preview
from ._denoising_params import DenoisingParams


class DoubleStageSampler:
    TITLE       = "💪TB | Double Stage Sampler"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Denoise the latent image in two stages (base and refiner)"

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_input" : ("LATENT"      , {"tooltip": "The latent image to denoise."}),
                "genparams"    : ("GENPARAMS"   , {"tooltip": "The generation parameters containing the sampler configuration."}),
                "model"        : ("MODEL"       , {"tooltip": "The model used for denoising the latent images."}),
                "clip"         : ("CLIP"        , {"tooltip": "The T5 encoder used for embedding the prompts."}),
                "transcoder"   : ("TRANSCODER"  , {"tooltip": "The transcoder model used for converting latent images from base to refiner."}),
                "refiner_model": ("MODEL"       , {"tooltip": "The model used for refining latent images."}),
                "refiner_clip" : ("CLIP"        , {"tooltip": "The CLIP model used for embedding text prompts during refining."}),
            }
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "double_sampling"
    RETURN_TYPES    = ("LATENT",)
    RETURN_NAMES    = ("latent_output",)
    OUTPUT_TOOLTIPS = ("Latent image after denoising.",)

    def double_sampling(self, latent_input, genparams, model, clip, transcoder, refiner_model, refiner_clip):

        # first step: base model
        denoising = DenoisingParams.from_genparams(genparams, "denoising.base", model_to_sample = model)
        encoded_positive, encoded_negative = self._encode(clip, denoising.positive, denoising.negative)
        latents = self._sample(model,
                               positive   = encoded_positive,
                               negative   = encoded_negative,
                               sampler    = denoising.sampler_object,
                               sigmas     = denoising.sigmas,
                               cfg        = denoising.cfg,
                               noise_seed = denoising.noise_seed,
                               latent     = latent_input,
                               add_noise  = True)

        # intermediate step: transcoder
        if transcoder is not None:
            latents["samples"] = transcoder( latents["samples"] )
            #latents = {"samples": latents}

        # if not refiner, return the result of the base model
        if (refiner_model is None) or (refiner_clip is None):
            return (latents, )

        # second step: refiner model
        denoising = DenoisingParams.from_genparams(genparams, "denoising.refiner", model_to_sample = refiner_model)
        encoded_positive, encoded_negative = self._encode(refiner_clip, denoising.positive, denoising.negative)
        latents = self._sample(refiner_model,
                               positive   = encoded_positive,
                               negative   = encoded_negative,
                               sampler    = denoising.sampler_object,
                               sigmas     = denoising.sigmas,
                               cfg        = denoising.cfg,
                               noise_seed = denoising.noise_seed,
                               latent     = latents,
                               add_noise  = True)

        return (latents, )


    #__ internal functions ________________________________

    @staticmethod
    def _encode(clip, positive, negative):
        if isinstance(positive,str):
            tokens = clip.tokenize(positive)
            positive = clip.encode_from_tokens_scheduled(tokens)
        if isinstance(negative,str):
            tokens = clip.tokenize(negative)
            negative = clip.encode_from_tokens_scheduled(tokens)
        return positive, negative


    @staticmethod
    def _sample(model,
                positive  : torch.Tensor,
                negative  : torch.Tensor,
                sampler   : comfy.samplers.KSAMPLER,
                sigmas    : torch.Tensor,
                cfg       : float,
                noise_seed: int,
                latent    : torch.Tensor,
                add_noise : bool
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        A custom version of the `sample` function from the node `SamplerCustom` from ComfyUI.
        https://github.com/comfyanonymous/ComfyUI/blob/v0.3.12/comfy_extras/nodes_custom_sampler.py#L474

        Args:
            model     : The model to use for sampling.
            positive  : The positive embedding to use for sampling.
            negative  : The negative embedding to use for sampling.
            sampler   : The `KSAMPLER` object.
            sigmas    : The sigmas to use in each step.
            cfg       : The classifier-free guidance scale.
            noise_seed: The seed used to generate the noise.
            latent    : The initial latent image.
            add_noise : Whether to add noise to the provided latent image.
        """
        batch_inds   = latent.get("batch_index")
        noise_mask   = latent.get("noise_mask")
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent["samples"])

        if add_noise:
            noise = comfy.sample.prepare_noise(latent_image, noise_seed, batch_inds)
        else:
            noise = torch.zeros(latent_image.shape, dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")

        x0_output = {}
        callback  = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
        samples   = comfy.sample.sample_custom(model,
                                               noise,
                                               cfg,
                                               sampler,
                                               sigmas,
                                               positive,
                                               negative,
                                               latent_image,
                                               noise_mask   = noise_mask,
                                               callback     = callback,
                                               disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED,
                                               seed         = noise_seed)

        latent = latent.copy()
        latent["samples"] = samples
        return latent



