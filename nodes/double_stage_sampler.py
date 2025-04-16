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
from .core.genparams                   import GenParams
from .core.denoising_params            import DenoisingParams
from .core.comfyui_bridge.progress_bar import ProgressPreview
from .core.comfyui_bridge.model        import Model
from .core.comfyui_bridge.clip         import CLIP
from .core.comfyui_bridge.transcoder   import Transcoder


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

    def double_sampling(self,
                        latent_input : dict,
                        genparams    : GenParams,
                        model        : Model,
                        clip         : CLIP,
                        transcoder   : Transcoder,
                        refiner_model: Model,
                        refiner_clip : CLIP,
                        ):

        samples       = latent_input["samples"]
        noise_mask    = latent_input.get("noise_mask")
        batch_indexes = latent_input.get("batch_index")

        # FIRST STEP: SAMPLING USING BASE MODEL
        denoising = DenoisingParams.from_genparams(genparams, "denoising.base", model_to_sample = model)
        encoded_positive, encoded_negative = self._encode(clip, denoising.positive, denoising.negative)
        samples = self._sample(model, samples,
                               positive       = encoded_positive,
                               negative       = encoded_negative,
                               sampler_object = denoising.sampler_object,
                               sigmas         = denoising.sigmas,
                               cfg            = denoising.cfg,
                               seed           = denoising.noise_seed,
                               noise_mask     = noise_mask,
                               batch_indexes  = batch_indexes,
                               add_noise      = True,
                               progress_range = (0, 50),
                               )

        # INTERMEDIATE STEP: TRANSCODER
        # only if transcoder is provided
        if transcoder:
            samples = transcoder( samples )

        # SECOND STEP: SAMPLING USING REFINER MODEL
        # only if refiner model and refiner clip are provided
        if refiner_model and refiner_clip:
            denoising = DenoisingParams.from_genparams(genparams, "denoising.refiner", model_to_sample = refiner_model)
            encoded_positive, encoded_negative = self._encode(refiner_clip, denoising.positive, denoising.negative)
            samples = self._sample(refiner_model, samples,
                                   positive       = encoded_positive,
                                   negative       = encoded_negative,
                                   sampler_object = denoising.sampler_object,
                                   sigmas         = denoising.sigmas,
                                   cfg            = denoising.cfg,
                                   seed           = denoising.noise_seed,
                                   noise_mask     = noise_mask,
                                   batch_indexes  = batch_indexes,
                                   add_noise      = True,
                                   progress_range = (50, 100),
                                   )

        latent_output = latent_input.copy()
        latent_output["samples"] = samples
        return (latent_output, )


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
    def _sample(model        : Model,
                latent_images: torch.Tensor,
                /,*,
                positive        : torch.Tensor,
                negative        : torch.Tensor,
                sampler_object  : comfy.samplers.KSAMPLER,
                sigmas          : torch.Tensor,
                cfg             : float,
                seed            : int,
                noise_mask      : torch.Tensor | None,
                batch_indexes   : list | None,
                add_noise       : bool,
                progress_range  : tuple[int, int],
                ) -> torch.Tensor:
        """
        A custom function for sampling latent images using a given model.
        (it is a simple wrapper around `comfy.sample.sample_custom`)

        Args:
            model         : The model to use for sampling.
            latent_images : The initial latent images to sample.
            positive      : The positive embedding to use for sampling.
            negative      : The negative embedding to use for sampling.
            sampler_object: The `KSAMPLER` object.
            sigmas        : The sigmas to use in each step.
            cfg           : The classifier-free guidance scale.
            seed          : The seed used to generate the noise.
            noise_mask    : A mask to apply to the noise image to avoid sampling in the masked area.
            batch_indexes : The indexes of the batches to process when processing multiple images.
            add_noise     : Whether to add noise to the provided latent image.
            progress_range: The range of the progress bar to use for this task.

        Returns:
            The sampled latent images.
        """
        latent_images = comfy.sample.fix_empty_latent_channels(model, latent_images)
        prog_min      = progress_range[0]
        prog_max      = progress_range[1]

        if add_noise:
            noise = comfy.sample.prepare_noise(latent_images, seed, batch_indexes)
        else:
            noise = torch.zeros(latent_images.shape, dtype=latent_images.dtype, layout=latent_images.layout, device="cpu")

        total_steps      = sigmas.shape[-1] - 1
        progress_preview = ProgressPreview.from_comfyui(model, 100)
        latent_images    = comfy.sample.sample_custom(
                            model,
                            noise,
                            cfg,
                            sampler_object,
                            sigmas,
                            positive,
                            negative,
                            latent_images,
                            noise_mask   = noise_mask,
                            callback     = ProgressPreview(total_steps, parent=(progress_preview,prog_min,prog_max)),
                            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED,
                            seed         = seed
                            )
        return latent_images



