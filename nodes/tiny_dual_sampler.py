"""
File    : tiny_dual_sampler.py
Purpose : Node for denoising latent images in two stages with caching to avoid redundant computations.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Apr 16, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import weakref
import comfy.utils
import comfy.sample
import comfy.samplers
from .core.system                      import logger
from .core.genparams                   import GenParams
from .core.denoising_params            import DenoisingParams
from .core.comfyui_bridge.progress_bar import ProgressPreview
from .core.comfyui_bridge.model        import Model
from .core.comfyui_bridge.clip         import CLIP
from .core.comfyui_bridge.transcoder   import Transcoder


class TinyDualSampler:
    TITLE       = "💪TB | Tiny Dual Sampler"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Denoises the latent image in two stages: first with a base model, then with a refiner model to add details and enhance image quality."


    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_input" : ("LATENT"     ,{"tooltip": "The latent image to denoise."}),
                "genparams"    : ("GENPARAMS"  ,{"tooltip": "The generation parameters containing the sampler configuration."}),
                "model"        : ("MODEL"      ,{"tooltip": "The model used for denoising the latent images."}),
                "clip"         : ("CLIP"       ,{"tooltip": "The T5 encoder used for embedding the prompts."}),
                "transcoder"   : ("TRANSCODER" ,{"tooltip": "The transcoder model used for converting latent images from base to refiner."}),
                "refiner_model": ("MODEL"      ,{"tooltip": "The model used for refining latent images."}),
                "refiner_clip" : ("CLIP"       ,{"tooltip": "The CLIP model used for embedding text prompts during refining."}),
                "enable_cache" : ("BOOLEAN"    ,{"tooltip": "Enable internal cache to avoid redundant computations. It might be necessary to disable it if errors occur or the image is not updating correctly.",
                                                 "default": True,
                                                }),
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
                        enable_cache : bool,
                        ):

        samples       = latent_input["samples"]
        noise_mask    = latent_input.get("noise_mask")
        batch_indexes = latent_input.get("batch_index")

        # FIRST STEP: SAMPLING USING BASE MODEL
        denoising = DenoisingParams.from_genparams(genparams, "denoising.base", model_to_sample = model)
        encoded_positive = self._cached_encode(clip, denoising.positive, cache=self.base_positive_cache)
        encoded_negative = self._cached_encode(clip, denoising.negative, cache=self.base_negative_cache)
        samples = self._cached_sample(model, samples,
                                      positive       = encoded_positive,
                                      negative       = encoded_negative,
                                      sampler        = denoising.sampler,
                                      sampler_object = denoising.sampler_object,
                                      sigmas         = denoising.sigmas,
                                      cfg            = denoising.cfg,
                                      seed           = denoising.noise_seed,
                                      noise_mask     = noise_mask,
                                      batch_indexes  = batch_indexes,
                                      transcoder     = transcoder,
                                      add_noise      = True,
                                      progress_range = (0, 50),
                                      cache          = self.base_cache if enable_cache else None,
                                      )

        # SECOND STEP: SAMPLING USING REFINER MODEL
        # only if refiner model and refiner clip are provided
        if refiner_model and refiner_clip:
            denoising = DenoisingParams.from_genparams(genparams, "denoising.refiner", model_to_sample = refiner_model)
            encoded_positive = self._cached_encode(refiner_clip, denoising.positive, cache=self.refiner_positive_cache)
            encoded_negative = self._cached_encode(refiner_clip, denoising.negative, cache=self.refiner_negative_cache)
            samples = self._cached_sample(refiner_model, samples,
                                          positive       = encoded_positive,
                                          negative       = encoded_negative,
                                          sampler        = denoising.sampler,
                                          sampler_object = denoising.sampler_object,
                                          sigmas         = denoising.sigmas,
                                          cfg            = denoising.cfg,
                                          seed           = denoising.noise_seed,
                                          noise_mask     = noise_mask,
                                          batch_indexes  = batch_indexes,
                                          transcoder     = None,
                                          add_noise      = True,
                                          progress_range = (50, 100),
                                          cache          = self.refiner_cache if enable_cache else None,
                                          )

        latent_output = latent_input.copy()
        latent_output["samples"] = samples
        return (latent_output, )


    def __init__(self):
        self.base_cache = {}
        self.base_positive_cache = {}
        self.base_negative_cache = {}
        self.refiner_cache = {}
        self.refiner_positive_cache = {}
        self.refiner_negative_cache = {}


    #__ internal functions ________________________________

    @classmethod
    def _cached_encode(cls,
                       clip: CLIP,
                       text: str,
                       /,*,
                       cache: dict | None,
                       ) -> torch.Tensor:
        """
        Encodes the given text using the provided CLIP model, utilizing a cache to avoid redundant computations.
        Args:
            clip  (CLIP): The CLIP model to use for encoding.
            text   (str): The text to encode.
            cache (dict): A dictionary to store and retrieve cached embeddings.
        Returns:
            The embedding tensor for the given text.  If the embedding is found in the cache,
            it is returned directly. Otherwise, it is computed, cached, and then returned.
        """
        # try to use the cache
        if isinstance(cache, dict) and "keys" in cache:
            keys = cache["keys"]
            if clip == keys[0] and text == keys[1]:
                logger.debug("Using cached clip embedding.")
                return cache["embeddings"]

        # if the cache does not match or is not available,
        # perform the encoding
        tokens     = clip.tokenize(text)
        embeddings = clip.encode_from_tokens_scheduled(tokens)

        # cache the results if the cache is available
        if isinstance(cache, dict):
            cache["keys"]       = (ObjectRef(clip), text)
            cache["embeddings"] = embeddings
        return embeddings


    @classmethod
    def _cached_sample(cls,
                       model        : Model,
                       latent_images: torch.Tensor,
                       /,*,
                       positive        : list,
                       negative        : list,
                       sampler         : str,
                       sampler_object  : comfy.samplers.KSAMPLER,
                       sigmas          : torch.Tensor,
                       cfg             : float,
                       seed            : int,
                       noise_mask      : torch.Tensor | None,
                       batch_indexes   : list | None,
                       transcoder      : Transcoder | None,
                       add_noise       : bool,
                       progress_range  : tuple[int, int],
                       cache           : dict | None,
                       ) -> torch.Tensor:
        """A version of the sample function that uses a cache to avoid redundant computations."""
        assert isinstance(positive, list), f"Positive must be a list, not {type(positive)}."
        assert isinstance(negative, list), f"Negative must be a list, not {type(negative)}."
        assert isinstance(sampler,  str ), f"Sampler must be a string, not {type(sampler)}."

        # try to use the cache
        if isinstance(cache, dict) and "keys" in cache:
            keys = cache["keys"]
            cache_match = [
                model                   == keys[ 0],
                torch.equal(latent_images, keys[ 1]),
                positive                is keys[ 2],
                negative                is keys[ 3],
                sampler                 == keys[ 4],
                torch.equal(sigmas,        keys[ 5]),
                cfg                     == keys[ 6],
                seed                    == keys[ 7],
                noise_mask              == keys[ 8],
                batch_indexes           == keys[ 9],
                transcoder              == keys[10],
                add_noise               == keys[11],
            ]
            for i, match in enumerate(cache_match):
                logger.debug(f"cache_match[{i}] : {match}")
            if all(cache_match):
                logger.debug("All cache matches, using cached sample.")
                return cache["samples"]

        # if the cache does not match or is not available,
        # perform the sampling operation
        samples =  cls._sample(model, latent_images,
                           positive       = positive,
                           negative       = negative,
                           sampler_object = sampler_object,
                           sigmas         = sigmas,
                           cfg            = cfg,
                           seed           = seed,
                           noise_mask     = noise_mask,
                           batch_indexes  = batch_indexes,
                           transcoder     = transcoder,
                           add_noise      = add_noise,
                           progress_range = progress_range)

        # cache the results if the cache is available
        if isinstance(cache, dict):
            cache["keys"]    = (ObjectRef(model), latent_images, positive, negative, sampler, sigmas,
                                cfg, seed, noise_mask, batch_indexes, ObjectRef(transcoder), add_noise)
            cache["samples"] = samples
        return samples


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
                transcoder      : Transcoder | None,
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
            transcoder    : An optional transcoder used at the output to convert the latent image into another space.
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

        if transcoder:
            latent_images = transcoder(latent_images)

        return latent_images



class ObjectRef:

    def __init__(self, obj: object | None):
        self.obj = weakref.ref(obj) if obj else None

    def get(self) -> object | None:
        return self.obj() if self.obj is not None else None

    def __eq__(self, obj: object | None):
        if self.obj is not None and obj is not None:
            return self.obj() is obj
        if self.obj is None and obj is None:
            return True
        return False
