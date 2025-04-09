"""
File    : unpack_sampler_params.py
Desc    : Node that unpacks the sampling parameters from the `gparams` line.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 19, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .core.denoising_params import DenoisingParams


class UpackSamplerParams:
    TITLE = "💪TB | Unpack Sampler Params"
    CATEGORY = "TinyBreaker"
    DESCRIPTION = "Unpacks the generation parameters from a `genparams` line into separate output values."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "genparams":("GENPARAMS", {"tooltip": "The generation parameters to be updated.",
                                       }),
            "prefix"   :("STRING"   , {"tooltip": "The prefix used to identify the unpacked parameters.",
                                       "default": "base"
                                       }),
            "model"    :("MODEL"    , {"tooltip": "The model to use for generation."
                                       }),
            "clip"     :("CLIP"     , {"tooltip": "The CLIP model used for encoding the prompts."
                                       }),
            }
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "unpack"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "SAMPLER", "SIGMAS", "FLOAT", "INT"       )
    RETURN_NAMES = ("model", "positive"    , "negative"    , "sampler", "sigmas", "cfg"  , "noise_seed")

    def unpack(self, prefix, model, clip, genparams=None):
        denoising = DenoisingParams.from_genparams(genparams, f"denoising.{prefix}", model_to_sample=model)
        positive, negative = self._encode(clip, denoising.positive, denoising.negative)
        return (model, positive, negative, denoising.sampler_object, denoising.sigmas, denoising.cfg, denoising.noise_seed)


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


