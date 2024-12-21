"""
File    : gparams_unpacker.py
Desc    : Node that unpacks the sampling parameters from the `gparams` line.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Dec 19, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import comfy.samplers
from .xcomfy.model import Model
_DISCARD_PENULTIMATE_SIGMA_SAMPLERS = comfy.samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS


DEFAULT_G_PARAMS = {
    "base.positive"       : "",
    "base.negative"       : "",
    "base.steps"          : 12,
    "base.cfg"            : 3.4,
    "base.sampler_name"   : "uni_pc",
    "base.scheduler"      : "simple",
    "base.start_at_step"  : 2,

    "refiner.positive"     : "",
    "refiner.negative"     : "",
    "refiner.steps"        : 11,
    "refiner.cfg"          : 2.0,
    "refiner.sampler_name" : "deis",
    "refiner.scheduler"    : "ddim_uniform",
    "refiner.start_at_step": 6,
}

_DEFAULT_POSITIVE      = ""
_DEFAULT_NEGATIVE      = ""
_DEFAULT_STEPS         = 12
_DEFAULT_CFG           = 3.5
_DEFAULT_SAMPLER_NAME  = "euler"
_DEFAULT_SCHEDULER     = "normal"
_DEFAULT_START_AT_STEP = 0



class GParamsUnpacker:
    TITLE = "xPixArt | GParams Unpacker"
    CATEGORY = "xPixArt"
    DESCRIPTION = ""

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"tooltip": "The prefix used to identify the unpacked parameters."}),
                "model" : ("MODEL" , {"tooltip": "The model to use for generation."}),
                "clip"  : ("CLIP"  , {"tooltip": "The CLIP model used for encoding the prompts."})
            },
            "optional": {
                "gparams": ("GPARAMS", {"tooltip": "The generation parameters to unpack."}),
            }
        }

    #-- FUNCTION --------------------------------#
    FUNCTION = "unpack"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "SAMPLER", "SIGMAS", "FLOAT", "GPARAMS")
    RETURN_NAMES = ("model", "positive"    , "negative"    , "sampler", "sigmas", "cfg"  , "gparams")

    def unpack(self, prefix, model, clip, gparams=None):

        print()
        print("##>> gparams:", gparams)
        print()

        # use default values if no gparams are provided
        if not gparams:
            gparams = DEFAULT_G_PARAMS

        # ensure that prefix always ends with a period '.'
        if prefix and not prefix.endswith('.'):
            prefix += '.'

        positive      = str(  gparams.get(f"{prefix}prompt"       , _DEFAULT_POSITIVE     ))
        negative      = str(  gparams.get(f"{prefix}negative"     , _DEFAULT_NEGATIVE     ))
        steps         = int(  gparams.get(f"{prefix}steps"        , _DEFAULT_STEPS        ))
        cfg           = float(gparams.get(f"{prefix}cfg"          , _DEFAULT_CFG          ))
        sampler_name  = str(  gparams.get(f"{prefix}sampler_name" , _DEFAULT_SAMPLER_NAME ))
        scheduler     = str(  gparams.get(f"{prefix}scheduler"    , _DEFAULT_SCHEDULER    ))
        start_at_step = int(  gparams.get(f"{prefix}start_at_step", _DEFAULT_START_AT_STEP))
        discard_penultimate_sigma = sampler_name in _DISCARD_PENULTIMATE_SIGMA_SAMPLERS

        print("#>>--------")
        print("#>> positive:", positive)
        print("#>> negative:", negative)
        print("#>>--------")

        positive, negative = self._encode(clip, positive, negative)
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas  = self._calculate_sigmas(model, scheduler, steps, start_at_step, discard_penultimate_sigma)
        return (model, positive, negative, sampler, sigmas, cfg, gparams)



    @staticmethod
    def _calculate_sigmas(model                    : Model,
                          scheduler                : str,
                          steps                    : int,
                          start_at_step            : int  = 0,
                          discard_penultimate_sigma: bool = False,
                          ) -> torch.Tensor:
        """Calculates the sigma values for a given model and scheduler."""

        # if no steps are specified, return an empty tensor
        # this is useful when no sampling should be performed
        if steps <= 0:
            return torch.FloatTensor([])

        # ensure start_at_step is within valid bounds
        start_at_step = min(start_at_step, steps-1)

        # if discard was requested,
        # first calculate all sigmas (plus one) and then discard the penultimate one
        if discard_penultimate_sigma:
            sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps+1).cpu()
            sigmas = torch.cat((sigmas[:-2], sigmas[-1:]))
        else:
            sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).cpu()

        # discard all sigmas before the specified start step
        if start_at_step>0:
            sigmas = sigmas[start_at_step:]

        return sigmas


    @staticmethod
    def _encode(clip, positive, negative):
        if isinstance(positive,str):
            tokens = clip.tokenize(positive)
            positive = clip.encode_from_tokens_scheduled(tokens)
        if isinstance(negative,str):
            tokens = clip.tokenize(negative)
            negative = clip.encode_from_tokens_scheduled(tokens)
        return positive, negative
