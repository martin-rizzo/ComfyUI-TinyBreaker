"""
File    : prefixed_double_stage_sampler.py
Purpose : 
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Nov 22, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
import torch
import comfy.sample
import latent_preview


QUALITY_LEVEL = [
    "fast",
    "quality"
]

SAMPLER_CONFIG = {
    "fast": {
        "steps"       : 12,
        "sampler_name": "uni_pc",
        "scheduler"   : "simple",
        "start_step"  : 2,
        "last_step"   : 1000,
    },
    "quality": {
        "steps"       : 29,
        "sampler_name": "dpmpp_2m",
        "scheduler"   : "karras",
        "start_step"  : 7,
        "last_step"   : 1000,
    }
}

REFINER_CONFIG = {
    "fast": {
        "steps"       : 11,
        "cfg"         : 2.0,
        "sampler_name": "deis",
        "scheduler"   : "ddim_uniform",
        "start_step"  : 6,
        "last_step"   : 1000
    },
    "quality": {
        "steps"       : 11,
        "cfg"         : 2.0,
        "sampler_name": "dpm_2_ancestral",
        "scheduler"   : "ddim_uniform",
        "start_step"  : 7,
        "last_step"   : 1000
    }
}


def common_ksampler(model,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    latent,
                    denoise=1.0,
                    disable_noise=False,
                    start_step=None,
                    last_step=None,
                    force_full_denoise=False
                    ):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return out


class PrefixedDoubleStageSampler:
    TITLE       = "xPixArt | Prefixed Double Stage Sampler"
    CATEGORY    = "xPixArt"
    DESCRIPTION = ""

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model"       : ("MODEL"       , ),
                "positive"    : ("CONDITIONING", ),
                "negative"    : ("CONDITIONING", ),
                "latent_image": ("LATENT"      , ),
                "noise_seed"  : ("INT"         , {"default": 0  , "min": 0  , "max": 0xffffffffffffffff}),
                "cfg"         : ("FLOAT"       , {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler"     : (QUALITY_LEVEL , {"default": QUALITY_LEVEL[0]}),
                "refiner"     : (QUALITY_LEVEL , {"default": QUALITY_LEVEL[0]}),
            },
            "optional": {
                "transcoder"       : ("TRANSCODER"  , {"tooltip": "The transcoder to use for the processing."}),
                "refiner_model"    : ("MODEL"       , ),
                "refiner_positive" : ("CONDITIONING", ),
                "refiner_negative" : ("CONDITIONING", ),
                "refiner_variation": ("INT"         , {"default": 0  , "min": 0  , "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    def sample(self,
               model,
               positive,
               negative,
               latent_image,
               noise_seed,
               cfg,
               sampler,
               refiner,
               transcoder=None,
               refiner_model=None,
               refiner_positive=None,
               refiner_negative=None,
               refiner_variation=0
               ):
        
        force_full_denoise = True
        disable_noise      = False
        denoise            = 1.0

        # first step: sampler
        kwargs = SAMPLER_CONFIG[sampler].copy()
        kwargs["model"        ] = model
        kwargs["seed"         ] = noise_seed
    #   kwargs["steps"        ] = K!
        kwargs["cfg"          ] = cfg
    #   kwargs["sampler_name" ] = K!
    #   kwargs["scheduler"    ] = K!
        kwargs["positive"     ] = positive
        kwargs["negative"     ] = negative
        kwargs["latent"       ] = latent_image
        kwargs["denoise"      ] = denoise
        kwargs["disable_noise"] = disable_noise
    #   kwargs["start_step"   ] = K!
    #   kwargs["last_step"    ] = K!
        kwargs["force_full_denoise"] = force_full_denoise # <- no hace falta!
        samples = common_ksampler( **kwargs )

        # intermediate step: transcoder
        if transcoder is not None:
            samples = transcoder( samples["samples"] )
            samples = {"samples": samples}

        # if not refiner, return the result of the sampler step.
        if (refiner_model is None) or (refiner_positive is None) or (refiner_negative is None):
            return (samples )

        # second step: refiner
        kwargs = REFINER_CONFIG[refiner].copy()
        kwargs["model"        ] = refiner_model
        kwargs["seed"         ] = refiner_variation
    #   kwargs["steps"        ] = K!
    #   kwargs["cfg"          ] = K!
    #   kwargs["sampler_name" ] = K!
    #   kwargs["scheduler"    ] = K!
        kwargs["positive"     ] = refiner_positive
        kwargs["negative"     ] = refiner_negative
        kwargs["latent"       ] = samples
        kwargs["denoise"      ] = denoise
        kwargs["disable_noise"] = disable_noise
    #   kwargs["start_step"   ] = K!
    #   kwargs["last_step"    ] = K!
        kwargs["force_full_denoise"] = force_full_denoise # <- no hace falta!
        samples = common_ksampler( **kwargs )

        return (samples, )


