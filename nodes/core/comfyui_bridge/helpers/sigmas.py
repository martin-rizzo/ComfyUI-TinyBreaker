"""
File    : xcomfy/helpers/sigmas.py
Purpose : Helper functions for manipulating sigma values in ComfyUI
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 21, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
from comfy.samplers  import KSampler, calculate_sigmas as comfy_calculate_sigmas


# Sampler names that will automatically discard the penultimate sigma value:
#  - "dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"
DISCARD_PENULTIMATE_SIGMA_SAMPLERS = KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS

def calculate_sigmas(model_sampling           : object,
                     sampler                  : str,
                     scheduler                : str,
                     steps                    : int,
                     steps_start              : int = 0,
                     steps_end                : int = 10000,
                     steps_nfactor            : int = 0,
                     discard_penultimate_sigma: bool | None = None,
                     ) -> torch.Tensor:
    """
    Calculates sigma values for a given model, sampler and scheduler across a specified range of steps.
    Args:
        model_sampling   (object): The sampling object for the model.
        sampler            (str) : The name of the sampler to use (e.g., "euler", "heun")
        scheduler          (str) : The name of the scheduler to use (e.g., "karras")
        steps              (int) : The total number of steps for which to calculate sigma values.
        steps_start    (optional): The starting step within the calculated range. Defaults to 0.
        steps_end      (optional): The ending step within the calculated range.
                                   Negative values are counted from the total number of steps (e.g.: -2 means `steps-2`)
        steps_nfactor  (optional): A custom value to adjust the number of denoising steps;
                                   positive values will increase the number of denoising steps,
                                   negative values will decrease it.
        discard_penultimate_sigma (optional): Whether to discard the penultimate sigma value.
                                              If not provided, it will be inferred from the sampler name.
    Returns:
        A tensor containing the sigma values for each step.
    """
    assert isinstance(sampler      , str), f"sampler must be a string. Got {type(sampler)}"
    assert isinstance(steps        , int), f"steps must be an integer. Got {type(steps)}"
    assert isinstance(steps_start  , int), f"steps_start must be an integer. Got {type(steps_start)}"
    assert isinstance(steps_end    , int), f"steps_end must be an integer. Got {type(steps_end)}"
    assert isinstance(steps_nfactor, int), f"steps_nfactor must be an integer. Got {type(steps_nfactor)}"
    if steps_end < 0:
        steps_end = steps + steps_end   # fixes negative values to count from end of total number of steps
    steps_start = max(0,   steps_start) # `steps_start` cannot be less than zero
    steps_end   = min(steps_end, steps) # `steps_end`   cannot be greater than steps

    # if discard_penultimate_sigma is not explicitly provided,
    # infer it from the sampler name
    if discard_penultimate_sigma is None:
        discard_penultimate_sigma = sampler in DISCARD_PENULTIMATE_SIGMA_SAMPLERS

    # 'steps_nfactor' modulates the number of steps. A positive value signifies
    # a proportional increase, while a negative value indicates a proportional
    # decrease in the total steps.
    # Note: The specific effect is context-dependent and may require careful adjustment.
    if steps_nfactor:
        if steps_nfactor > 0:
            expand_start = min( steps_nfactor, steps_start )
            expand_end   = steps_nfactor - expand_start
        elif steps_nfactor < 0:
            expand_start = steps_nfactor
            expand_end   = 0
        steps_start = max(0, steps_start - expand_start)
        steps_end   = max(0, steps_end   + expand_end  )
        steps       = max(0, steps       + expand_end  )

    # if no steps to execute, just return an empty tensor
    # (this is useful when no sampling should be performed)
    if steps == 0 or steps_start >= steps_end:
        return torch.FloatTensor([])

    # if discard was requested,
    # first calculate all sigmas (plus one) and then discard the penultimate one
    if discard_penultimate_sigma:
        sigmas = comfy_calculate_sigmas(model_sampling, scheduler, steps+1).cpu()
        sigmas = torch.cat((sigmas[:-2], sigmas[-1:]))
    else:
        sigmas = comfy_calculate_sigmas(model_sampling, scheduler, steps).cpu()

    # the original code from ComfyUI uses this adjustment in some parts,
    # no idea if it's necessary, but I'll leave it just in case
    sigmas = sigmas[-( steps+1 ):]

    # discard all sigmas outside of the specified range
    sigmas = sigmas[steps_start:steps_end+1]
    return sigmas


def split_sigmas(sigmas: torch.Tensor,
                 step  : int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Splits a tensor of sigmas into two parts based on a given step.

    Args:
        sigmas: The input tensor of sigmas to be split.
        step  : The index where the tensor will be split.
    Returns:
        A tuple containing two tensors:
         - The first tensor contains elements from the beginning up
           toand including the 'step' index.
         - The second tensor contains elements from 'step' index to
           the end.
    """
    first_part  = sigmas[:step + 1]
    second_part = sigmas[step:]
    return (first_part, second_part)

