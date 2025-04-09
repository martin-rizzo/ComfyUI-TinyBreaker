"""
File    : _denoising_params.py
Purpose : A class representing all the parameters required for denoising an image.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 20, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import comfy.samplers
from .genparams        import GenParams, normalize_prefix
from ..xcomfy.helpers.sigmas import calculate_sigmas
from ..utils.system          import logger


class DenoisingParams:
    """
    A class representing all the parameters required for denoising an image.

    This class encapsulates all necessary parameters required for any sampler to
    denoise an image. It provides a convenient way to store and access these
    parameters, as well as the ability to create this object using information
    contained in a `GenParams` instance.

    Args:
        positive        (str): The positive prompt.
        negative        (str): The negative prompt.
        sampler         (str): The name of the sampler to use.
        scheduler       (str): The scheduler to use.
        steps           (int): The number of denoising steps.
        steps_start     (int): The starting step for denoising.
        steps_end       (int): The ending step for denoising.
        steps_nfactor   (int): A custom value to adjust the number of denoising steps.
        sigmas (torch.Tensor): The sigma values for denoising.
        cfg           (float): The classifier-free guidance (CFG) scale.
        noise_seed      (int): The seed to use for noise generation in the start of the denoising process.
        model        (object): The model to use for denoising (used to calculate sigmas).
        discard_penultimate_sigma (bool): Whether to discard the penultimate sigma value.)
    """
    def __init__(self,
                 *,# keyword-only arguments #
                 positive     : str          = None,
                 negative     : str          = None,
                 sampler      : str          = None,
                 scheduler    : str          = None,
                 steps        : int          = None,
                 steps_start  : int          = None,
                 steps_end    : int          = None,
                 steps_nfactor: int          = None,
                 sigmas       : torch.Tensor = None,
                 cfg          : float        = None,
                 noise_seed   : int          = None,
                 model        : object       = None,
                 discard_penultimate_sigma: bool = None,
                 ):
        """
        Initialize a DenoisingParams object with the given parameters.
        """
        # sanitize inputs
        positive      = str(positive)      if positive      is not None else ""
        negative      = str(negative)      if negative      is not None else ""
        sampler       = str(sampler)       if sampler       is not None else "euler"
        scheduler     = str(scheduler)     if scheduler     is not None else "normal"
        steps         = int(steps)         if steps         is not None else 12
        steps_start   = int(steps_start)   if steps_start   is not None else 0
        steps_end     = int(steps_end)     if steps_end     is not None else 10000
        steps_nfactor = int(steps_nfactor) if steps_nfactor is not None else 0
        cfg           = float(cfg)         if cfg           is not None else 3.5
        noise_seed    = int(noise_seed)    if noise_seed    is not None else 1

        # if sigmas is not provided, calculate it from the model
        if not isinstance(sigmas, torch.Tensor):
            if model:
                sigmas = calculate_sigmas(model.get_model_object("model_sampling"),
                                          sampler,
                                          scheduler,
                                          steps,
                                          steps_start,
                                          steps_end,
                                          steps_nfactor,
                                          discard_penultimate_sigma)
            else:
                logger.warning("sigmas could not be calculated because the model was not provided")
                sigmas = torch.tensor([1.0])

        # set all attributes of this object
        self.positive       = positive
        self.negative       = negative
        self.sampler        = sampler
        self.sampler_object = comfy.samplers.sampler_object(sampler)
        self.scheduler      = scheduler
        self.steps          = steps
        self.steps_start    = steps_start
        self.steps_end      = steps_end
        self.steps_nfactor  = steps_nfactor
        self.sigmas         = sigmas
        self.cfg            = cfg
        self.noise_seed     = noise_seed
        self.discard_penultimate_sigma = discard_penultimate_sigma


    @classmethod
    def from_genparams(cls,
                       genparams      : GenParams,
                       prefix         : str,
                       model_to_sample: object
                       ) -> 'DenoisingParams':
        """
        Create a DenoisingParams object from the information contained in a `GenParams` instance.
        Args:
            genparams      : The GenParams instance to extract the information from.
            prefix         : The prefix to use to access the information in `genparams`.
            model_to_sample: The model that will be used for image generation.
        """
        prefix = normalize_prefix(prefix)
        return cls(positive      = genparams.get( f"{prefix}prompt"        ),
                   negative      = genparams.get( f"{prefix}negative"      ),
                   sampler       = genparams.get( f"{prefix}sampler"       ),
                   scheduler     = genparams.get( f"{prefix}scheduler"     ),
                   steps         = genparams.get( f"{prefix}steps"         ),
                   steps_start   = genparams.get( f"{prefix}steps_start"   ),
                   steps_end     = genparams.get( f"{prefix}steps_end"     ),
                   steps_nfactor = genparams.get( f"{prefix}steps_nfactor" ),
                   cfg           = genparams.get( f"{prefix}cfg"           ),
                   noise_seed    = genparams.get( f"{prefix}noise_seed"    ),
                   model         = model_to_sample)


    def __str__(self):
        """Return a string representation of the DenoisingParams object."""
        string = "DenoisingParams(\n"
        for key, value in self.__dict__.items():
            if isinstance(value, str):
                value = f'"{value}"'
            string += f"    {key:13} = {value}\n"
        string += ")"
        return string


