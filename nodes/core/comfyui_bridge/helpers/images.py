"""
File    : xcomfy/helpers/images.py
Purpose : Helper functions for manipulating images in ComfyUI
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
import comfy.sample
from ..model import Model


def normalize_images(images: torch.Tensor,
                     /,*,
                     max_channels  : int = 3,
                     max_batch_size: int = None,
                     ) -> torch.Tensor:
    """
    Normalizes a batch of images to default ComfyUI format.

    This function ensures that the input image tensor has a consistent shape
    of [batch_size, height, width, channels].

    Args:
        images           (Tensor): A tensor representing a batch of images.
        max_channels   (optional): The maximum number of color channels allowed. Defaults to 3.
        max_batch_size (optional): The maximum batch size allowed. Defaults to None (no limit).
    Returns:
        A normalized image tensor with shape [batch_size, height, width, channels].
    """
    images_dimension = len(images.shape)

    # if 'images' is a single image, add a batch_size dimension to it
    if images_dimension == 3:
        images = images.unsqueeze(0)

    # if 'images' has more than 4 dimensions,
    # colapse the extra dimensions into the batch_size dimension
    if images_dimension > 4:
        images = images.reshape(-1, *images.shape[-3:])

    if (max_channels is not None) and images.shape[-1] > max_channels:
        images = images[ : , : , : , 0:max_channels ]

    if (max_batch_size is not None) and images.shape[0] > max_batch_size:
        images = images[ 0:max_batch_size , : , : , : ]

    return images


def refine_latent_image(latent: torch.Tensor,
                        /,*,
                        model     : Model,
                        add_noise : bool,
                        noise_seed: int,
                        cfg       : float,
                        sampler   : object,
                        sigmas    : torch.Tensor,
                        positive  : list[ list[torch.Tensor, dict] ],
                        negative  : list[ list[torch.Tensor, dict] ],
                        ) -> torch.Tensor:
    if add_noise:
        noise = comfy.sample.prepare_noise(latent, noise_seed, None)
    else:
        noise = torch.zeros_like(latent) #, device=latent.device)

    # use the native ComfyUI sampling function to refine the image
    return comfy.sample.sample_custom(model,
                                      noise,
                                      cfg,
                                      sampler,
                                      sigmas,
                                      positive,
                                      negative,
                                      latent,
                                      noise_mask   = None,
                                      callback     = None,
                                      disable_pbar = True,
                                      seed         = noise_seed)




