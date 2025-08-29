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
import comfy.model_management
import comfy.utils
from spandrel import ImageModelDescriptor
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


def upscale_with_model(image        : torch.Tensor,
                       upscale_model: ImageModelDescriptor
                       ) -> torch.Tensor:
        """
        Upscales an image (or a batch of images) using a specified upscale model

        The scale model should be a valid ImageModelDescriptor object loaded
        with the "Load Upscale Model" native node.

        Args:
            image (torch.Tensor): The image or batch of images to be upscaled.
            upscale_model (ImageModelDescriptor): The upscale model (e.g. ESRGAN)
        Returns:
            The upscaled image (or batch of images).
        """
        # The following code is based on the code from "ImageUpscaleWithModel" comfyui node
        # https://github.com/comfyanonymous/ComfyUI/blob/v0.3.52/comfy_extras/nodes_upscale_model.py#L50

        device = comfy.model_management.get_torch_device()

        memory_required = comfy.model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0 #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
        memory_required += image.nelement() * image.element_size()
        comfy.model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
            except comfy.model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        upscale_model.to("cpu")
        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
        return s


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



