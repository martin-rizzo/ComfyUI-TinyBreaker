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
import comfy.utils


def normalize_images(images  : torch.Tensor,
                     channels: int = 3
                     ) -> torch.Tensor:
    """
    Normalizes a batch of images to default ComfyUI format.

    This function ensures that the input image tensor has a consistent shape of
    [batch_size, width, height, channels] with the specified number of color channels.

    Args:
        images  (Tensor): A tensor representing a batch of images.
        channels   (int): Number of color channels in each image. Default is 3 (RGB)
    Returns:
        A normalized image tensor with shape [batch_size, width, height, channels].
    """
    assert channels == 3 or channels == 4, "Only RGB and RGBA are supported but 'channels' parameter is " + str(channels)
    assert images.shape[-1] <= 4         , "The channels dimension must be the last dimension in the 'images' tensor"
    images_dimension = len(images.shape)

    # if 'images' is a single image, add a batch_size dimension to it
    if images_dimension == 3:
        images = images.unsqueeze(0)

    # if 'images' has more than 4 dimensions,
    # colapse the extra dimensions into the batch_size dimension
    if images_dimension > 4:
        images = images.reshape(-1, *images.shape[-3:])

    # if 'images' has more than 3 channels, remove the extra channels
    if images.shape[-1] > channels:
        images = images[:, :, :, :channels]

    return images


def upscale_images(images: torch.Tensor,
                   width : int,
                   height: int,
                   *,
                   method: str = "nearest",
                   crop  : str = None
                   ) -> torch.Tensor:
    """
    Upscales a batch of images to a specified width and height.

    Input and output image tensors are expected to have a shape of:
    [batch_size, width, height, channels]

    Args:
        images (Tensor): A tensor containing the images to upscale.
        width     (int): The desired width of the upscaled images.
        height    (int): The desired height of the upscaled images.
        method    (str): The upscaling method to use. Available methods are:
                         'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear',
                         'area', 'nearest-exact', 'lanczos' and 'bislerp'.
                         Defaults to "nearest".
        crop (optional): The cropping method to apply after upscaling. Available options are:
                         None (no cropping) and "center" (crops the center portion).
                         Defaults to None.
    Returns:
        A tensor containing the upscaled images.  Shape is (B, C, height, width).
    """
    # ATTENTION: for some reason common_upscale(..) expects 'images'
    # to have a shape of [batch_size, channels, height, width] (??)
    images = images.transpose(1, -1)
    images = comfy.utils.common_upscale(images, width, height, upscale_method=method, crop=crop)
    images = images.transpose(1, -1)
    return images
