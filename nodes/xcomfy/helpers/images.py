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
import torch.nn.functional               as F
import torchvision.transforms.functional as TF
import comfy.utils
import comfy.sample
from ..model                             import Model
from ..vae                               import VAE


def normalize_images(images: torch.Tensor,
                     /,*,
                     max_channels  : int = 3,
                     max_batch_size: int = None,
                     ) -> torch.Tensor:
    """
    Normalizes a batch of images to default ComfyUI format.

    This function ensures that the input image tensor has a consistent shape
    of [batch_size, width, height, channels].

    Args:
        images           (Tensor): A tensor representing a batch of images.
        max_channels   (optional): The maximum number of color channels allowed. Defaults to 3.
        max_batch_size (optional): The maximum batch size allowed. Defaults to None (no limit).
    Returns:
        A normalized image tensor with shape [batch_size, width, height, channels].
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
        A tensor containing the upscaled images in the standard ComfyUI
        shape: [batch_size, width, height, channels]
    """
    # ATTENTION: for some reason common_upscale(..) expects 'images'
    # to have a shape of [batch_size, channels, height, width] (??)
    images = images.transpose(1,-1)
    images = comfy.utils.common_upscale(images, width, height, upscale_method=method, crop=crop)
    images = images.transpose(1,-1)
    return images


def gaussian_blur(images: torch.Tensor,
                  /,*,
                  sigma : float
                  ) -> torch.Tensor:
    """
    Returns a batch of images with Gaussian blur applied.
    Args:
        images  (Tensor): The input images tensor, in the standard ComfyUI shape.
        sigma   (float) : The standard deviation of the Gaussian.
    Returns:
        A tensor containing the blurred images int standard ComfyUI
        shape: [batch_size, width, height, channels].
    """
    if sigma <= 0.0:
        return images

    # calculate the kernel_size for the given sigma
    # the result should always be odd and equal to or greater than 3
    unrounded_kernel_size = ((sigma - 0.8) / 0.3 + 1) * 2 + 1
    kernel_size           = int(unrounded_kernel_size)
    if kernel_size < unrounded_kernel_size:
        kernel_size += 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)

    # apply gaussian blur using the calculated kernel_size and sigma
    return TF.gaussian_blur(images.transpose(1,-1), kernel_size=kernel_size, sigma=sigma).transpose(1,-1)


def prepare_image_for_superresolution(image: torch.Tensor,
                                      /,*,
                                      patch_size: int | tuple[int, int],
                                      upscale_by: float,
                                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares an input image for super-resolution processing.

    Args:
        image     : The input image tensor, in the standard ComfyUI shape.
        upscale_by: The scaling factor by which the image will be upscaled.
        patch_size: The size of the patches.
                    It's highly recommended that this matches the VAE block size.
                    If an integer is provided, it's assumed to be the square patch
                    size (patch_width == patch_height).
    Returns:
        A tuple containing the prepared image and the low-frequency guide image.
        The prepared image is cropped to dimensions that are multiples of the patch size.
        Both images are in the standard ComfyUI shape: [batch_size, width, height, channels]
    """
    image                     = normalize_images(image, max_channels=3, max_batch_size=1)
    patch_width, patch_height = _getXY(patch_size)
    gaussian_blur_sigma       = 2.0 # scale_by * 0.1
    batch_size, image_width, image_height, channels = image.shape
    assert batch_size == 1  , f"Superresolution only supports batch size of 1. Got: {batch_size}"
    assert channels   == 3  , f"Superresolution only supports RGB (3 channels) images. Got: {channels}"
    assert upscale_by >= 1.0, f"Upscale factor must be greater than or equal to 1. Got: {upscale_by}"

    # calculate upscaled dimensions that are multiples of `patch_size`
    upscaled_width  = int( round(image_width * upscale_by) // patch_width  * patch_width  )
    upscaled_height = int( round(image_height* upscale_by) // patch_height * patch_height )
    new_width       = min( int( round(upscaled_width  / upscale_by) ), image_width  )
    new_height      = min( int( round(upscaled_height / upscale_by) ), image_height )

    # crop image to new dimensions
    if (new_width != image_width) or (new_height != image_height):
        start_x = (image_width  - new_width ) // 2
        start_y = (image_height - new_height) // 2
        prepared_image = image[ : , start_x:start_x+new_width , start_y:start_y+new_height , : ]
    else:
        prepared_image = image

    # create the low-frequency guide image using gaussian blur
    lowfreq_guide_image = gaussian_blur(prepared_image, sigma=gaussian_blur_sigma)

    return prepared_image, lowfreq_guide_image, upscaled_width, upscaled_height


def generate_superresolution_image(image: torch.Tensor,
                                   /,*,
                                   patch_size: int | tuple[int, int],
                                   upscaled_width : int,
                                   upscaled_height: int,

                                   model   : Model,
                                   vae     : VAE,
                                   seed    : int,
                                   cfg     : float,
                                   sampler : object,
                                   sigmas  : torch.Tensor,
                                   positive: list[ list[torch.Tensor, dict] ],
                                   negative: list[ list[torch.Tensor, dict] ],

                                   tile_pos           : tuple[int, int] = None,
                                   tile_size          : tuple[int, int] = None,
                                   lowfreq_guide_image: torch.Tensor    = None,
                                   lowfreq_guide_ratio: float           = 0.0,

                                   ) -> torch.Tensor:

    batch_size, image_width, image_height, channels = image.shape
    assert batch_size == 1, f"Superresolution only supports batch size of 1. Got: {batch_size}"
    assert channels   == 3, f"Superresolution only supports RGB (3 channels) images. Got: {channels}"

    # if tile is not provided, generate a single tile that covers the entire image
    if tile_pos is None:
        tile_pos = (0, 0)
    if tile_size is None:
        tile_size = (upscaled_width-tile_pos[0], upscaled_height-tile_pos[1])

    # extract the section of the image that will be upscaled (tile)
    x    = int( tile_pos[0] * image_width  / upscaled_width  )
    y    = int( tile_pos[1] * image_height / upscaled_height )
    xend = int( (tile_pos[0] + tile_size[0]) * image_width  / upscaled_width  )
    yend = int( (tile_pos[1] + tile_size[1]) * image_height / upscaled_height )
    image_section = image[ : , x:xend , y:yend , : ]

    # generate the tile upscaling the image_section
    tile = torch.nn.functional.interpolate(image_section.transpose(1,-1),
                                           size = tile_size,
                                           mode = "bilinear").transpose(1,-1)

    noise_level = 0.8  # adjust this value to control the amount of noise

    # apply diffusion to the tile
    latent_tile = vae.encode(tile)

    # add noise to the tile
    latent_tile = latent_tile + torch.randn_like(latent_tile) * noise_level

    noise       = comfy.sample.prepare_noise(latent_tile, seed, None)
    latent_tile = comfy.sample.sample_custom(model,
                                             noise,
                                             cfg,
                                             sampler,
                                             sigmas,
                                             positive,
                                             negative,
                                             latent_tile,
                                             noise_mask   = None,
                                             callback     = None,
                                             disable_pbar = True,
                                             seed         = seed)
    tile = vae.decode(latent_tile)

    return normalize_images( tile )



def _getXY(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    elif isinstance(size, (tuple, list)):
        if len(size) >= 2:
            return size[0], size[1]
        else:
            return size[0], size[0]
    elif isinstance(size, str):
        x, y = size.split('x', 1)
        return int(x.strip()), int(y.strip())

    raise ValueError("position/size must be an integer or a tuple of two integers")

