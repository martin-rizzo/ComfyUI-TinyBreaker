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
    batch_size, image_height, image_width, channels = image.shape
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

    batch_size, image_height, image_width, channels = image.shape
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
    image_section = image[ : , y:yend , x:xend , : ]

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

def _getXY(xy: int | tuple[int, int]) -> tuple[int, int]:

    if isinstance(xy, int):
        return xy, xy
    elif isinstance(xy, (tuple, list)):
        if len(xy) >= 2:
            return xy[0], xy[1]
        else:
            return xy[0], xy[0]
    elif isinstance(xy, str):
        x, y = xy.split('x', 1)
        return int(x.strip()), int(y.strip())

    raise ValueError("position/size must be an integer or a tuple of two integers")



#=========================== UPSCALE PROTOTYPE 2 ===========================#

def _normalize_dim_order(dim_order: tuple[int, int] | str) -> tuple[int, int]:
    if isinstance(dim_order, str):
        if   dim_order == "bhwc":  return (-2, -3)
        elif dim_order == "bchw":  return (-1, -2)
        else:
            raise ValueError(f'Invalid dim_order value: expected "bchw", "bhwc" or a tuple of integers. Got {dim_order}.')
    return dim_order


def shrink_tensor_2d(tensor: torch.Tensor,
                     left     : int,
                     top      : int,
                     right    : int,
                     bottom   : int, /,
                     dim_order: tuple[int, int] | str = "bchw"
                     ):
    width_dim, height_dim = _normalize_dim_order(dim_order)
    left, top, right, bottom = max(0, left), max(0, top), max(0, right), max(0, bottom)

    # if nothing to shrink, return the original tensor
    if left == 0 and top == 0 and right == 0 and bottom == 0:
        return tensor

    # calculate the new dimensions after shrinking
    new_width  = tensor.shape[width_dim]  - left - right
    new_height = tensor.shape[height_dim] - top  - bottom
    if new_width <= 0 or new_height <= 0:
        return None

    # shrink the tensor
    return tensor.narrow(height_dim, top, new_height).narrow(width_dim , left, new_width)



def get_section_2d(tensor: torch.Tensor,
                   x     : int,
                   y     : int,
                   width : int,
                   height: int, /,
                   dim_order: tuple[int, int] | str = (-2, -3)
                   ) -> torch.Tensor | None:
    """Extracts a section of a 2D tensor (image or latent).

    If the provided section is partially outside of the tensor, it will
    be cropped to fit within the tensor.
    Args:
        tensor: The input tensor (image or latent).
        x     : The x-coordinate of the top-left corner of the section to extract.
        y     : The y-coordinate of the top-left corner of the section to extract.
        width : The width of the section to extract.
        height: The height of the section to extract.
        dim_order: Tuple indicating the position of width/height in the tensor.
                   Default is (-2, -3) for tensors with shape [-, H, W, -].
    Returns:
        A tensor representing the extracted section.
        Returns `None` if the section is completely outside of the tensor.
    """
    width_dim, height_dim = _normalize_dim_order(dim_order)

    # fix rectangle position/size to fit in the tensor
    excess = -x
    if excess > 0:  x = 0 ; width -= excess
    excess = -y
    if excess > 0:  y = 0 ; height -= excess
    excess = (x+width ) - tensor.shape[ width_dim ]
    if excess > 0:  width -= excess
    excess = (y+height) - tensor.shape[ height_dim ]
    if excess > 0:  height -= excess

    # if the rectangle is outside of the tensor, return `None`
    if width<=0 or height<=0:
        return None

    # extract the section from the tensor
    return tensor.narrow(height_dim, y, height).narrow(width_dim, x, width)




def _overlay_latent(dest  : torch.Tensor,
                    x     : int,
                    y     : int,
                    source: torch.Tensor,
                    ) -> None:
    """Overlays the source tensor onto the destination tensor at specified coordinates.

    The function adjusts the source tensor's dimensions to fit within the
    destination tensor's boundaries. It then adds the value of the source
    tensor onto the destination.
    Args:
        dest   (Tensor): The destination tensor to which the source tensor will be overlayed (added)
        x         (int): The x-coordinate of the top-left corner of the section to overlay.
        y         (int): The y-coordinate of the top-left corner of the section to overlay.
        source (Tensor): The source tensor.
    Returns:
        Nothing. The function modifies the `dest` tensor in place.
    """
    sour_x, sour_y = (0,0)
    _, _, sour_height, sour_width = source.shape
    _, _, dest_height, dest_width = dest.shape

    # fix the source size to fit in the destination
    excess = (x+sour_width ) - dest_width
    if excess > 0:  sour_width -= excess
    excess = (y+sour_height) - dest_height
    if excess > 0:  sour_height -= excess

    # fix the position to fit in the destination
    offset = -x
    if offset > 0:  x += offset ; sour_x += offset ; sour_width -= offset
    offset = -y
    if offset > 0:  y += offset ; sour_y += offset ; sour_height -= offset

    # add the source section into the destination
    if sour_width<=0 or sour_height<=0:
        return
    dest[ : , : , y:y+sour_height , x:x+sour_width ] \
        += source[ : , : , sour_y:sour_y+sour_height , sour_x:sour_x+sour_width ]


def _multiply_latent(dest  : torch.Tensor,
                     x     : int,
                     y     : int,
                     source: torch.Tensor,
                     ) -> None:
    """Multiplies the source tensor onto the destination tensor at specified coordinates.

    The function adjusts the source tensor's dimensions to fit within the
    destination tensor's boundaries. It then multiplies the value of the
    source tensor onto the destination.
    Args:
        dest   (Tensor): The destination tensor to which the source tensor will be multiplied.
        x         (int): The x-coordinate of the top-left corner of the section to multiply.
        y         (int): The y-coordinate of the top-left corner of the section to multiply.
        source (Tensor): The source tensor.
    Returns:
        None. The function modifies the `dest` tensor in place.
    """
    sour_x, sour_y = (0,0)
    _, _, sour_height, sour_width = source.shape
    _, _, dest_height, dest_width = dest.shape

    # fix the source size to fit in the destination
    excess = (x+sour_width ) - dest_width
    if excess > 0:  sour_width -= excess
    excess = (y+sour_height) - dest_height
    if excess > 0:  sour_height -= excess

    # fix the position to fit in the destination
    offset = -x
    if offset > 0:  x += offset ; sour_x += offset ; sour_width -= offset
    offset = -y
    if offset > 0:  y += offset ; sour_y += offset ; sour_height -= offset

    # multiply the source section into the destination
    if sour_width<=0 or sour_height<=0:
        return
    dest[ : , : , y:y+sour_height, x:x+sour_width ] \
      *= source[ : , : , sour_y:sour_y+sour_height , sour_x:sour_x+sour_width ]


def _create_lantent_mask(width      : int,
                         height     : int,
                         gradient   : int,
                         zero_border: int
                         ) -> torch.Tensor:
    """Creates a latent mask with gradient and zero borders."""
    #
    #             returned width
    #      |<--------------------------~
    #      :                 width
    #      :             |<------------~
    # 1.0  :             :    __________ 1.0
    #      :             :  /
    #      :             :/
    #      :            /:
    # 0.0   _________ /  :               0.0
    #      |  zero  |grad|
    #
    #                 2x grad
    #               |---------|
    #
    returned_width  = zero_border + gradient + width  + gradient + zero_border
    returned_height = zero_border + gradient + height + gradient + zero_border
    total_gradient  = (gradient * 2) - 1

    # create a mask
    mask = torch.ones((1, 1, returned_height, returned_width), dtype=torch.float32)

    # set a linear gradient for the left and right borders
    for i in range(0, total_gradient+zero_border):
        gradient_level = float(i-zero_border) / total_gradient if i>zero_border else 0.0
        mask[:, :, :,  i  ] = gradient_level
        mask[:, :, :, -i-1] = gradient_level

    # set a linear gradient for the top and bottom borders
    for i in range(0, total_gradient+zero_border):
        gradient_level = float(i-zero_border) / total_gradient if i>=zero_border else 0.0
        mask[:, :,  i  , :] *= gradient_level
        mask[:, :, -i-1, :] *= gradient_level

    return mask


def tiny_encode(image: torch.Tensor,
                /,*,
                vae           : VAE,
                tile_size     : int = 512,
                tile_padding  : int = 128,
                vae_channels  : int = None,
                vae_patch_size: int = None,
                ) -> torch.Tensor:
    """
    Encodes an image into a latent representation using the provided VAE.

    This function divides the input image into overlapping tiles, encodes each
    tile using the provided VAE, and combines the encoded tiles to create the
    complete latent representation.

    Args:
        image            (Tensor): The input image tensor, expected to be in the
                                   format [batch_size, image_height, image_width, channels].
        vae                 (VAE): The Variational Autoencoder to use for encoding the image.
        tile_size      (optional): The size of each tile expressed in pixels.
                                   Defaults to 512.
        tile_padding   (optional): The amount of padding to add around each tile expressed in pixels.
                                   This creates overlap between tiles. Defaults to 128.
        vae_channels   (optional): The number of channels in the VAE's latent space.
                                   Defaults to the VAE's `latent_channels` attribute.
        vae_patch_size (optional): The downscale ratio used by the VAE.
                                   Defaults to the VAE's `downscale_ratio` attribute.
    Returns:
        The latent representation of the image, with the
        shape: [batch_size, vae_channels, latent_height, latent_width].
    """
    assert tile_size >= tile_padding*4, f"Tile size must be larger or equal to 4x tile padding. Got {tile_size} with padding {tile_padding}."
    batch_size, image_height, image_width, _ = image.shape
    vae_channels   = int(vae_channels   or vae.latent_channels) # = 4
    vae_patch_size = int(vae_patch_size or vae.downscale_ratio) # = 8
    latent_width   = int(image_width  // vae_patch_size)
    latent_height  = int(image_height // vae_patch_size)

    # convert sizes (in pixels) to latent sizes
    tile_size      = int( ((tile_size   -1) // vae_patch_size) + 1 )
    tile_padding   = int( ((tile_padding-1) // vae_patch_size) + 1 )
    safe_border    = 2 * tile_padding

    # calculate the tile size and the step beetween tiles
    # (tile size is greater than tile step because the tiles overlap)
    tile_latent_step = int( tile_size )
    tile_latent_size = tile_padding + tile_latent_step + tile_padding
    tile_size        = int( tile_latent_size * vae_patch_size )

    # create the overlap mask
    # TODO: more testing to determine the best gradient and zero border values
    latent_tile_mask = _create_lantent_mask(width       = tile_latent_step,
                                            height      = tile_latent_step,
                                            gradient    = tile_padding // 2,
                                            zero_border = tile_padding - (tile_padding // 2),
                                            )

    # create an empty `latent_canvas` and fill it tile by tile
    latent_canvas = torch.zeros((batch_size, vae_channels, latent_height, latent_width), dtype=torch.float32)
    for ylatent in range(-safe_border, latent_height, tile_latent_step):
        for xlatent in range(-safe_border, latent_width, tile_latent_step):

            # extract a tile from the main image
            tile = get_section_2d(image,
                                   xlatent * vae_patch_size,
                                   ylatent * vae_patch_size,
                                   tile_size, tile_size
                                   )
            if tile is None: continue

            # encode the tile to the latent space
            latent_tile = vae.encode(tile)

            # multiply the tile with the mask and overlay it on the canvas
            # (the mask is used ensure that the tiles overlap each other)
            _multiply_latent(latent_tile,
                             xlatent if xlatent<0 else 0,
                             ylatent if ylatent<0 else 0,
                             source = latent_tile_mask
                             )
            _overlay_latent(latent_canvas,
                            xlatent if xlatent>0 else 0,
                            ylatent if ylatent>0 else 0,
                            source = latent_tile
                            )

    # [batch_size, vae_channels, image_height//vae_patch_size, image_width//vae_patch_size]
    return latent_canvas



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
                        tile_pos  : tuple[int,int]       = None,
                        tile_size : tuple[int,int] | int = None,
                        ) -> torch.Tensor:
    _, _, latent_height, latent_width = latent.shape

    # resolve tile parameters
    if not tile_pos:
        tile_pos  = (0,0)
    if not tile_size:
        tile_size = (latent_width-tile_pos[0], latent_height-tile_pos[1])
    elif isinstance(tile_size, int):
        tile_size = (tile_size,tile_size)

    # extract the tile from the latent image
    # and generate a noise tensor if required
    tile = get_section_2d(latent, tile_pos[0], tile_pos[1], tile_size[0], tile_size[1], dim_order="bchw")
    if add_noise:
        noise = comfy.sample.prepare_noise(tile, noise_seed, None)
    else:
        noise = torch.zeros_like(tile) #, device=latent.device)

    # use the naive ComfyUI sampling function to refine the image
    return comfy.sample.sample_custom(model,
                                      noise,
                                      cfg,
                                      sampler,
                                      sigmas,
                                      positive,
                                      negative,
                                      tile,
                                      noise_mask   = None,
                                      callback     = None,
                                      disable_pbar = True,
                                      seed         = noise_seed)


