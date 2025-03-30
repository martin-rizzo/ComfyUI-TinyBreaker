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
import comfy.sample
from .tiles   import get_section_2d, overlay_latent, multiply_latent, create_tile_mask
from ..model  import Model
from ..vae    import VAE



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

    # create the latent overlap mask
    # TODO: more testing to determine the best gradient and zero border values
    latent_tile_mask = create_tile_mask(width       = tile_latent_step,
                                        height      = tile_latent_step,
                                        gradient    = tile_padding // 2,
                                        zero_border = tile_padding - (tile_padding // 2),
                                        dim_order   = "bchw"
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
            multiply_latent(latent_tile,
                            xlatent if xlatent<0 else 0,
                            ylatent if ylatent<0 else 0,
                            source = latent_tile_mask
                            )
            overlay_latent(latent_canvas,
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


