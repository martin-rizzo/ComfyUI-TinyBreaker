"""
File    : functions/tiny_encode_decode.py
Purpose : Functions for encoding/decoding images to/from the latent space.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Apr 8, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
from ..xcomfy.vae import VAE
from .tiles import create_tile_mask, get_tile, multiply_tile, overlay_tile
_IMAGE_DIM_ORDER  = "bhwc"
_LATENT_DIM_ORDER = "bchw"



def tiny_encode(image: torch.Tensor,
                /,*,
                vae            : VAE,
                tile_size      : int = 512,
                tile_padding   : int = 128,
                vae_patch_size : int = None,
                latent_channels: int = None,
                ) -> torch.Tensor:
    """
    Encodes an image into a latent representation using the provided VAE.

    This function divides the input image into overlapping tiles, encodes each
    tile using the provided VAE, and combines the encoded tiles to create the
    complete latent representation.

    Args:
        image             (Tensor): The input image tensor, expected to be in the
                                    format [batch_size, image_height, image_width, channels].
        vae                  (VAE): The Variational Autoencoder to use for encoding the image.
        tile_size       (optional): The size of each tile expressed in pixels.
                                    Defaults to 512.
        tile_padding    (optional): The amount of padding to add around each tile expressed in pixels.
                                    This creates overlap between tiles. Defaults to 128.
        vae_patch_size  (optional): The downscale ratio used by the VAE.
                                    Defaults to the VAE's `downscale_ratio` attribute.
        latent_channels (optional): The number of channels in the VAE's latent space.
                                    Defaults to the VAE's `latent_channels` attribute.
    Returns:
        The latent representation of the image, with the shape:
          [batch_size, latent_channels, latent_height, latent_width].
    """
    assert tile_size >= tile_padding*4, f"Tile size must be larger or equal to 4x tile padding. Got {tile_size} with padding {tile_padding}."
    batch_size, image_height, image_width, _ = image.shape
    latent_channels = int(latent_channels or vae.latent_channels) # = 4
    vae_patch_size  = int(vae_patch_size  or vae.downscale_ratio) # = 8
    latent_width    = int(image_width  // vae_patch_size)
    latent_height   = int(image_height // vae_patch_size)

    # convert tile size/padding from pixels to latent size
    tile_latent_step    = int( ((tile_size   -1) // vae_patch_size) + 1 )
    tile_latent_padding = int( ((tile_padding-1) // vae_patch_size) + 1 )
    tile_latent_size    = tile_latent_padding + tile_latent_step + tile_latent_padding
    safe_border = 2 * tile_latent_padding

    # create the latent overlap mask
    latent_tile_mask = create_tile_mask(width       = tile_latent_step,
                                        height      = tile_latent_step,
                                        gradient    = (tile_latent_padding // 2),
                                        zero_border = tile_latent_padding - (tile_latent_padding // 2),
                                        dim_order   = _LATENT_DIM_ORDER
                                        )

    # create an empty `latent_canvas` and fill it tile by tile
    latent_canvas = torch.zeros((batch_size, latent_channels, latent_height, latent_width), dtype=torch.float32)
    for ylatent in range(-safe_border, latent_height, tile_latent_step):
        for xlatent in range(-safe_border, latent_width, tile_latent_step):

            # extract a tile from the main image
            tile = get_tile(image,
                            xlatent          * vae_patch_size,
                            ylatent          * vae_patch_size,
                            tile_latent_size * vae_patch_size,
                            tile_latent_size * vae_patch_size,
                            dim_order = _IMAGE_DIM_ORDER
                            )
            if tile is None: continue

            # encode the tile to the latent space
            latent_tile = vae.encode(tile)

            # multiply the tile with the mask and overlay it on the canvas
            # (the mask is used ensure that the tiles overlap each other)
            multiply_tile(latent_tile,
                          xlatent if xlatent<0 else 0,
                          ylatent if ylatent<0 else 0,
                          source    = latent_tile_mask,
                          dim_order = _LATENT_DIM_ORDER
                          )
            overlay_tile(latent_canvas,
                         xlatent if xlatent>0 else 0,
                         ylatent if ylatent>0 else 0,
                         source    = latent_tile,
                         dim_order = _LATENT_DIM_ORDER
                         )

    # [batch_size, latent_channels, image_height//vae_patch_size, image_width//vae_patch_size]
    return latent_canvas



def tiny_decode(latent: torch.Tensor,
                /,*,
                vae           : VAE,
                tile_size     : int = 512,
                tile_padding  : int = 128,
                vae_patch_size: int = None,
                image_channels: int = None,
                ) -> torch.Tensor:
    """
    Decodes a latent representation back into an image using the provided VAE.

    This function divides the latent representation into overlapping tiles,
    decodes each tile using the provided VAE, and combines the decoded tiles
    to create the complete image.

    Args:
        latent           (Tensor): The input latent representation tensor, expected to be in the
                                   format [batch_size, latent_channels, latent_height, latent_width].
        vae                 (VAE): The Variational Autoencoder to use for decoding the latent representation.
        tile_size      (optional): The size of each tile expressed in image pixels (not latent)
                                   Defaults to 512.
        tile_padding   (optional): The amount of padding to add around each tile expressed in image pixels.
                                   This creates overlap between tiles. Defaults to 128.
        vae_patch_size (optional): The upscale ratio used by the VAE.
                                   Defaults to the VAE's `upscale_ratio` attribute.
        image_channels (optional): The number of channels in the output image.
                                   Defaults to 3 (RGB).
    Returns:
        The decoded image, with the shape:
          [batch_size, image_height, image_width, image_channels].
    """
    assert tile_size >= tile_padding*4, f"Tile size must be larger or equal to 4x tile padding. Got {tile_size} with padding {tile_padding}."
    batch_size, _, latent_height, latent_width = latent.shape
    vae_patch_size = int(vae_patch_size or vae.upscale_ratio) # = 8
    image_width    = int(latent_width  * vae_patch_size)
    image_height   = int(latent_height * vae_patch_size)
    image_channels = int(image_channels or 3) # default to RGB

    # convert tile size/padding from pixels to latent size
    tile_latent_step    = int( ((tile_size   -1) // vae_patch_size) + 1 )
    tile_latent_padding = int( ((tile_padding-1) // vae_patch_size) + 1 )
    tile_latent_size    = tile_latent_padding + tile_latent_step + tile_latent_padding
    safe_border = 2 * tile_latent_padding

    # create the tile overlap mask
    tile_mask_padding = tile_latent_padding * vae_patch_size
    tile_mask         = create_tile_mask(width       = (tile_latent_step * vae_patch_size),
                                         height      = (tile_latent_step * vae_patch_size),
                                         gradient    = (tile_mask_padding // 2),
                                         zero_border = tile_mask_padding - (tile_mask_padding // 2),
                                         dim_order = _IMAGE_DIM_ORDER
                                         )

    # create an empty `image_canvas` and fill it tile by tile
    image_canvas = torch.zeros((batch_size, image_height, image_width, image_channels), dtype=torch.float32)
    for ylatent in range(-safe_border, latent_height, tile_latent_step):
        for xlatent in range(-safe_border, latent_width, tile_latent_step):

            # extract a tile from the latent
            latent_tile = get_tile(latent,
                                   xlatent,
                                   ylatent,
                                   tile_latent_size,
                                   tile_latent_size,
                                   dim_order = _LATENT_DIM_ORDER
                                   )
            if latent_tile is None: continue

            # decode the tile to the image space
            tile = vae.decode(latent_tile)

            # multiply the tile with the mask and overlay it on the canvas
            # (the mask is used ensure that the tiles overlap each other)
            multiply_tile(tile,
                          (xlatent * vae_patch_size) if xlatent<0 else 0,
                          (ylatent * vae_patch_size) if ylatent<0 else 0,
                          source    = tile_mask,
                          dim_order = _IMAGE_DIM_ORDER
                          )
            overlay_tile(image_canvas,
                         (xlatent * vae_patch_size) if xlatent>0 else 0,
                         (ylatent * vae_patch_size) if ylatent>0 else 0,
                         source    = tile,
                         dim_order = _IMAGE_DIM_ORDER
                         )

    # [batch_size, latent_height*vae_patch_size, latent_width*vae_patch_size, image_channels]
    return image_canvas

