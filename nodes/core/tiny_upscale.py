"""
File    : functions/tiny_upscale.py
Purpose : Functions to perform image upscaling using any model through an experimental method
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
import torch.nn.functional as F
from .tiles                         import get_tile, apply_tiles_tlbr, apply_tiles_brtl
from .tiny_encode_decode            import tiny_encode, tiny_decode
from .comfyui_bridge.model          import Model
from .comfyui_bridge.vae            import VAE
from .comfyui_bridge.helpers.sigmas import calculate_sigmas
from .comfyui_bridge.helpers.images import normalize_images, refine_latent_image


import comfy.samplers
import comfy.utils

def tiny_upscale(image             : torch.Tensor,
                 model             : Model,
                 vae               : VAE,
                 positive          : list[ list[torch.Tensor, dict] ],
                 negative          : list[ list[torch.Tensor, dict] ],
                 seed              : int,
                 steps             : int,
                 start_at_step     : int,
                 end_at_step       : int,
                 cfg               : float,
                 sampler           : str,
                 scheduler         : str,
                 extra_noise       : float,
                 upscale_by        : float,
                 tile_size         : int = 1024,
                 overlap_percent   : int = 100,
                 interpolation_mode: str = "bilinear" # "nearest"
                 ):

    sampler = comfy.samplers.sampler_object(sampler)
    image   = normalize_images(image)
    _, image_height, image_width, _ = image.shape

    # calculate sigmas (old)
    # total_steps    = int(steps / strength)
    # steps_start    = total_steps - steps
    # steps_end      = 10000
    # model_sampling = model.get_model_object("model_sampling")
    # sigmas         = calculate_sigmas(model_sampling, sampler, scheduler, total_steps, steps_start, steps_end)

    # calculate sigmas (new)
    model_sampling = model.get_model_object("model_sampling")
    sigmas         = calculate_sigmas(model_sampling, sampler, scheduler, steps, start_at_step, end_at_step)
    sigmas         = sigmas[:-1]

    # upscale the image using simple interpolation
    upscaled_width  = int( round(image_width  * upscale_by) )
    upscaled_height = int( round(image_height * upscale_by) )
    upscaled_image  = F.interpolate(image.transpose(1,-1),
                                    size = (upscaled_width, upscaled_height),
                                    mode = interpolation_mode).transpose(1,-1)

    # encode the image into latent space
    upscaled_latent = tiny_encode(upscaled_image,
                                    vae          = vae,
                                    tile_size    = tile_size,
                                    tile_padding = (tile_size*overlap_percent//400),
                                    )

    # add extra noise to the latent image if requested
    if extra_noise > 0.0:
        upscaled_latent += torch.randn_like(upscaled_latent) * extra_noise


    number_of_steps = len(sigmas)-1
    pbar, pstep     = comfy.utils.ProgressBar(1000), int(1000 / number_of_steps)
    for step in range(number_of_steps):
        progress_bar = (pbar,step*pstep,pstep)

        # a lambda function that creates a refined tile from the latent image
        #     latent       : the latent image where the tile will be extracted from
        #     x, y         : the coordinates of the tile
        #     width, height: the size of the tile
        create_refined_tile = lambda latent, x, y, width, height: \
            _create_refined_tile(latent, x, y, width, height,
                                 model      = model,
                                 add_noise  = (step==0),
                                 noise_seed = seed,
                                 cfg        = cfg,
                                 sampler    = sampler,
                                 sigmas     = torch.Tensor( [sigmas[step], sigmas[step+1]] ),
                                 positive   = positive,
                                 negative   = negative,
                                 )

        # process the latent image in tiles using the refinement function,
        # alternating between top-left to bottom-right and
        # bottom-right to top-left directions
        if step % 2 == 0:
            apply_tiles_tlbr(upscaled_latent,
                             create_tile_func = create_refined_tile,
                             tile_size        = int(tile_size//8),
                             progress_bar     = progress_bar)
        else:
            apply_tiles_brtl(upscaled_latent,
                             create_tile_func = create_refined_tile,
                             tile_size        = int(tile_size//8),
                             progress_bar     = progress_bar)

    upscaled_image = tiny_decode(upscaled_latent, vae=vae)
    return upscaled_image


def _create_refined_tile(latent: torch.Tensor,
                         x     : int,
                         y     : int,
                         width : int ,
                         height: int,
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

    # extract the tile from the latent image and refine it
    tile = get_tile(latent, x, y, width, height, dim_order="bchw")
    if tile is None:
        return None
    return refine_latent_image(tile,
                               model      = model,
                               add_noise  = add_noise,
                               noise_seed = noise_seed,
                               cfg        = cfg,
                               sampler    = sampler,
                               sigmas     = sigmas,
                               positive   = positive,
                               negative   = negative
                               )


