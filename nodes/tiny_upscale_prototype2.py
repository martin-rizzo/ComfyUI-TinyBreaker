"""
File    : tiny_upscale_prototype2.py
Purpose : Node to upscale an image using an experimental method (Prototype 2)
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 24, 2025
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
import comfy.sample
import comfy.samplers
from .xcomfy.helpers.sigmas import calculate_sigmas
from .xcomfy.helpers.images import normalize_images, tiny_encode, refine_latent_image, shrink_tensor_2d
from .xcomfy.vae            import VAE
from .xcomfy.model          import Model


class TinyUpscalePrototype2:
    TITLE       = "💪TB | Tiny Upscale (Prototype 2)"
    CATEGORY    = "TinyBreaker"
    DESCRIPTION = "Upscale an image using an experimental method."

    #__ PARAMETERS ________________________________________
    @classmethod
    def INPUT_TYPES(cls):
        return {
        "required": {
            "image"     :("IMAGE"        ,{"tooltip": "The image to upscale.",
                                          }),
            "model"    :("MODEL"         ,{"tooltip": "The model to use for the upscale.",
                                          }),
            "vae"       :("VAE"          ,{"tooltip": "The VAE to use for the upscale.",
                                          }),
            "positive" :("CONDITIONING"  ,{"tooltip": "The positive conditioning to use for the upscale.",
                                          }),
            "negative" :("CONDITIONING"  ,{"tooltip": "The negative conditioning to use for the upscale.",
                                          }),
            "seed"     :("INT"           ,{"tooltip": "The random seed used for creating the noise.",
                                           "default": 0, "min": 0, "max": 0xffffffffffffffff,
                                           "control_after_generate": True,
                                          }),
            "steps"    :("INT"           ,{"tooltip": "The number of steps used in the denoising process.",
                                           "default": 5, "min": 1, "max": 50,
                                          }),
            "cfg"      :("FLOAT"         , {"tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
                                            "default": 3.0, "min": 0.0, "max": 15.0, "step":0.1, "round": 0.01,
                                           }),
            "sampler"  :(cls._samplers() , {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output.",
                                            "default": "dpmpp_2m",
                                           }),
            "scheduler":(cls._schedulers(),{"tooltip": "The scheduler controls how noise is gradually removed to form the image.",
                                            "default": "karras",
                                           }),
            "strength" :("FLOAT"         , {"tooltip": "The strength of the denoising process. Higher values result in more alterations to the image.",
                                            "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.02,
                                           }),
            "extra_noise":("FLOAT"       ,{"tooltip": "The amount of extra noise to add to the image during the denoising process.",
                                           "default": 0.6, "min": 0.0, "max": 1.5, "step":0.1,
                                           }),
            "upscale_by":("FLOAT"        ,{"tooltip": "The upscale factor.",
                                           "default": 2.0, "min": 1.0, "max": 5.0, "step":0.5,
                                          }),
            },
        }

    #__ FUNCTION __________________________________________
    FUNCTION = "upscale"
    RETURN_TYPES    = ("LATENT",)
    RETURN_NAMES    = ("latent",)
    OUTPUT_TOOLTIPS = ("The upscaled image in latent space.",)

    def upscale(self,
                image     : torch.Tensor,
                model     : Model,
                vae       : VAE,
                positive  : list[ list[torch.Tensor, dict] ],
                negative  : list[ list[torch.Tensor, dict] ],
                seed      : int,
                steps     : int,
                cfg       : float,
                sampler   : str,
                scheduler : str,
                strength  : float,
                extra_noise: float,
                upscale_by: float
                ):
        image = normalize_images(image)
        batch_size, image_height, image_width, channels = image.shape
        tile_size        = 1024
        interpolate_mode = "bilinear" # "nearest" # "bilinear"

        # calculate sigmas
        total_steps    = int(steps / strength)
        steps_start    = total_steps - steps
        model_sampling = model.get_model_object("model_sampling")
        sigmas = calculate_sigmas(model_sampling, sampler, scheduler, total_steps, steps_start)

        # get sampler object
        sampler = comfy.samplers.sampler_object(sampler)



        # upscale the image using simple bilinear interpolation
        upscaled_width  = int( round(image_width  * upscale_by) )
        upscaled_height = int( round(image_height * upscale_by) )
        upscaled_image  = F.interpolate(image.transpose(1,-1),
                                        size = (upscaled_width, upscaled_height),
                                        mode = interpolate_mode).transpose(1,-1)

        # encode the image into latent space
        upscaled_latent = tiny_encode(upscaled_image,
                                      vae          = vae,
                                      tile_size    = tile_size,
                                      tile_padding = (tile_size/4),
                                      )

        # add extra noise to the latent image if requested
        if extra_noise > 0.0:
            upscaled_latent += torch.randn_like(upscaled_latent) * extra_noise

        for step in range(len(sigmas)-1):

            # build the function (lambda) for refining tiles of the latent image
            #     latent       : the latent image to refine
            #     x, y         : the coordinates of the tile to refine
            #     width, height: the size of the tile to refine
            refine_tile = lambda latent, x, y, width, height: \
                refine_latent_image(latent,
                                    model      = model,
                                    add_noise  = (step==0),
                                    noise_seed = seed,
                                    cfg        = cfg,
                                    sampler    = sampler,
                                    sigmas     = torch.Tensor( [sigmas[step], sigmas[step+1]] ),
                                    positive   = positive,
                                    negative   = negative,
                                    tile_pos   = (x,y),
                                    tile_size  = (width,height),
                                    )

            # process the latent image in tiles using the refinement function,
            # alternating between top-left to bottom-right and
            # bottom-right to top-left directions
            if step % 2 == 0:
                apply_tiles_tlbr(upscaled_latent,
                                 create_tile_func = refine_tile,
                                 tile_size        = int(tile_size//8))
            else:
                apply_tiles_brtl(upscaled_latent,
                                 create_tile_func = refine_tile,
                                 tile_size        = int(tile_size//8))

        return ({"samples":upscaled_latent}, )


    #__ internal functions ________________________________

    @staticmethod
    def _samplers():
        return comfy.samplers.KSampler.SAMPLERS


    @staticmethod
    def _schedulers():
        return comfy.samplers.KSampler.SCHEDULERS





def apply_tiles_tlbr(canvas: torch.Tensor,
                     /,*,
                     create_tile_func: callable,
                     tile_size       : int,
                     overlap         : int = None,
                     discard         : int = None,
                     ) -> None:
    """
    Applies tiles to a canvas in Top-Left to Bottom-Right order.

    This function iterates, creating tiles of a specified size and applying them
    to the canvas. It supports overlapping regions between tiles and allows for
    discarding portions of tiles near the edges.

    Args:
        canvas            : The canvas to which the tiles will be applied.
        create_tile_func  : A function that takes (canvas, x, y, tile width, tile_height)
                            as input, and returns a tensor representing the tile to be
                            applied at that position.
        tile_size         : The size of the tiles
        overlap (optional): The amount of overlap between adjacent tiles. Defaults to tile_size // 8.
        discard (optional): The amount of discardable region around the edges of the tiles. Defaults to tile_size // 8.

    Returns:
        This function modifies the input `canvas` tensor in-place and does not return any value.
    """
    if overlap is None:  overlap = (tile_size // 8)
    if discard is None:  discard = (tile_size // 8)
    tile_step   = tile_size - overlap
    max_valid_x = canvas.shape[-1] - tile_size
    max_valid_y = canvas.shape[-2] - tile_size

    last_row_bottom = 0
    for y in range(0, max_valid_y+tile_step, tile_step):
        y              = min(y, max_valid_y)
        discard_bottom = min(discard, max_valid_y - y)

        last_column_right = 0
        for x in range(0, max_valid_x+tile_step, tile_step):
            x             = min(x, max_valid_x)
            discard_right = min(discard, max_valid_x - x)

            # create the tile for the current position
            tile = create_tile_func(canvas, x, y, tile_size+discard_right, tile_size+discard_bottom)
            assert tile.shape[-1] == tile_size+discard_right and tile.shape[-2] == tile_size+discard_bottom, \
                "Invalid tile size returned by 'create_tile_func' function. ['create_tile_func' is a parameter of apply_tiles_tlbr(..)]"

            # remove overlapping & discardable regions from the created tile
            overlap_left = max(0, last_column_right - x)
            overlap_top  = max(0, last_row_bottom   - y)
            tile = shrink_tensor_2d(tile,
                                    overlap_left, overlap_top,
                                    discard_right, discard_bottom,
                                    dim_order="bchw")

            # apply the generated tile to the canvas at (x,y)
            if tile is not None:
                canvas[ : , : , y+overlap_top:y+tile_size , x+overlap_left:x+tile_size ] = tile

            last_column_right = (x + tile_size)

        last_row_bottom = (y + tile_size)


def apply_tiles_brtl(canvas: torch.Tensor,
                     /,*,
                     create_tile_func: callable,
                     tile_size       : int,
                     overlap         : int = None,
                     discard         : int = None,
                     ) -> None:
    """
    Applies tiles to a canvas in Bottom-Right to Top-Left order.

    This function iterates, creating tiles of a specified size and applying them
    to the canvas. It supports overlapping regions between tiles and allows for
    discarding portions of tiles near the edges.

    Args:
        canvas            : The canvas to which the tiles will be applied.
        create_tile_func  : A function that takes (canvas, x, y, tile width, tile_height)
                            as input, and returns a tensor representing the tile to be
                            applied at that position.
        tile_size         : The size of the tiles
        overlap (optional): The amount of overlap between adjacent tiles. Defaults to tile_size // 8.
        discard (optional): The amount of discardable region around the edges of the tiles. Defaults to tile_size // 8.

    Returns:
        This function modifies the input `canvas` tensor in-place and does not return any value.
    """
    if overlap is None:  overlap = (tile_size // 8)
    if discard is None:  discard = (tile_size // 8)
    canvas_width  = canvas.shape[-1]
    canvas_height = canvas.shape[-2]
    tile_step     = tile_size - overlap
    max_valid_x   = canvas_width  - tile_size
    max_valid_y   = canvas_height - tile_size

    last_row_top = canvas_height
    for y in range(max_valid_y, -tile_step, -tile_step):
        y           = max(0, y)
        discard_top = min(discard, y)

        last_column_left = canvas_width
        for x in range(max_valid_x, -tile_step, -tile_step):
            x            = max(0, x)
            discard_left = min(discard, x)

            # create the tile for the current position
            tile = create_tile_func(canvas, x-discard_left, y-discard_top, discard_left+tile_size, discard_top+tile_size)
            assert tile.shape[-1] == discard_left+tile_size and tile.shape[-2] == discard_top+tile_size, \
                "Invalid tile size returned by 'create_tile_func' function. ['create_tile_func' is a parameter of apply_tiles_brtl(..)]"

            # remove overlapping & discardable regions from the created tile
            overlap_right  = (x + tile_size) - last_column_left
            overlap_bottom = (y + tile_size) - last_row_top
            tile = shrink_tensor_2d(tile,
                                    discard_left, discard_top,
                                    overlap_right, overlap_bottom,
                                    dim_order="bchw")

            # apply the generated tile to the canvas at (x,y)
            if tile is not None:
                canvas[ : , : , y:y+tile_size-overlap_bottom , x:x+tile_size-overlap_right ] = tile

            last_column_left = x

        last_row_top = y
