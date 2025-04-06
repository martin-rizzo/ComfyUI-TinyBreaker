"""
File    : xcomfy/helpers/tiles.py
Purpose : Helper functions to process tensors by dividing them into tiles
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 29, 2025
Repo    : https://github.com/martin-rizzo/ComfyUI-TinyBreaker
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-TinyBreaker
 ComfyUI nodes for experimenting with the capabilities of the TinyBreaker model.
  (TinyBreaker is a hybrid model that combines the strengths of PixArt and SD)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch


def get_tile(tensor: torch.Tensor,
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
    width_dim, height_dim = _extract_wh_indices(dim_order)

    # fix rectangle position/size to fit in the tensor
    excess = -x
    if excess > 0:  x = 0 ; width -= excess
    excess = -y
    if excess > 0:  y = 0 ; height -= excess
    excess = (x+width ) - tensor.shape[ width_dim ]
    if excess > 0:  width -= excess
    excess = (y+height) - tensor.shape[ height_dim ]
    if excess > 0:  height -= excess

    # if the rectangle is completely outside of the tensor, return `None`
    if width<=0 or height<=0:
        return None

    # extract the section from the tensor
    return tensor.narrow(height_dim, y, height).narrow(width_dim, x, width)



def overlay_tile(dest: torch.Tensor,
                 x   : int,
                 y   : int,
                 /,*,
                 source: torch.Tensor,
                 dim_order: str = "bchw"
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
        dim_order      : The order of dimensions. Defaults to "bchw".
    Returns:
        Nothing. The function modifies the `dest` tensor in place.
    """
    width_dim, height_dim = _extract_wh_indices(dim_order)

    # get the source/destination sizes
    sour_x, sour_y = (0,0)
    sour_width, sour_height = source.shape[width_dim], source.shape[height_dim]
    dest_width, dest_height =   dest.shape[width_dim],   dest.shape[height_dim]

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

    # if the overlay section is out of the destination,
    # return without doing anything
    if sour_width<=0 or sour_height<=0:
        return

    # add the source section into the destination
    if width_dim == -1 and height_dim == -2:
        dest[ : , : , y:y+sour_height , x:x+sour_width ] \
            += source[ : , : , sour_y:sour_y+sour_height , sour_x:sour_x+sour_width ]
    elif width_dim == -2 and height_dim == -3:
        dest[ : , y:y+sour_height , x:x+sour_width , : ] \
            += source[ : , sour_y:sour_y+sour_height , sour_x:sour_x+sour_width , : ]
    else:
        raise ValueError("Invalid dim_order: " + dim_order)



def multiply_tile(dest: torch.Tensor,
                  x   : int,
                  y   : int,
                  /,*,
                  source: torch.Tensor,
                  dim_order: str = "bchw"
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
        dim_order      : The order of dimensions. Defaults to "bchw".
    Returns:
        None. The function modifies the `dest` tensor in place.
    """
    width_dim, height_dim = _extract_wh_indices(dim_order)

    # get the source/destination sizes
    sour_x, sour_y = (0,0)
    sour_width, sour_height = source.shape[width_dim], source.shape[height_dim]
    dest_width, dest_height =   dest.shape[width_dim],   dest.shape[height_dim]

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

    # if the multiply section is out of the destination,
    # return without doing anything
    if sour_width<=0 or sour_height<=0:
        return

    # multiply the source section into the destination
    if width_dim == -1 and height_dim == -2:
        dest[ : , : , y:y+sour_height , x:x+sour_width ] \
            *= source[ : , : , sour_y:sour_y+sour_height , sour_x:sour_x+sour_width ]
    elif width_dim == -2 and height_dim == -3:
        dest[ : , y:y+sour_height , x:x+sour_width , : ] \
            *= source[ : , sour_y:sour_y+sour_height , sour_x:sour_x+sour_width , : ]
    else:
        raise ValueError("Invalid dim_order: " + dim_order)



def create_tile_mask(width      : int,
                     height     : int,
                     gradient   : int,
                     zero_border: int,
                     dim_order: tuple[int, int] | str = (-1, -2)
                     ) -> torch.Tensor:
    """Creates a tile mask with gradient and zero borders to be used to blend tiles"""
    width_dim, height_dim = _extract_wh_indices(dim_order)
    #
    #             returned width
    #      |<--------------------------~
    #      :                 width
    #      :             |<------------~
    # 1.0  :             :    __________ 1.0
    #      :             :  / :
    #      :             :/   :
    #      :            /:    :
    # 0.0   _________ /  :    :          0.0
    #      |  zero  |grad|    :
    #               :         :
    #               : 2x grad :
    #               |---------|
    #
    mask_shape = [1, 1, 1, 1]
    mask_shape[ width_dim] = zero_border + gradient + width  + gradient + zero_border
    mask_shape[height_dim] = zero_border + gradient + height + gradient + zero_border
    total_gradient = (gradient * 2) - 1

    # create the mask with the correct shape and fill it with ones
    mask = torch.ones(mask_shape, dtype=torch.float32)

    # set a linear gradient for the left and right borders
    for i in range(0, total_gradient+zero_border):
        gradient_level = float(i-zero_border) / total_gradient if i>zero_border else 0.0
        mask.narrow(width_dim,  i  , 1).fill_(gradient_level)
        mask.narrow(width_dim, -i-1, 1).fill_(gradient_level)

    # set a linear gradient for the top and bottom borders
    for i in range(0, total_gradient+zero_border):
        gradient_level = float(i-zero_border) / total_gradient if i>=zero_border else 0.0
        mask.narrow(height_dim, i  , 1).mul_(gradient_level)
        mask.narrow(height_dim,-i-1, 1).mul_(gradient_level)

    return mask



def shrink_tile(tile  : torch.Tensor,
                left  : int,
                top   : int,
                right : int,
                bottom: int,
                /,*,
                dim_order: tuple[int, int] | str = "bchw"
                ) -> torch.Tensor | None:
    """Shrinks a tile by a specified amount from each side.

    Args:
        tile   (Tensor): The tile tensor to shrink.
        left      (int): The number of elements to remove from the left side.
        top       (int): The number of elements to remove from the top side.
        right     (int): The number of elements to remove from the right side.
        bottom    (int): The number of elements to remove from the bottom side.
        dim_order: The order of dimensions. Defaults to "bchw".
    Returns:
        The shrunk tile tensor.
        Returns `None` if the resulting tensor has no elements.
    """
    width_dim, height_dim = _extract_wh_indices(dim_order)
    left, top, right, bottom = max(0, left), max(0, top), max(0, right), max(0, bottom)

    # if nothing to shrink, return the original tile
    if left == 0 and top == 0 and right == 0 and bottom == 0:
        return tile

    # calculate the new dimensions after shrinking
    new_width  = tile.shape[ width_dim] - left - right
    new_height = tile.shape[height_dim] - top  - bottom
    if new_width <= 0 or new_height <= 0:
        return None

    # shrink the tile by removing elements from each side
    return tile.narrow(height_dim, top, new_height).narrow(width_dim , left, new_width)



#================= FUNCTIONS TO APPLY MOSAICS ON A CANVAS ==================#

def apply_tiles_tlbr(canvas: torch.Tensor,
                     /,*,
                     create_tile_func: callable,
                     tile_size       : int,
                     overlap         : int = None,
                     discard         : int = None,
                     progress_bar: tuple = None,
                     ) -> None:
    """Applies tiles to a canvas in Top-Left to Bottom-Right order.

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

    last_row_bottom = 0
    for y in range(0, max_valid_y+tile_step, tile_step):
        y              = min(y, max_valid_y)
        discard_bottom = min(discard, max_valid_y - y)
        if progress_bar:
            _pbar = progress_bar[0]
            _pbar.update_absolute( progress_bar[1] + int(progress_bar[2] * y / canvas_height) )

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
            tile = shrink_tile(tile,
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
                     progress_bar : tuple = None,
                     ) -> None:
    """Applies tiles to a canvas in Bottom-Right to Top-Left order.

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
        if progress_bar:
            _pbar = progress_bar[0]
            _pbar.update_absolute( progress_bar[1] + int(progress_bar[2] * (canvas_height-y) / canvas_height) )

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
            tile = shrink_tile(tile,
                               discard_left, discard_top,
                               overlap_right, overlap_bottom,
                               dim_order="bchw")

            # apply the generated tile to the canvas at (x,y)
            if tile is not None:
                canvas[ : , : , y:y+tile_size-overlap_bottom , x:x+tile_size-overlap_right ] = tile

            last_column_left = x

        last_row_top = y



#============================ INTERNAL HELPERS =============================#

def _extract_wh_indices(dim_order: str | tuple[int, int]) -> tuple[int, int]:
    """Interprets the dimension order string or tuple.
    Args:
        dim_order: A string representing the dimension order ("bchw", "bhwc", etc)
                   or a tuple of two integers representing the width and height
                   dimensions indices.
    Returns:
        A tuple of two integers representing the width and height dim indices in the tensors.
    """
    if isinstance(dim_order, str):
        if   dim_order[-2:  ] == "hw":  return (-1, -2)
        elif dim_order[-3:-1] == "hw":  return (-2, -3)
    if isinstance(dim_order, tuple) and len(dim_order) == 2:
        return dim_order
    raise ValueError(f'Invalid dim_order value: expected "bchw", "bhwc" or a tuple of integers. Got {dim_order}.')

