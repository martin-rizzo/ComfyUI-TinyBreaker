"""
File    : load_checkpoint.py
Purpose : Node to load PixArt checkpoints.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 14, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .xcomfy.model      import Model
from .utils.directories import PIXART_CHECKPOINTS_DIR


class LoadCheckpoint:
    TITLE       = "xPixArt | Load Checkpoint"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Load PixArt checkpoints."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (PIXART_CHECKPOINTS_DIR.get_filename_list(), {"tooltip": "The PixArt checkpoint to load."}),
                }
            }
    
    #-- FUNCTION --------------------------------#
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "VAE", "META"  )

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):

        # model = Model.from_safetensors(safetensors_path, prefix)
        # vae   = VAE.from_safetensors(safetensors_path, prefix)
        # meta  = Meta.from_predefined("sigma", 2048)
        # return (model, vae, meta)

        safetensors_path = PIXART_CHECKPOINTS_DIR.get_full_path(ckpt_name)
        model = None
        vae   = None
        meta  = None

        model = Model.from_safetensors(
            safetensors_path,
            prefix = "",
            weight_inplace_update = False
            );

        return (model, vae, meta)
