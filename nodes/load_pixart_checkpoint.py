"""
File    : load_pixart_checkpoint.py
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
import comfy.utils
from .xcomfy.model      import Model
from .utils.directories import PIXART_CHECKPOINTS_DIR


class LoadPixArtCheckpoint:
    TITLE       = "xPixArt | Load PixArt Checkpoint"
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
    FUNCTION = "load_pixart_checkpoint"
    RETURN_TYPES = ("MODEL", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "VAE", "META"  )

    def load_pixart_checkpoint(self, ckpt_name):

        ckpt_path  = PIXART_CHECKPOINTS_DIR.get_full_path(ckpt_name)
        state_dict = comfy.utils.load_torch_file(ckpt_path)

        model = Model.from_state_dict(state_dict, prefix="", resolution=1024)
        vae   = None  # VAE.from_state_dict(state_dict, prefix="")
        meta  = None  # Meta.from_predefined("sigma", 2048)
        return (model, vae, meta)
