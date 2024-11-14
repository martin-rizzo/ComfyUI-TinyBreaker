"""
File    : checkpoint_loader.py
Purpose : Load PixArt checkpoints for use in ComfyUI.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 14, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from .packs       import Model_pack
from .directories import PIXART_CHECKPOINTS_DIR


class CheckpointLoader:
    TITLE       = "xPixArt | Checkpoint Loader"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Load PixArt checkpoints."

    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (PIXART_CHECKPOINTS_DIR.get_filename_list(), ),
                }
            }
    
    #-- FUNCTION --------------------------------#
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", "VAE", "T5", "STRING")
    RETURN_NAMES = ("MODEL", "VAE", "T5", "META")

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):

        # safetensors_path = PIXART_CHECKPOINTS_DIR.get_full_path(ckpt_name)
        # model = Model_packet_.from_safetensors(safetensors_path, prefix)
        # vae   = VAE_packet_.from_safetensors(safetensors_path, prefix)
        # t5    = T5_packet_.from_safetensors(safetensors_path, prefix)
        # meta  = Meta_packet_.from_predefined("sigma", 2048)
        # return (model, vae, t5, meta)


        safetensors_path = PIXART_CHECKPOINTS_DIR.get_full_path(ckpt_name)
        model_pack = None
        vae_pack   = None
        t5_pack    = None
        meta_pack  = None

        model_pack = Model_pack.from_safetensors(
            safetensors_path,
            prefix = "",
            weight_inplace_update = False
            );

        return (model_pack, vae_pack, t5_pack, meta_pack)
