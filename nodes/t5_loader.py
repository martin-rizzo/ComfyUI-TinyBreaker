"""
File    : t5_loader.py
Purpose : Implements a data loader for the T5 model.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 4, 2024
Repo    : https://github.com/martin-rizzo/ComfyUI-xPixArt
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                              ComfyUI-xPixArt
    ComfyUI nodes providing experimental support for PixArt-Sigma model
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
from .packs       import T5_pack
from ..utils.directories import T5_CHECKPOINTS_DIR

class T5Loader:
    TITLE       = "xPixArt | T5 Loader"
    CATEGORY    = "xPixArt"
    DESCRIPTION = "Loads a T5 model checkpoint from T5_CHECKPOINTS_DIR"
   
    #-- PARAMETERS -----------------------------#
    @classmethod
    def INPUT_TYPES(cls):
        devices = ["auto", "cpu", "gpu"]
        for k in range(1, torch.cuda.device_count()):
            devices.append(f"cuda:{k}")
        return {
            "required": {
                "t5_name" : (T5_CHECKPOINTS_DIR.get_filename_list(), ),
                "device"  : (devices, {"default":"cpu"}),
            }
        }

    #-- FUNCTION --------------------------------#
    FUNCTION = "load"
    RETURN_TYPES = ("T5",)

    def load(self, t5_name, device):
        # safetensors_path = T5_CHECKPOINTS_DIR.get_full_path(t5_name)
        # t5_pack = T5_pack.from_safetensors(safetensors_path, prefix, device)
        # return (t5_pack,)

        t5_pack = T5_pack(T5_CHECKPOINTS_DIR.get_full_path(t5_name),
                          device = device
                          )
        return (t5_pack,)
